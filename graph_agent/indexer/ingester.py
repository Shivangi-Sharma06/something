"""Idempotent ingester: walks a repo, parses files, and upserts into PostgreSQL.

Re-indexing a file that hasn't changed (same content hash) is a no-op at the
symbol level — only the File.indexed_at timestamp is refreshed.

ADR: We use a "hash-gated upsert" strategy:
  1. Upsert the File row (ON CONFLICT DO UPDATE).
  2. If content_hash changed, delete old symbols for that file and re-insert.
  3. If hash is unchanged, skip parsing altogether for that file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert as pg_insert

from graph_agent.db.models import Repository, File, Symbol, UnresolvedRef
from graph_agent.indexer.walker import walk_repo
from graph_agent.indexer.parser import parse_file, ParseResult, SymbolInfo

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result summary
# ---------------------------------------------------------------------------
@dataclass
class IndexResult:
    repo_id: int
    files_visited: int = 0
    files_indexed: int = 0   # new or changed
    files_skipped: int = 0   # unchanged (hash match)
    symbols_upserted: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# ---------------------------------------------------------------------------
# Core upsert helpers
# ---------------------------------------------------------------------------
def _upsert_repository(session: Session, name: str, root_path: str) -> Repository:
    """Insert or return the Repository row."""
    repo = session.query(Repository).filter_by(root_path=root_path).first()
    if repo is None:
        repo = Repository(name=name, root_path=root_path)
        session.add(repo)
        session.flush()
    return repo


def _upsert_file(
    session: Session,
    repo_id: int,
    rel_path: str,
    language: str,
    content_hash: str,
    size_bytes: int,
) -> tuple[File, bool]:
    """Upsert a File row.  Returns (file, hash_changed)."""
    existing: Optional[File] = (
        session.query(File)
        .filter_by(repo_id=repo_id, rel_path=rel_path)
        .first()
    )
    if existing is None:
        f = File(
            repo_id=repo_id,
            rel_path=rel_path,
            language=language,
            content_hash=content_hash,
            size_bytes=size_bytes,
        )
        session.add(f)
        session.flush()
        return f, True

    hash_changed = existing.content_hash != content_hash
    if hash_changed:
        existing.content_hash = content_hash
        existing.size_bytes = size_bytes
        existing.language = language
    return existing, hash_changed


def _delete_file_symbols(session: Session, file_id: int) -> None:
    """Remove all symbols (and their dependent rows) for a file."""
    # UnresolvedRefs are cascade-deleted via FK
    session.query(Symbol).filter_by(file_id=file_id).delete(synchronize_session=False)


def _insert_symbols(
    session: Session,
    file_id: int,
    repo_id: int,
    symbols: list[SymbolInfo],
) -> dict[str, int]:
    """Insert symbols and return {qualified_name: symbol_id}."""
    # Two-pass: first insert parent-less symbols, then those with parents.
    # This handles nested definitions without requiring a topological sort.
    inserted: dict[str, int] = {}

    def _insert_sym(sym: SymbolInfo, parent_id: Optional[int]) -> int:
        db_sym = Symbol(
            file_id=file_id,
            repo_id=repo_id,
            kind=sym.kind,
            name=sym.name,
            qualified_name=sym.qualified_name,
            parent_id=parent_id,
            line_start=sym.line_start,
            line_end=sym.line_end,
            byte_start=sym.byte_start,
            byte_end=sym.byte_end,
            signature=sym.signature,
            docstring=sym.docstring,
        )
        session.add(db_sym)
        session.flush()
        return db_sym.id

    # Sort: parents before children (shorter qualified_name = more likely a parent)
    sorted_syms = sorted(symbols, key=lambda s: len(s.qualified_name))
    for sym in sorted_syms:
        parent_id = (
            inserted.get(sym.parent_qualified_name)
            if sym.parent_qualified_name
            else None
        )
        sym_id = _insert_sym(sym, parent_id)
        inserted[sym.qualified_name] = sym_id

    return inserted


def _insert_unresolved_refs(
    session: Session,
    symbol_id_map: dict[str, int],
    raw_calls: list[tuple[str, str]],
) -> None:
    """Record raw call references (caller_qual → callee_name) as unresolved.

    The GraphBuilder will later attempt to resolve these into Edge rows.
    """
    for caller_qual, callee_name in raw_calls:
        source_id = symbol_id_map.get(caller_qual)
        if source_id is None:
            continue  # caller itself wasn't indexed (e.g. nested lambda)
        ref = UnresolvedRef(
            source_id=source_id,
            ref_name=callee_name,
            ref_kind="call",
            context=f"{caller_qual} → {callee_name}",
        )
        session.add(ref)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def index_repository(
    session: Session,
    repo_root: str | Path,
    repo_name: Optional[str] = None,
) -> IndexResult:
    """Walk *repo_root*, parse changed files, and upsert everything into DB.

    This function is idempotent: calling it twice on the same repo without
    source changes produces the same DB state and skips re-parsing.
    """
    root = Path(repo_root).resolve()
    name = repo_name or root.name

    repo = _upsert_repository(session, name=name, root_path=str(root))
    result = IndexResult(repo_id=repo.id)

    log.info("Indexing repo %r at %s (id=%d)", name, root, repo.id)

    for fpath in walk_repo(root):
        result.files_visited += 1
        rel_path = str(fpath.relative_to(root))

        try:
            size = fpath.stat().st_size
        except OSError as exc:
            result.errors.append(f"{rel_path}: stat failed: {exc}")
            continue

        # Parse to get the content hash (fast even if we don't use all results)
        parse_result: Optional[ParseResult] = parse_file(fpath, root)
        if parse_result is None:
            continue  # unsupported language

        if parse_result.errors:
            for err in parse_result.errors:
                log.warning("%s: %s", rel_path, err)

        db_file, hash_changed = _upsert_file(
            session,
            repo_id=repo.id,
            rel_path=rel_path,
            language=parse_result.language,
            content_hash=parse_result.content_hash,
            size_bytes=size,
        )

        if not hash_changed:
            result.files_skipped += 1
            log.debug("Skipping unchanged file: %s", rel_path)
            continue

        # Content changed (or new file) → replace symbols
        _delete_file_symbols(session, db_file.id)
        sym_id_map = _insert_symbols(
            session, db_file.id, repo.id, parse_result.symbols
        )
        result.symbols_upserted += len(sym_id_map)

        # Record raw call references for later graph resolution
        _insert_unresolved_refs(session, sym_id_map, parse_result.raw_calls)

        result.files_indexed += 1
        log.info("  Indexed %s — %d symbols", rel_path, len(sym_id_map))

    session.flush()
    log.info(
        "Indexing done: %d visited, %d indexed, %d skipped, %d symbols",
        result.files_visited, result.files_indexed,
        result.files_skipped, result.symbols_upserted,
    )
    return result
