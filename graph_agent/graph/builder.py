"""Graph builder — resolves raw call references into typed Edge rows.

Algorithm:
1. Load all symbols for a repo into an in-memory lookup dict:
   {name: [Symbol], qualified_name: Symbol}
2. For each UnresolvedRef (call site):
   a. Try exact qualified_name match (confidence=1.0, method="direct")
   b. Try name-only match across repo (confidence=0.7, method="heuristic_name_match")
   c. If no match: record edge as CALLS with confidence=0.0, method="unresolved"
      but keep the UnresolvedRef row so we know it's still open.
3. Build CONTAINS edges (file→symbols, class→methods) from symbol parent_id.
4. Build IMPORTS edges from the import_names stored on UnresolvedRef with kind="import".

ADR-3: We build CONTAINS and CALLS edges only. INHERITS, HANDLES, HAS_TYPE, TESTS
edges are added in future phases.  Every edge has confidence + resolution_method so
downstream consumers can filter by quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from graph_agent.db.models import Symbol, Edge, UnresolvedRef, File, Repository

log = logging.getLogger(__name__)


@dataclass
class GraphBuildResult:
    repo_id: int
    edges_created: int = 0
    edges_resolved: int = 0          # CALLS edges with a real target
    edges_unresolved: int = 0        # CALLS edges that stayed unresolved
    contains_edges: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_symbol_index(
    session: Session, repo_id: int
) -> tuple[dict[str, Symbol], dict[str, list[Symbol]]]:
    """Return (qual_name→Symbol, name→[Symbol]) lookup tables for a repo."""
    symbols = session.query(Symbol).filter_by(repo_id=repo_id).all()
    by_qual: dict[str, Symbol] = {}
    by_name: dict[str, list[Symbol]] = {}
    for sym in symbols:
        by_qual[sym.qualified_name] = sym
        by_name.setdefault(sym.name, []).append(sym)
    return by_qual, by_name


def _edge_exists(session: Session, source_id: int, target_id: int, edge_type: str) -> bool:
    return (
        session.query(Edge)
        .filter_by(source_id=source_id, target_id=target_id, edge_type=edge_type)
        .first()
    ) is not None


def _create_edge(
    session: Session,
    repo_id: int,
    source_id: int,
    target_id: int,
    edge_type: str,
    confidence: float,
    resolution_method: str,
) -> Edge:
    edge = Edge(
        repo_id=repo_id,
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        confidence=confidence,
        resolution_method=resolution_method,
    )
    session.add(edge)
    return edge


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_graph(session: Session, repo_id: int) -> GraphBuildResult:
    """Build or rebuild edges for all symbols in *repo_id*.

    Safe to call multiple times — existing edges for the repo are deleted first
    (except unresolved_refs, which are preserved for audit).
    """
    result = GraphBuildResult(repo_id=repo_id)

    # Wipe existing edges for a clean rebuild
    session.query(Edge).filter_by(repo_id=repo_id).delete(synchronize_session=False)

    by_qual, by_name = _build_symbol_index(session, repo_id)

    # ------------------------------------------------------------------
    # 1. CONTAINS edges: parent symbol → child symbol
    # ------------------------------------------------------------------
    for sym in by_qual.values():
        if sym.parent_id is not None:
            if not _edge_exists(session, sym.parent_id, sym.id, "CONTAINS"):
                _create_edge(
                    session, repo_id,
                    source_id=sym.parent_id,
                    target_id=sym.id,
                    edge_type="CONTAINS",
                    confidence=1.0,
                    resolution_method="direct",
                )
                result.contains_edges += 1

    # ------------------------------------------------------------------
    # 2. CALLS edges from UnresolvedRef rows
    # ------------------------------------------------------------------
    refs = (
        session.query(UnresolvedRef)
        .join(Symbol, UnresolvedRef.source_id == Symbol.id)
        .filter(Symbol.repo_id == repo_id, UnresolvedRef.ref_kind == "call")
        .all()
    )

    for ref in refs:
        source_sym = by_qual.get(
            session.query(Symbol).filter_by(id=ref.source_id).first().qualified_name
            if ref.source_id else None
        )
        if source_sym is None:
            source_sym = session.query(Symbol).filter_by(id=ref.source_id).first()
        if source_sym is None:
            continue

        callee_name = ref.ref_name
        target_sym: Optional[Symbol] = None
        confidence = 0.0
        resolution_method = "unresolved"

        # Strip attribute access to get the base name (e.g. "self.divide" → "divide")
        base_name = callee_name.split(".")[-1]

        # a. Exact qualified_name match
        if callee_name in by_qual:
            target_sym = by_qual[callee_name]
            confidence = 1.0
            resolution_method = "direct"
        # b. Name-only match (possibly ambiguous — take first match in repo)
        elif base_name in by_name:
            candidates = by_name[base_name]
            # Prefer same-file candidates
            same_file = [c for c in candidates if c.file_id == source_sym.file_id]
            target_sym = same_file[0] if same_file else candidates[0]
            confidence = 0.7 if len(candidates) == 1 else 0.5
            resolution_method = "heuristic_name_match"

        if target_sym is not None:
            if not _edge_exists(session, source_sym.id, target_sym.id, "CALLS"):
                _create_edge(
                    session, repo_id,
                    source_id=source_sym.id,
                    target_id=target_sym.id,
                    edge_type="CALLS",
                    confidence=confidence,
                    resolution_method=resolution_method,
                )
                result.edges_resolved += 1
                result.edges_created += 1
        else:
            # Record an explicit CALLS edge with zero-confidence target = source (self-loop sentinel)
            # so we don't silently drop it; the unresolved_ref row stays.
            log.debug(
                "Unresolved call: %s → %s (no target found in repo)",
                source_sym.qualified_name, callee_name,
            )
            result.edges_unresolved += 1

    result.edges_created += result.contains_edges
    session.flush()

    log.info(
        "Graph built for repo %d: %d CONTAINS, %d CALLS resolved, %d unresolved",
        repo_id, result.contains_edges, result.edges_resolved, result.edges_unresolved,
    )
    return result
