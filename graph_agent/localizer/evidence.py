"""Evidence extractor — maps bug reports / stack traces / test names to symbols.

Returns a list of EvidenceItem objects, each tagged as KNOWN (directly mentioned
in the bug evidence) vs HEURISTIC (inferred).

KNOWN sources:
- Stack frame file+line mentions
- Failing test function name
- Explicit function name mentions in bug description

HEURISTIC sources (Phase 3 only):
- Import chain proximity
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from graph_agent.db.models import Symbol


class EvidenceKind(str, Enum):
    KNOWN = "KNOWN"         # directly mentioned in bug evidence
    HEURISTIC = "HEURISTIC"  # inferred via graph proximity


@dataclass
class EvidenceItem:
    symbol_id: int
    qualified_name: str
    kind: EvidenceKind
    source: str       # e.g. "stack_frame", "failing_test", "description_mention"
    weight: float     # initial score boost from this evidence piece


# ---------------------------------------------------------------------------
# Stack trace parser
# ---------------------------------------------------------------------------
# Matches Python traceback lines like:
#   File "app/routers/auth.py", line 42, in login
_FRAME_RE = re.compile(
    r'File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+(\S+)'
)
# Also match bare "function_name (file:line)" style
_SIMPLE_FRAME_RE = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+\(([^:)]+):(\d+)\)')


def _parse_stack_frames(stack_trace: str) -> list[tuple[str, int, str]]:
    """Return list of (rel_file, lineno, function_name) from a Python traceback."""
    frames = []
    for m in _FRAME_RE.finditer(stack_trace):
        rel_file = m.group(1).strip().lstrip("./")
        lineno = int(m.group(2))
        func_name = m.group(3)
        frames.append((rel_file, lineno, func_name))
    return frames


def _parse_test_name(failing_test: str) -> list[str]:
    """Extract function name(s) from a failing test string.

    Accepts formats like:
      - test_divide_by_zero
      - tests/test_utils.py::test_divide_by_zero
      - TestCalc::test_divide_by_zero
    """
    names = []
    # Last segment after :: or /
    for part in re.split(r'[/:]+', failing_test):
        part = part.strip()
        if part and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', part):
            names.append(part)
    return names


# ---------------------------------------------------------------------------
# Symbol lookup helpers
# ---------------------------------------------------------------------------
def _find_by_name(session: Session, repo_id: int, name: str) -> list[Symbol]:
    return (
        session.query(Symbol)
        .filter_by(repo_id=repo_id, name=name)
        .all()
    )


def _find_by_file_line(
    session: Session, repo_id: int, rel_path_fragment: str, lineno: int
) -> list[Symbol]:
    """Find symbols whose source range overlaps *lineno* in a file matching *rel_path_fragment*."""
    rows = session.execute(
        text("""
            SELECT s.id, s.qualified_name, s.kind
            FROM graph_agent.symbols s
            JOIN graph_agent.files f ON f.id = s.file_id
            WHERE s.repo_id = :repo_id
              AND f.rel_path LIKE :path_like
              AND s.line_start <= :lineno
              AND s.line_end   >= :lineno
            ORDER BY (s.line_end - s.line_start) ASC  -- prefer tightest enclosing scope
        """),
        {
            "repo_id": repo_id,
            "path_like": f"%{rel_path_fragment}%",
            "lineno": lineno,
        },
    ).mappings().all()
    return [session.query(Symbol).filter_by(id=r["id"]).first() for r in rows if r["id"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_evidence(
    session: Session,
    repo_id: int,
    description: str = "",
    stack_trace: str = "",
    failing_test: str = "",
) -> list[EvidenceItem]:
    """Map all available bug evidence to repository symbols.

    Returns a deduplicated list of EvidenceItem sorted by weight descending.
    """
    items: dict[int, EvidenceItem] = {}  # symbol_id → best item

    def _add(sym: Symbol, kind: EvidenceKind, source: str, weight: float):
        if sym is None:
            return
        existing = items.get(sym.id)
        if existing is None or existing.weight < weight:
            items[sym.id] = EvidenceItem(
                symbol_id=sym.id,
                qualified_name=sym.qualified_name,
                kind=kind,
                source=source,
                weight=weight,
            )

    # ------------------------------------------------------------------
    # 1. Stack trace — KNOWN evidence, highest weight
    # ------------------------------------------------------------------
    if stack_trace:
        frames = _parse_stack_frames(stack_trace)
        for rel_file, lineno, func_name in frames:
            # Try file+line first (most precise)
            syms = _find_by_file_line(session, repo_id, rel_file, lineno)
            for sym in syms[:1]:  # tightest match only
                _add(sym, EvidenceKind.KNOWN, "stack_frame_line", weight=1.0)

            # Also add by name match (covers cases where lineno drifts)
            for sym in _find_by_name(session, repo_id, func_name):
                _add(sym, EvidenceKind.KNOWN, "stack_frame_name", weight=0.9)

    # ------------------------------------------------------------------
    # 2. Failing test name — KNOWN evidence
    # ------------------------------------------------------------------
    if failing_test:
        for name in _parse_test_name(failing_test):
            for sym in _find_by_name(session, repo_id, name):
                _add(sym, EvidenceKind.KNOWN, "failing_test", weight=0.9)

    # ------------------------------------------------------------------
    # 3. Description keyword scan — HEURISTIC (lower weight)
    # ------------------------------------------------------------------
    if description:
        # Find all potential identifiers in the description
        idents = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b', description)
        for ident in set(idents):
            for sym in _find_by_name(session, repo_id, ident):
                _add(sym, EvidenceKind.HEURISTIC, "description_mention", weight=0.3)

    return sorted(items.values(), key=lambda x: x.weight, reverse=True)
