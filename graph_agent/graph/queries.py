"""Recursive SQL CTE queries for graph traversal.

All queries return plain rows (not ORM objects) for performance.
Parameters are passed as keyword args to avoid SQL injection.

ADR-2 (from plan): cycle protection via `path` array + `is_cycle` flag.
Depth is bounded by :max_depth parameter (default from config).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from graph_agent.settings import get_settings


def _max_depth() -> int:
    return get_settings().graph.max_traversal_depth


# ---------------------------------------------------------------------------
# Callers (N-hop): who calls a given symbol?
# ---------------------------------------------------------------------------
CALLERS_CTE = """
WITH RECURSIVE callers(symbol_id, depth, path, is_cycle) AS (
    SELECT e.source_id, 1, ARRAY[e.target_id, e.source_id], false
    FROM graph_agent.edges e
    WHERE e.target_id = :start_id
      AND e.edge_type = 'CALLS'
      AND e.repo_id   = :repo_id
  UNION ALL
    SELECT e.source_id,
           c.depth + 1,
           c.path || e.source_id,
           e.source_id = ANY(c.path)
    FROM graph_agent.edges e
    JOIN callers c ON e.target_id = c.symbol_id
    WHERE e.edge_type = 'CALLS'
      AND e.repo_id   = :repo_id
      AND NOT c.is_cycle
      AND c.depth < :max_depth
)
SELECT DISTINCT
    s.id              AS symbol_id,
    s.qualified_name,
    s.kind,
    s.file_id,
    callers.depth,
    callers.path
FROM callers
JOIN graph_agent.symbols s ON s.id = callers.symbol_id
WHERE NOT callers.is_cycle
ORDER BY callers.depth, s.qualified_name
"""


def get_callers(
    session: Session,
    repo_id: int,
    start_symbol_id: int,
    max_depth: int | None = None,
) -> list[dict[str, Any]]:
    """Return all symbols that (directly or transitively) call *start_symbol_id*."""
    depth = max_depth or _max_depth()
    rows = session.execute(
        text(CALLERS_CTE),
        {"start_id": start_symbol_id, "repo_id": repo_id, "max_depth": depth},
    ).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Callees (N-hop): what does a given symbol call?
# ---------------------------------------------------------------------------
CALLEES_CTE = """
WITH RECURSIVE callees(symbol_id, depth, path, is_cycle) AS (
    SELECT e.target_id, 1, ARRAY[e.source_id, e.target_id], false
    FROM graph_agent.edges e
    WHERE e.source_id = :start_id
      AND e.edge_type = 'CALLS'
      AND e.repo_id   = :repo_id
  UNION ALL
    SELECT e.target_id,
           c.depth + 1,
           c.path || e.target_id,
           e.target_id = ANY(c.path)
    FROM graph_agent.edges e
    JOIN callees c ON e.source_id = c.symbol_id
    WHERE e.edge_type = 'CALLS'
      AND e.repo_id   = :repo_id
      AND NOT c.is_cycle
      AND c.depth < :max_depth
)
SELECT DISTINCT
    s.id              AS symbol_id,
    s.qualified_name,
    s.kind,
    s.file_id,
    callees.depth,
    callees.path
FROM callees
JOIN graph_agent.symbols s ON s.id = callees.symbol_id
WHERE NOT callees.is_cycle
ORDER BY callees.depth, s.qualified_name
"""


def get_callees(
    session: Session,
    repo_id: int,
    start_symbol_id: int,
    max_depth: int | None = None,
) -> list[dict[str, Any]]:
    """Return all symbols transitively called by *start_symbol_id*."""
    depth = max_depth or _max_depth()
    rows = session.execute(
        text(CALLEES_CTE),
        {"start_id": start_symbol_id, "repo_id": repo_id, "max_depth": depth},
    ).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# N-hop neighborhood (bidirectional, any edge type)
# ---------------------------------------------------------------------------
NEIGHBORHOOD_CTE = """
WITH RECURSIVE nbr(symbol_id, depth, path, is_cycle) AS (
    SELECT CAST(:start_id AS int), 0, ARRAY[CAST(:start_id AS int)], false
  UNION ALL
    SELECT
        CASE WHEN e.source_id = n.symbol_id THEN e.target_id ELSE e.source_id END,
        n.depth + 1,
        n.path || CASE WHEN e.source_id = n.symbol_id THEN e.target_id ELSE e.source_id END,
        (CASE WHEN e.source_id = n.symbol_id THEN e.target_id ELSE e.source_id END) = ANY(n.path)
    FROM graph_agent.edges e
    JOIN nbr n ON (e.source_id = n.symbol_id OR e.target_id = n.symbol_id)
    WHERE NOT n.is_cycle
      AND e.repo_id = :repo_id
      AND n.depth < :max_depth
)
SELECT DISTINCT
    s.id             AS symbol_id,
    s.qualified_name,
    s.kind,
    s.file_id,
    MIN(nbr.depth)   AS depth
FROM nbr
JOIN graph_agent.symbols s ON s.id = nbr.symbol_id
WHERE NOT nbr.is_cycle
GROUP BY s.id, s.qualified_name, s.kind, s.file_id
ORDER BY depth, s.qualified_name
"""


def get_neighborhood(
    session: Session,
    repo_id: int,
    start_symbol_id: int,
    max_depth: int | None = None,
) -> list[dict[str, Any]]:
    """Return the N-hop neighbourhood of *start_symbol_id* (all edge types)."""
    depth = max_depth or _max_depth()
    rows = session.execute(
        text(NEIGHBORHOOD_CTE),
        {"start_id": start_symbol_id, "repo_id": repo_id, "max_depth": depth},
    ).mappings().all()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Convenience: direct edges in/out of a symbol
# ---------------------------------------------------------------------------
def get_direct_edges(
    session: Session,
    repo_id: int,
    symbol_id: int,
    edge_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return all direct edges where symbol_id is source or target."""
    q = """
    SELECT e.id, e.source_id, e.target_id, e.edge_type, e.confidence,
           e.resolution_method,
           ss.qualified_name AS source_name,
           st.qualified_name AS target_name
    FROM graph_agent.edges e
    JOIN graph_agent.symbols ss ON ss.id = e.source_id
    JOIN graph_agent.symbols st ON st.id = e.target_id
    WHERE e.repo_id = :repo_id
      AND (e.source_id = :sym_id OR e.target_id = :sym_id)
    """
    params: dict[str, Any] = {"repo_id": repo_id, "sym_id": symbol_id}
    if edge_type:
        q += " AND e.edge_type = :edge_type"
        params["edge_type"] = edge_type
    q += " ORDER BY e.edge_type, e.confidence DESC"
    rows = session.execute(text(q), params).mappings().all()
    return [dict(r) for r in rows]
