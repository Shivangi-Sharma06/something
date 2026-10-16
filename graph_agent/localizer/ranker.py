"""Bug localizer — traverses the code graph and scores candidate symbols.

Scoring formula (all weights configurable in config/default.toml):
  score(sym) = Σ weight_i × evidence_i

Evidence sources and their weight keys:
  stack_frame_exact   — symbol is directly in a stack frame
  test_mention        — symbol referenced by failing test
  caller_depth_1      — direct caller of a known-bad symbol
  caller_depth_2      — caller 2 hops away
  callee_depth_1      — called by a known-bad symbol
  file_co_change      — (placeholder) historical co-change heuristic

Each candidate in the output includes a per-feature score breakdown and
an evidence_kind label (KNOWN vs HEURISTIC) for transparency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy import text

from graph_agent.settings import get_settings
from graph_agent.localizer.evidence import EvidenceItem, EvidenceKind, extract_evidence
from graph_agent.graph.queries import get_callers, get_callees

log = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    stack_frame_exact: float = 0.0
    test_mention: float = 0.0
    caller_depth_1: float = 0.0
    caller_depth_2: float = 0.0
    callee_depth_1: float = 0.0
    description_mention: float = 0.0
    total: float = 0.0


@dataclass
class ScoredSymbol:
    symbol_id: int
    qualified_name: str
    kind: str
    file_id: int
    score: float
    breakdown: ScoreBreakdown
    evidence_kind: EvidenceKind    # KNOWN or HEURISTIC
    evidence_source: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_sym_meta(session: Session, symbol_id: int) -> dict[str, Any]:
    row = session.execute(
        text("""
            SELECT s.id, s.qualified_name, s.kind, s.file_id
            FROM graph_agent.symbols s
            WHERE s.id = :sym_id
        """),
        {"sym_id": symbol_id},
    ).mappings().first()
    return dict(row) if row else {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def localize(
    session: Session,
    repo_id: int,
    description: str = "",
    stack_trace: str = "",
    failing_test: str = "",
    top_k: int | None = None,
) -> list[ScoredSymbol]:
    """Rank repository symbols by suspiciousness.

    Steps:
    1. Extract evidence items from all inputs.
    2. For each evidence symbol, traverse callers/callees up to depth 2.
    3. Accumulate weighted scores.
    4. Return top-k sorted by score desc.
    """
    w = get_settings().localization
    k = top_k or w.top_k

    evidence_items = extract_evidence(
        session, repo_id,
        description=description,
        stack_trace=stack_trace,
        failing_test=failing_test,
    )

    if not evidence_items:
        log.warning("No evidence items found — returning empty ranking")
        return []

    # Accumulate scores per symbol_id
    scores: dict[int, ScoreBreakdown] = {}
    meta: dict[int, dict[str, Any]] = {}
    kinds: dict[int, EvidenceKind] = {}
    sources: dict[int, str] = {}

    def _ensure(sym_id: int, sym_meta: dict):
        if sym_id not in scores:
            scores[sym_id] = ScoreBreakdown()
            meta[sym_id] = sym_meta
            kinds[sym_id] = EvidenceKind.HEURISTIC
            sources[sym_id] = "graph_traversal"

    # ------------------------------------------------------------------
    # Direct evidence scoring
    # ------------------------------------------------------------------
    for ev in evidence_items:
        sym_meta = _get_sym_meta(session, ev.symbol_id)
        if not sym_meta:
            continue
        _ensure(ev.symbol_id, sym_meta)
        kinds[ev.symbol_id] = ev.kind  # promote to KNOWN if applicable
        sources[ev.symbol_id] = ev.source

        if ev.source in ("stack_frame_line", "stack_frame_name"):
            scores[ev.symbol_id].stack_frame_exact += w.stack_frame_exact
        elif ev.source == "failing_test":
            scores[ev.symbol_id].test_mention += w.test_mention
        elif ev.source in ("description_mention", "repo_symbol_context"):
            scores[ev.symbol_id].description_mention += w.file_co_change

    # ------------------------------------------------------------------
    # Graph-based propagation (callers/callees of evidence symbols)
    # ------------------------------------------------------------------
    for ev in evidence_items:
        if ev.kind != EvidenceKind.KNOWN:
            continue  # only propagate from confirmed evidence

        # Callers at depth 1 and 2
        callers = get_callers(session, repo_id, ev.symbol_id, max_depth=2)
        for row in callers:
            sym_id = row["symbol_id"]
            sym_meta = {"id": sym_id, "qualified_name": row["qualified_name"],
                        "kind": row["kind"], "file_id": row["file_id"]}
            _ensure(sym_id, sym_meta)
            depth = row["depth"]
            if depth == 1:
                scores[sym_id].caller_depth_1 += w.caller_depth_1
            elif depth == 2:
                scores[sym_id].caller_depth_2 += w.caller_depth_2

        # Callees at depth 1
        callees = get_callees(session, repo_id, ev.symbol_id, max_depth=1)
        for row in callees:
            sym_id = row["symbol_id"]
            sym_meta = {"id": sym_id, "qualified_name": row["qualified_name"],
                        "kind": row["kind"], "file_id": row["file_id"]}
            _ensure(sym_id, sym_meta)
            scores[sym_id].callee_depth_1 += w.callee_depth_1

    # ------------------------------------------------------------------
    # Compute totals and build results
    # ------------------------------------------------------------------
    results: list[ScoredSymbol] = []
    for sym_id, bd in scores.items():
        bd.total = (
            bd.stack_frame_exact
            + bd.test_mention
            + bd.caller_depth_1
            + bd.caller_depth_2
            + bd.callee_depth_1
            + bd.description_mention
        )
        sym_meta = meta.get(sym_id, {})
        if not sym_meta:
            continue
        results.append(ScoredSymbol(
            symbol_id=sym_id,
            qualified_name=sym_meta.get("qualified_name", ""),
            kind=sym_meta.get("kind", ""),
            file_id=sym_meta.get("file_id", 0),
            score=bd.total,
            breakdown=bd,
            evidence_kind=kinds.get(sym_id, EvidenceKind.HEURISTIC),
            evidence_source=sources.get(sym_id, ""),
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:k]
