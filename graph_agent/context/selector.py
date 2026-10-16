"""ContextSelector — selects source context to send to the LLM.

Interface (ABC): ContextSelector.select(candidates, budget_tokens) → SelectedContext
Default impl:   BudgetContextSelector
  - Full source for top-K candidates
  - Signature-only for the rest
  - Never exceeds budget_tokens (rough estimate: len(text) / 4)
  - Output is deterministic for identical inputs

ADR-4: The ContextSelector is a clean extension point.  An info-gain or
entropy-based selector can be plugged in by subclassing and registering via
the config (context.selector_class).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from graph_agent.localizer.ranker import ScoredSymbol
from graph_agent.settings import get_settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class FileSlice:
    """A slice of source code from a file."""
    rel_path: str
    qualified_name: str
    content: str          # either full source or signature only
    is_full: bool         # True = full implementation, False = signature only
    line_start: int
    line_end: int


@dataclass
class SelectedContext:
    slices: list[FileSlice] = field(default_factory=list)
    total_tokens_estimate: int = 0
    budget_tokens: int = 0
    truncated: bool = False   # True if budget forced us to drop candidates


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------
class ContextSelector(ABC):
    """Interface for context selection strategies."""

    @abstractmethod
    def select(
        self,
        session: "Session",
        repo_id: int,
        candidates: list[ScoredSymbol],
        budget_tokens: int,
    ) -> SelectedContext:
        """Select and format context for the LLM."""
        ...


# ---------------------------------------------------------------------------
# Source retrieval helpers
# ---------------------------------------------------------------------------
def _get_file_rel_path(session: Session, file_id: int) -> str:
    row = session.execute(
        text("SELECT rel_path FROM graph_agent.files WHERE id = :fid"),
        {"fid": file_id},
    ).first()
    return row[0] if row else ""


def _read_source_lines(repo_root: Path, rel_path: str, start: int, end: int) -> str:
    """Read lines [start, end] (1-indexed, inclusive) from a repo file."""
    fpath = repo_root / rel_path
    try:
        lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
        # Convert to 0-indexed
        s = max(0, start - 1)
        e = min(len(lines), end)
        return "\n".join(lines[s:e])
    except OSError:
        return f"# [could not read {rel_path}]"


def _get_symbol_source(
    session: Session, repo_id: int, sym: ScoredSymbol, repo_root: Path, full: bool
) -> Optional[str]:
    """Return the source text for sym (full body or signature only)."""
    row = session.execute(
        text("""
            SELECT s.line_start, s.line_end, s.signature, f.rel_path
            FROM graph_agent.symbols s
            JOIN graph_agent.files f ON f.id = s.file_id
            WHERE s.id = :sid
        """),
        {"sid": sym.symbol_id},
    ).first()
    if not row or len(row) < 4:
        return None

    line_start, line_end, signature, rel_path = row[0], row[1], row[2], row[3]
    if full and line_start and line_end:
        return _read_source_lines(repo_root, rel_path, line_start, line_end)
    elif signature:
        return signature
    return None


def _estimate_tokens(text_: str) -> int:
    """Rough token estimate: 1 token ≈ 4 chars."""
    return max(1, len(text_) // 4)


# ---------------------------------------------------------------------------
# Default implementation
# ---------------------------------------------------------------------------
class BudgetContextSelector(ContextSelector):
    """Top-K full source + signatures for the rest, bounded by token budget."""

    def select(
        self,
        session: Session,
        repo_id: int,
        candidates: list[ScoredSymbol],
        budget_tokens: int,
    ) -> SelectedContext:
        s = get_settings()
        repo_row = session.execute(
            text("SELECT root_path FROM graph_agent.repositories WHERE id = :rid"),
            {"rid": repo_id},
        ).first()
        repo_root = Path(repo_row[0]) if repo_row else Path(".")

        full_source_top_k = s.context.full_source_top_k
        result = SelectedContext(budget_tokens=budget_tokens)
        tokens_used = 0

        for i, sym in enumerate(candidates):
            is_full = i < full_source_top_k
            src = _get_symbol_source(session, repo_id, sym, repo_root, full=is_full)
            if not src:
                continue

            token_cost = _estimate_tokens(src)
            if tokens_used + token_cost > budget_tokens:
                result.truncated = True
                log.debug(
                    "Budget exceeded at candidate %d (%s), stopping. "
                    "Used %d / %d tokens",
                    i, sym.qualified_name, tokens_used, budget_tokens,
                )
                break

            rel_path = _get_file_rel_path(session, sym.file_id)
            row = session.execute(
                text("SELECT line_start, line_end FROM graph_agent.symbols WHERE id = :sid"),
                {"sid": sym.symbol_id},
            ).first()
            line_start, line_end = (row[0] or 0, row[1] or 0) if row else (0, 0)

            result.slices.append(FileSlice(
                rel_path=rel_path,
                qualified_name=sym.qualified_name,
                content=src,
                is_full=is_full,
                line_start=line_start,
                line_end=line_end,
            ))
            tokens_used += token_cost

        result.total_tokens_estimate = tokens_used
        return result


def get_default_selector() -> ContextSelector:
    return BudgetContextSelector()
