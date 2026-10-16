"""Unit tests for graph_agent.context.selector."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graph_agent.localizer.ranker import ScoredSymbol, ScoreBreakdown
from graph_agent.localizer.evidence import EvidenceKind


SAMPLE_REPO = Path(__file__).parent.parent / "fixtures" / "sample_repo"


def _make_candidate(sym_id: int, qname: str, score: float = 1.0) -> ScoredSymbol:
    return ScoredSymbol(
        symbol_id=sym_id,
        qualified_name=qname,
        kind="function",
        file_id=1,
        score=score,
        breakdown=ScoreBreakdown(total=score),
        evidence_kind=EvidenceKind.KNOWN,
        evidence_source="stack_frame",
    )


def _pg_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(
        "postgresql+psycopg2://graph_agent:graph_agent_pass@127.0.0.1:5432/graph_agent_db",
        future=True,
    )
    return sessionmaker(bind=engine, expire_on_commit=False)()


def _cleanup(session, root_path: str):
    from graph_agent.db.models import Repository
    repo = session.query(Repository).filter_by(root_path=root_path).first()
    if repo:
        session.delete(repo)
        session.commit()


class TestBudgetContextSelectorUnit:
    """Pure-unit tests using mock sessions — no DB needed."""

    def test_token_estimate(self):
        from graph_agent.context.selector import _estimate_tokens
        assert _estimate_tokens("a" * 400) == 100
        assert _estimate_tokens("") == 1  # min 1

    def test_empty_candidates_returns_empty_context(self):
        from graph_agent.context.selector import BudgetContextSelector
        selector = BudgetContextSelector()
        mock_session = MagicMock()
        mock_session.execute.return_value.first.return_value = ("/tmp/repo",)
        ctx = selector.select(mock_session, repo_id=1, candidates=[], budget_tokens=1000)
        assert ctx.slices == []
        assert ctx.total_tokens_estimate == 0

    def test_budget_zero_truncates_all(self):
        """Budget of 0 tokens → slices empty (either truncated or no content returned)."""
        from graph_agent.context.selector import BudgetContextSelector
        selector = BudgetContextSelector()
        mock_session = MagicMock()
        # Simulate repo root lookup returning sample_repo path
        mock_session.execute.return_value.first.return_value = (str(SAMPLE_REPO),)
        candidates = [_make_candidate(1, "utils.divide")]
        ctx = selector.select(mock_session, repo_id=1, candidates=candidates, budget_tokens=0)
        # With budget=0, either truncated=True or slices are empty
        # (source lookup via mock may return None, causing 0 slices)
        assert ctx.truncated or len(ctx.slices) == 0


@pytest.mark.integration
class TestBudgetContextSelectorIntegration:
    def setup_method(self):
        self.session = _pg_session()
        self.root = str(SAMPLE_REPO)
        _cleanup(self.session, self.root)
        from graph_agent.indexer.ingester import index_repository
        from graph_agent.graph.builder import build_graph
        from graph_agent.localizer.ranker import localize
        r = index_repository(self.session, SAMPLE_REPO, repo_name="ctx_test")
        self.session.commit()
        self.repo_id = r.repo_id
        build_graph(self.session, self.repo_id)
        self.session.commit()
        self.candidates = localize(
            self.session, self.repo_id,
            stack_trace=(
                'File "utils.py", line 18, in divide\n'
                'ZeroDivisionError: division by zero'
            ),
            top_k=5,
        )

    def teardown_method(self):
        _cleanup(self.session, self.root)
        self.session.close()

    def test_never_exceeds_budget(self):
        from graph_agent.context.selector import BudgetContextSelector
        budget = 500
        selector = BudgetContextSelector()
        ctx = selector.select(self.session, self.repo_id, self.candidates, budget)
        assert ctx.total_tokens_estimate <= budget, (
            f"Context exceeded budget: {ctx.total_tokens_estimate} > {budget}"
        )

    def test_top_k_get_full_source(self):
        from graph_agent.context.selector import BudgetContextSelector
        from graph_agent.settings import get_settings
        selector = BudgetContextSelector()
        ctx = selector.select(
            self.session, self.repo_id, self.candidates,
            budget_tokens=8000
        )
        full_k = get_settings().context.full_source_top_k
        full_count = sum(1 for s in ctx.slices if s.is_full)
        assert full_count <= full_k

    def test_deterministic_output(self):
        from graph_agent.context.selector import BudgetContextSelector
        selector = BudgetContextSelector()
        ctx1 = selector.select(self.session, self.repo_id, self.candidates, 8000)
        ctx2 = selector.select(self.session, self.repo_id, self.candidates, 8000)
        names1 = [s.qualified_name for s in ctx1.slices]
        names2 = [s.qualified_name for s in ctx2.slices]
        assert names1 == names2, "Context selection is not deterministic"

    def test_slices_have_content(self):
        from graph_agent.context.selector import BudgetContextSelector
        selector = BudgetContextSelector()
        ctx = selector.select(self.session, self.repo_id, self.candidates, 8000)
        for slc in ctx.slices:
            assert slc.content.strip(), f"Empty content for {slc.qualified_name}"
