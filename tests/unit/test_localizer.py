"""Unit tests for graph_agent.localizer.evidence and .ranker."""

from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_REPO = Path(__file__).parent.parent / "fixtures" / "sample_repo"


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


# ---------------------------------------------------------------------------
# Evidence extractor unit tests (no DB needed)
# ---------------------------------------------------------------------------
class TestParseStackTrace:
    def test_extracts_frames(self):
        from graph_agent.localizer.evidence import _parse_stack_frames
        trace = (
            'Traceback (most recent call last):\n'
            '  File "utils.py", line 18, in divide\n'
            '    return numerator / denominator\n'
            'ZeroDivisionError: division by zero\n'
        )
        frames = _parse_stack_frames(trace)
        assert len(frames) >= 1
        rel_file, lineno, func_name = frames[0]
        assert "utils.py" in rel_file
        assert lineno == 18
        assert func_name == "divide"

    def test_multiple_frames(self):
        from graph_agent.localizer.evidence import _parse_stack_frames
        trace = (
            '  File "main.py", line 34, in run_divide\n'
            '  File "utils.py", line 18, in divide\n'
        )
        frames = _parse_stack_frames(trace)
        assert len(frames) == 2

    def test_empty_trace(self):
        from graph_agent.localizer.evidence import _parse_stack_frames
        assert _parse_stack_frames("") == []


class TestParseTestName:
    def test_simple_name(self):
        from graph_agent.localizer.evidence import _parse_test_name
        names = _parse_test_name("test_divide_by_zero")
        assert "test_divide_by_zero" in names

    def test_path_with_colons(self):
        from graph_agent.localizer.evidence import _parse_test_name
        names = _parse_test_name("tests/test_utils.py::test_divide_by_zero")
        assert "test_divide_by_zero" in names

    def test_class_method(self):
        from graph_agent.localizer.evidence import _parse_test_name
        names = _parse_test_name("TestCalc::test_run")
        assert "test_run" in names


# ---------------------------------------------------------------------------
# Integration tests (live PG)
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestExtractEvidence:
    def setup_method(self):
        self.session = _pg_session()
        self.root = str(SAMPLE_REPO)
        _cleanup(self.session, self.root)
        from graph_agent.indexer.ingester import index_repository
        from graph_agent.graph.builder import build_graph
        r = index_repository(self.session, SAMPLE_REPO, repo_name="ev_test")
        self.session.commit()
        self.repo_id = r.repo_id
        build_graph(self.session, self.repo_id)
        self.session.commit()

    def teardown_method(self):
        _cleanup(self.session, self.root)
        self.session.close()

    def test_extracts_from_stack_trace(self):
        from graph_agent.localizer.evidence import extract_evidence, EvidenceKind
        trace = (
            'File "utils.py", line 18, in divide\n'
            '    return numerator / denominator\n'
            'ZeroDivisionError: division by zero'
        )
        items = extract_evidence(self.session, self.repo_id, stack_trace=trace)
        assert len(items) > 0
        # At least one KNOWN item
        known = [i for i in items if i.kind == EvidenceKind.KNOWN]
        assert len(known) > 0

    def test_extracts_from_failing_test(self):
        from graph_agent.localizer.evidence import extract_evidence
        items = extract_evidence(
            self.session, self.repo_id,
            failing_test="tests/test_utils.py::test_divide_by_zero"
        )
        # test_divide_by_zero is a symbol in the fixture
        names = [i.qualified_name for i in items]
        assert any("test_divide_by_zero" in n for n in names)

    def test_description_adds_heuristic(self):
        from graph_agent.localizer.evidence import extract_evidence, EvidenceKind
        items = extract_evidence(
            self.session, self.repo_id,
            description="The divide function raises ZeroDivisionError"
        )
        heuristic = [i for i in items if i.kind == EvidenceKind.HEURISTIC]
        assert len(heuristic) > 0


@pytest.mark.integration
class TestLocalize:
    def setup_method(self):
        self.session = _pg_session()
        self.root = str(SAMPLE_REPO)
        _cleanup(self.session, self.root)
        from graph_agent.indexer.ingester import index_repository
        from graph_agent.graph.builder import build_graph
        r = index_repository(self.session, SAMPLE_REPO, repo_name="loc_test")
        self.session.commit()
        self.repo_id = r.repo_id
        build_graph(self.session, self.repo_id)
        self.session.commit()

    def teardown_method(self):
        _cleanup(self.session, self.root)
        self.session.close()

    STACK_TRACE = (
        'Traceback (most recent call last):\n'
        '  File "main.py", line 34, in run_divide\n'
        '  File "utils.py", line 18, in divide\n'
        '    return numerator / denominator\n'
        'ZeroDivisionError: division by zero\n'
    )

    def test_buggy_function_in_top_k(self):
        """The seeded bug in `divide` must appear in the top-5 candidates."""
        from graph_agent.localizer.ranker import localize
        candidates = localize(
            self.session, self.repo_id,
            stack_trace=self.STACK_TRACE,
            failing_test="test_divide_by_zero",
            top_k=10,
        )
        names = [c.qualified_name for c in candidates]
        top5 = names[:5]
        assert any("divide" in n for n in top5), (
            f"'divide' not in top-5. Got: {top5}"
        )

    def test_scores_are_positive(self):
        from graph_agent.localizer.ranker import localize
        candidates = localize(
            self.session, self.repo_id,
            stack_trace=self.STACK_TRACE,
        )
        assert all(c.score > 0 for c in candidates)

    def test_score_breakdown_populated(self):
        from graph_agent.localizer.ranker import localize
        candidates = localize(
            self.session, self.repo_id,
            stack_trace=self.STACK_TRACE,
        )
        for c in candidates:
            assert c.breakdown.total == pytest.approx(c.score)

    def test_sorted_descending(self):
        from graph_agent.localizer.ranker import localize
        candidates = localize(
            self.session, self.repo_id,
            stack_trace=self.STACK_TRACE,
        )
        scores = [c.score for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_empty_evidence_returns_empty(self):
        from graph_agent.localizer.ranker import localize
        candidates = localize(self.session, self.repo_id)
        assert candidates == []
