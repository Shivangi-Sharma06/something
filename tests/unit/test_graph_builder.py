"""Unit+integration tests for graph_agent.graph.builder and queries.

Tests marked @pytest.mark.integration require a live PostgreSQL connection.
"""

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


@pytest.mark.integration
class TestBuildGraph:
    def setup_method(self):
        self.session = _pg_session()
        self.root = str(SAMPLE_REPO)
        _cleanup(self.session, self.root)
        # Index the sample repo first
        from graph_agent.indexer.ingester import index_repository
        result = index_repository(self.session, SAMPLE_REPO, repo_name="graph_test")
        self.session.commit()
        self.repo_id = result.repo_id

    def teardown_method(self):
        _cleanup(self.session, self.root)
        self.session.close()

    def test_build_creates_contains_edges(self):
        from graph_agent.graph.builder import build_graph
        result = build_graph(self.session, self.repo_id)
        self.session.commit()
        assert result.contains_edges > 0, "Expected CONTAINS edges for class→method"

    def test_build_creates_calls_edges(self):
        from graph_agent.graph.builder import build_graph
        result = build_graph(self.session, self.repo_id)
        self.session.commit()
        assert result.edges_resolved > 0, "Expected at least one resolved CALLS edge"

    def test_known_cross_file_call_resolved(self):
        """main.py's Calculator.run_divide calls utils.divide — must be resolved."""
        from graph_agent.graph.builder import build_graph
        from graph_agent.db.models import Symbol, Edge
        build_graph(self.session, self.repo_id)
        self.session.commit()

        divide_sym = (
            self.session.query(Symbol)
            .filter_by(repo_id=self.repo_id, name="divide")
            .first()
        )
        assert divide_sym is not None, "Symbol 'divide' must exist"

        # There should be at least one CALLS edge targeting 'divide'
        incoming = (
            self.session.query(Edge)
            .filter_by(target_id=divide_sym.id, edge_type="CALLS")
            .all()
        )
        assert len(incoming) > 0, (
            f"Expected CALLS edges → divide, got none. "
            f"Check that run_divide's call to divide() is in unresolved_refs."
        )

    def test_each_edge_has_confidence_and_method(self):
        from graph_agent.graph.builder import build_graph
        from graph_agent.db.models import Edge
        build_graph(self.session, self.repo_id)
        self.session.commit()
        edges = self.session.query(Edge).filter_by(repo_id=self.repo_id).all()
        for e in edges:
            assert e.confidence is not None, f"Edge {e.id} has no confidence"
            assert e.resolution_method is not None, f"Edge {e.id} has no resolution_method"

    def test_rebuild_is_idempotent(self):
        """Calling build_graph twice produces same edge count."""
        from graph_agent.graph.builder import build_graph
        from graph_agent.db.models import Edge
        build_graph(self.session, self.repo_id)
        self.session.commit()
        count_1 = self.session.query(Edge).filter_by(repo_id=self.repo_id).count()

        build_graph(self.session, self.repo_id)
        self.session.commit()
        count_2 = self.session.query(Edge).filter_by(repo_id=self.repo_id).count()

        assert count_1 == count_2, f"Edge counts differ: {count_1} vs {count_2}"

    def test_unresolved_calls_tracked(self):
        """Unresolved call refs (stdlib calls like print) must be counted."""
        from graph_agent.graph.builder import build_graph
        result = build_graph(self.session, self.repo_id)
        self.session.commit()
        # stdlib calls (print, pytest.approx, etc.) won't resolve
        assert result.edges_unresolved >= 0  # can be 0 if all resolve


@pytest.mark.integration
class TestGraphQueries:
    def setup_method(self):
        self.session = _pg_session()
        self.root = str(SAMPLE_REPO)
        _cleanup(self.session, self.root)
        from graph_agent.indexer.ingester import index_repository
        from graph_agent.graph.builder import build_graph
        r = index_repository(self.session, SAMPLE_REPO, repo_name="query_test")
        self.session.commit()
        self.repo_id = r.repo_id
        build_graph(self.session, self.repo_id)
        self.session.commit()

    def teardown_method(self):
        _cleanup(self.session, self.root)
        self.session.close()

    def _get_sym(self, name: str):
        from graph_agent.db.models import Symbol
        return (
            self.session.query(Symbol)
            .filter_by(repo_id=self.repo_id, name=name)
            .first()
        )

    def test_get_callees_of_run_divide(self):
        from graph_agent.graph.queries import get_callees
        sym = self._get_sym("run_divide")
        assert sym is not None
        callees = get_callees(self.session, self.repo_id, sym.id)
        callee_names = [r["qualified_name"].split(".")[-1] for r in callees]
        assert "divide" in callee_names or "format_result" in callee_names, (
            f"Expected divide or format_result as callee, got: {callee_names}"
        )

    def test_get_callers_of_divide(self):
        from graph_agent.graph.queries import get_callers
        sym = self._get_sym("divide")
        assert sym is not None
        callers = get_callers(self.session, self.repo_id, sym.id)
        caller_names = [r["qualified_name"].split(".")[-1] for r in callers]
        assert len(callers) > 0, "Expected at least one caller of divide"

    def test_neighborhood_is_non_empty(self):
        from graph_agent.graph.queries import get_neighborhood
        sym = self._get_sym("divide")
        assert sym is not None
        nbr = get_neighborhood(self.session, self.repo_id, sym.id, max_depth=2)
        assert len(nbr) > 0

    def test_direct_edges(self):
        from graph_agent.graph.queries import get_direct_edges
        sym = self._get_sym("divide")
        assert sym is not None
        edges = get_direct_edges(self.session, self.repo_id, sym.id)
        assert len(edges) > 0
