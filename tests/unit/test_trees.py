"""Unit and integration tests for Permanent AST Tree and Temporary Decision Tree."""

from __future__ import annotations

from pathlib import Path
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from graph_agent.graph.ast_tree import build_permanent_ast_tree
from graph_agent.graph.decision_tree import build_temporary_decision_tree, classify_prompt_intent
from graph_agent.graph.builder import build_graph
from graph_agent.indexer.ingester import index_repository
from graph_agent.db.models import Repository

SAMPLE_REPO = Path(__file__).parent.parent / "fixtures" / "sample_repo"


def _pg_session():
    engine = create_engine(
        "postgresql+psycopg2://graph_agent:graph_agent_pass@127.0.0.1:5432/graph_agent_db",
        future=True,
    )
    return sessionmaker(bind=engine, expire_on_commit=False)()


def _cleanup(session, root_path: str):
    repo = session.query(Repository).filter_by(root_path=root_path).first()
    if repo:
        session.delete(repo)
        session.commit()


def test_classify_prompt_intent():
    assert classify_prompt_intent("Fix ZeroDivisionError in divide()") == "repair"
    assert classify_prompt_intent("Fix bug with missing guard") == "repair"
    assert classify_prompt_intent("Refactor calculate function to use list comprehension") == "refactor"
    assert classify_prompt_intent("How does the calculation logic work?") == "qa_explanation"


@pytest.mark.integration
class TestTreesIntegration:
    def setup_method(self):
        self.session = _pg_session()
        self.root = str(SAMPLE_REPO)
        _cleanup(self.session, self.root)
        result = index_repository(self.session, SAMPLE_REPO, repo_name="tree_test")
        self.session.commit()
        self.repo_id = result.repo_id
        build_graph(self.session, self.repo_id)
        self.session.commit()

    def teardown_method(self):
        _cleanup(self.session, self.root)
        self.session.close()

    def test_permanent_ast_tree_structure(self):
        ast_tree = build_permanent_ast_tree(self.session, self.repo_id)

        assert ast_tree.repo_id == self.repo_id
        assert ast_tree.total_files > 0
        assert ast_tree.total_symbols > 0
        assert len(ast_tree.files) > 0

        # Check Rich tree rendering
        rich_tree = ast_tree.to_rich_tree()
        assert rich_tree is not None

    def test_temporary_decision_tree_structure(self):
        prompt = "Fix ZeroDivisionError in divide function in utils.py"

        dt, context, candidates = build_temporary_decision_tree(
            self.session, self.repo_id, prompt, top_k=5
        )

        assert dt.prompt == prompt
        assert dt.intent == "repair"
        assert dt.repo_id == self.repo_id
        assert len(dt.symbol_nodes) > 0
        assert len(dt.target_files) > 0

        # Check Rich tree rendering
        rich_tree = dt.to_rich_tree()
        assert rich_tree is not None

        # Check prompt context string
        context_str = dt.to_prompt_context()
        assert "Decision Tree Reasoning" in context_str
        assert "divide" in context_str
