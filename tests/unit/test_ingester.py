"""Unit tests for graph_agent.indexer.ingester.

Uses a PostgreSQL test database (graph_agent_db) to exercise the full
upsert pipeline including idempotency.

These tests are marked @pytest.mark.integration and require a live PG connection.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from graph_agent.indexer.ingester import index_repository, IndexResult


SAMPLE_REPO = Path(__file__).parent.parent / "fixtures" / "sample_repo"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pg_session():
    """Return a live PostgreSQL session for integration tests."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    engine = create_engine(
        "postgresql+psycopg2://graph_agent:graph_agent_pass@127.0.0.1:5432/graph_agent_db",
        future=True,
    )
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    return Session()


def _cleanup(session, root_path: str):
    """Remove the test repository row (cascades to all child rows)."""
    from graph_agent.db.models import Repository
    repo = session.query(Repository).filter_by(root_path=root_path).first()
    if repo:
        session.delete(repo)
        session.commit()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
class TestIndexRepository:
    def setup_method(self):
        self.session = _pg_session()
        self.root = str(SAMPLE_REPO)
        _cleanup(self.session, self.root)

    def teardown_method(self):
        _cleanup(self.session, self.root)
        self.session.close()

    def test_indexes_sample_repo(self):
        result = index_repository(self.session, SAMPLE_REPO, repo_name="test_fixture")
        self.session.commit()
        assert result.repo_id > 0
        assert result.files_indexed >= 2   # main.py + utils.py minimum
        assert result.symbols_upserted > 0

    def test_idempotent_reindex(self):
        """Re-indexing the same repo should skip all unchanged files."""
        r1 = index_repository(self.session, SAMPLE_REPO, repo_name="test_fixture")
        self.session.commit()
        r2 = index_repository(self.session, SAMPLE_REPO, repo_name="test_fixture")
        self.session.commit()

        assert r2.files_indexed == 0        # nothing changed
        assert r2.files_skipped == r1.files_indexed  # all skipped
        # Row counts must be the same
        from graph_agent.db.models import Symbol, File, Repository
        repo = self.session.query(Repository).filter_by(root_path=self.root).first()
        file_count = self.session.query(File).filter_by(repo_id=repo.id).count()
        sym_count = self.session.query(Symbol).filter_by(repo_id=repo.id).count()
        assert file_count == r1.files_indexed
        assert sym_count == r1.symbols_upserted

    def test_known_symbols_present(self):
        """After indexing, specific symbols from the fixture must be in the DB."""
        index_repository(self.session, SAMPLE_REPO, repo_name="test_fixture")
        self.session.commit()

        from graph_agent.db.models import Symbol, Repository
        repo = self.session.query(Repository).filter_by(root_path=self.root).first()
        sym_names = {s.name for s in self.session.query(Symbol).filter_by(repo_id=repo.id).all()}
        assert {"add", "subtract", "divide", "format_result", "Calculator"}.issubset(sym_names)

    def test_unresolved_refs_recorded(self):
        """Raw call references must be stored as unresolved_refs after indexing."""
        index_repository(self.session, SAMPLE_REPO, repo_name="test_fixture")
        self.session.commit()

        from graph_agent.db.models import UnresolvedRef, Symbol, Repository
        repo = self.session.query(Repository).filter_by(root_path=self.root).first()
        syms = self.session.query(Symbol).filter_by(repo_id=repo.id).all()
        sym_ids = {s.id for s in syms}
        refs = self.session.query(UnresolvedRef).filter(
            UnresolvedRef.source_id.in_(sym_ids)
        ).all()
        assert len(refs) > 0  # at least one call reference recorded

    def test_reindex_after_change_updates_symbols(self, tmp_path):
        """Modifying a file triggers re-extraction of its symbols."""
        # Write a simple one-function file
        f = tmp_path / "mod.py"
        f.write_text("def version_one(): pass\n", encoding="utf-8")

        r1 = index_repository(self.session, tmp_path, repo_name="tmp_test")
        self.session.commit()

        from graph_agent.db.models import Symbol, Repository
        repo = self.session.query(Repository).filter_by(root_path=str(tmp_path)).first()
        names_v1 = {s.name for s in self.session.query(Symbol).filter_by(repo_id=repo.id).all()}
        assert "version_one" in names_v1

        # Modify the file
        f.write_text("def version_two(): pass\n", encoding="utf-8")
        r2 = index_repository(self.session, tmp_path, repo_name="tmp_test")
        self.session.commit()

        names_v2 = {s.name for s in self.session.query(Symbol).filter_by(repo_id=repo.id).all()}
        assert "version_two" in names_v2
        assert "version_one" not in names_v2  # old symbol removed

        _cleanup(self.session, str(tmp_path))
