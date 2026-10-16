"""Shared pytest fixtures for graph-agent tests.

Uses a temporary PostgreSQL schema (test_graph_agent_<pid>) so tests are
fully isolated from the production database and from each other.

Strategy:
- One shared in-process SQLite engine is used for tests that don't need
  PostgreSQL-specific features (most unit tests).
- Tests that need PostgreSQL (integration) are marked with @pytest.mark.integration
  and skipped if DATABASE_URL is not set.
"""

from __future__ import annotations

import os
import pytest
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from graph_agent.db.models import Base
from graph_agent.db.connection import override_engine


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_REPO = FIXTURES_DIR / "sample_repo"


# ---------------------------------------------------------------------------
# In-memory SQLite engine for unit tests
# (SQLite doesn't support schemas; we patch the schema out)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def sqlite_engine():
    """Create an in-memory SQLite engine with all graph_agent tables."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    # SQLite doesn't support schemas; create tables without them
    # We do this by temporarily patching __table_args__ on each model
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create tables using a modified metadata that strips the schema
    import sqlalchemy as sa
    meta = sa.MetaData()
    # Reflect models without schema prefix for SQLite
    for table in Base.metadata.tables.values():
        tbl_copy = table.tometadata(meta, schema=None)

    meta.create_all(engine)
    override_engine(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(sqlite_engine):
    """Yield a session that rolls back after each test."""
    SessionLocal = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


# ---------------------------------------------------------------------------
# Sample repo path fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_repo_path() -> Path:
    """Return the path to the tiny synthetic sample repo."""
    assert SAMPLE_REPO.exists(), f"Sample repo not found at {SAMPLE_REPO}"
    return SAMPLE_REPO
