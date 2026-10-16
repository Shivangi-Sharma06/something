"""SQLAlchemy engine + session factory.

ADR: We use a module-level engine singleton (created lazily) rather than
per-request connections.  The pool is configured from Settings.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from graph_agent.settings import get_settings

_engine = None
_SessionLocal = None


def get_engine():
    """Return (creating if necessary) the global SQLAlchemy engine."""
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_engine(
            s.database.url,
            pool_size=s.database.pool_size,
            max_overflow=s.database.max_overflow,
            future=True,
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a transactional session; rolls back on exception."""
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Return True if the database is reachable."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def override_engine(engine) -> None:
    """Replace the global engine (used in tests with a temp DB)."""
    global _engine, _SessionLocal
    _engine = engine
    _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
