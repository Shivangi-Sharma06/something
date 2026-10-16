"""SQLAlchemy ORM models mirroring the graph_agent PostgreSQL schema."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, Column, Float, ForeignKey,
    Integer, String, Text, TIMESTAMP, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class Repository(Base):
    __tablename__ = "repositories"
    __table_args__ = {"schema": "graph_agent"}

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    name: str = Column(String(255), nullable=False)
    root_path: str = Column(Text, nullable=False, unique=True)
    vcs_type: str = Column(String(32), nullable=False, server_default="git")
    head_commit: Optional[str] = Column(String(64))
    indexed_at: datetime = Column(TIMESTAMP(timezone=True), server_default=func.now())
    metadata_: dict = Column("metadata", JSONB, server_default="{}")

    files = relationship("File", back_populates="repository", cascade="all, delete-orphan")
    symbols = relationship("Symbol", back_populates="repository", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="repository", cascade="all, delete-orphan")
    bug_reports = relationship("BugReport", back_populates="repository", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "files"
    __table_args__ = (
        UniqueConstraint("repo_id", "rel_path", name="uq_file_repo_path"),
        {"schema": "graph_agent"},
    )

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    repo_id: int = Column(Integer, ForeignKey("graph_agent.repositories.id", ondelete="CASCADE"), nullable=False)
    rel_path: str = Column(Text, nullable=False)
    language: Optional[str] = Column(String(32))
    content_hash: str = Column(String(64), nullable=False)
    size_bytes: Optional[int] = Column(Integer)
    indexed_at: datetime = Column(TIMESTAMP(timezone=True), server_default=func.now())

    repository = relationship("Repository", back_populates="files")
    symbols = relationship("Symbol", back_populates="file", cascade="all, delete-orphan")


class Symbol(Base):
    __tablename__ = "symbols"
    __table_args__ = {"schema": "graph_agent"}

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    file_id: int = Column(Integer, ForeignKey("graph_agent.files.id", ondelete="CASCADE"), nullable=False)
    repo_id: int = Column(Integer, ForeignKey("graph_agent.repositories.id", ondelete="CASCADE"), nullable=False)
    kind: str = Column(String(32), nullable=False)   # function|class|method|import|parameter
    name: str = Column(Text, nullable=False)
    qualified_name: str = Column(Text, nullable=False)
    parent_id: Optional[int] = Column(Integer, ForeignKey("graph_agent.symbols.id"), nullable=True)
    line_start: Optional[int] = Column(Integer)
    line_end: Optional[int] = Column(Integer)
    byte_start: Optional[int] = Column(Integer)
    byte_end: Optional[int] = Column(Integer)
    signature: Optional[str] = Column(Text)
    docstring: Optional[str] = Column(Text)
    metadata_: dict = Column("metadata", JSONB, server_default="{}")

    file = relationship("File", back_populates="symbols")
    repository = relationship("Repository", back_populates="symbols")
    # Self-referential adjacency list: one parent Symbol → many child Symbols.
    # `remote_side=[id]` marks `id` as the "remote" (parent) end so SA knows
    # `parent_id` is the FK on the child (many) side.
    parent = relationship(
        "Symbol",
        back_populates="children",
        foreign_keys=[parent_id],
        remote_side="Symbol.id",
    )
    children = relationship(
        "Symbol",
        back_populates="parent",
        foreign_keys="[Symbol.parent_id]",
    )
    outgoing_edges = relationship("Edge", foreign_keys="Edge.source_id", back_populates="source", cascade="all, delete-orphan")
    incoming_edges = relationship("Edge", foreign_keys="Edge.target_id", back_populates="target")
    unresolved_refs = relationship("UnresolvedRef", back_populates="source_symbol", cascade="all, delete-orphan")


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = {"schema": "graph_agent"}

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    repo_id: int = Column(Integer, ForeignKey("graph_agent.repositories.id", ondelete="CASCADE"), nullable=False)
    source_id: int = Column(Integer, ForeignKey("graph_agent.symbols.id", ondelete="CASCADE"), nullable=False)
    target_id: int = Column(Integer, ForeignKey("graph_agent.symbols.id", ondelete="CASCADE"), nullable=False)
    edge_type: str = Column(String(32), nullable=False)  # CONTAINS|IMPORTS|CALLS|INHERITS|HANDLES|HAS_TYPE|TESTS
    confidence: float = Column(Float, nullable=False, server_default="1.0")
    resolution_method: Optional[str] = Column(String(64))
    metadata_: dict = Column("metadata", JSONB, server_default="{}")

    repository = relationship("Repository", back_populates="edges")
    source = relationship("Symbol", foreign_keys=[source_id], back_populates="outgoing_edges")
    target = relationship("Symbol", foreign_keys=[target_id], back_populates="incoming_edges")


class UnresolvedRef(Base):
    __tablename__ = "unresolved_refs"
    __table_args__ = {"schema": "graph_agent"}

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    source_id: int = Column(Integer, ForeignKey("graph_agent.symbols.id", ondelete="CASCADE"), nullable=False)
    ref_name: str = Column(Text, nullable=False)
    ref_kind: str = Column(String(32), nullable=False, server_default="call")
    context: Optional[str] = Column(Text)

    source_symbol = relationship("Symbol", back_populates="unresolved_refs")


class BugReport(Base):
    __tablename__ = "bug_reports"
    __table_args__ = {"schema": "graph_agent"}

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    repo_id: int = Column(Integer, ForeignKey("graph_agent.repositories.id", ondelete="CASCADE"), nullable=False)
    title: Optional[str] = Column(Text)
    description: Optional[str] = Column(Text)
    stack_trace: Optional[str] = Column(Text)
    failing_test: Optional[str] = Column(Text)
    created_at: datetime = Column(TIMESTAMP(timezone=True), server_default=func.now())
    metadata_: dict = Column("metadata", JSONB, server_default="{}")

    repository = relationship("Repository", back_populates="bug_reports")
    repair_attempts = relationship("RepairAttempt", back_populates="bug_report", cascade="all, delete-orphan")


class RepairAttempt(Base):
    __tablename__ = "repair_attempts"
    __table_args__ = {"schema": "graph_agent"}

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    bug_report_id: int = Column(Integer, ForeignKey("graph_agent.bug_reports.id"), nullable=False)
    attempt_num: int = Column(Integer, nullable=False)
    patch_diff: Optional[str] = Column(Text)
    model_used: Optional[str] = Column(String(128))
    root_cause: Optional[str] = Column(Text)
    test_results: Optional[dict] = Column(JSONB)
    status: str = Column(String(32), nullable=False, server_default="pending")
    created_at: datetime = Column(TIMESTAMP(timezone=True), server_default=func.now())

    bug_report = relationship("BugReport", back_populates="repair_attempts")
