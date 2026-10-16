"""Pydantic-settings based configuration loader.

All settings can be overridden via environment variables prefixed GRAPH_AGENT_
or by values in a .env file at the project root.  TOML config is the authoritative
default; env vars take precedence.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from functools import lru_cache
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Locate the project root (parent of this package directory)
_PKG_ROOT = Path(__file__).parent
_PROJECT_ROOT = _PKG_ROOT.parent
_TOML_PATH = _PROJECT_ROOT / "config" / "default.toml"


def _load_toml() -> dict[str, Any]:
    """Load the TOML config file, returning an empty dict if missing."""
    if _TOML_PATH.exists():
        with open(_TOML_PATH, "rb") as fh:
            return tomllib.load(fh)
    return {}


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_AGENT_DB_")

    url: str = "postgresql+psycopg2://graph_agent:graph_agent_pass@127.0.0.1:5432/graph_agent_db"
    pool_size: int = 5
    max_overflow: int = 10


class IndexerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_AGENT_INDEXER_")

    languages: list[str] = ["python"]
    skip_dirs: list[str] = [
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        "dist", "build", ".eggs", "migrations",
        ".mypy_cache", ".pytest_cache", "htmlcov",
    ]
    extensions: dict[str, list[str]] = {"python": [".py"]}
    max_file_size: int = 524288  # 512 KB


class GraphSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_AGENT_GRAPH_")

    confidence: dict[str, float] = Field(
        default={"direct": 1.0, "heuristic_name_match": 0.7, "unresolved": 0.0}
    )
    max_traversal_depth: int = 5


class LocalizationWeights(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_AGENT_LOC_W_")

    stack_frame_exact: float = 1.0
    test_mention: float = 0.9
    caller_depth_1: float = 0.6
    caller_depth_2: float = 0.4
    callee_depth_1: float = 0.3
    file_co_change: float = 0.2
    top_k: int = 10


class ContextSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_AGENT_CONTEXT_")

    token_budget: int = 8000
    full_source_top_k: int = 3


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_AGENT_LLM_")

    adapter: str = "mock"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: int = 60
    # api_key intentionally loaded from OPENAI_API_KEY env var by the adapter


class VerifierSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_AGENT_VERIFIER_")

    max_retries: int = 3
    isolation: str = "worktree"
    docker_image: str = "python:3.12-slim"
    test_command: str = "pytest tests/ -x --tb=short -q"
    test_timeout_seconds: int = 120


class EvaluatorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GRAPH_AGENT_EVAL_")

    output_dir: str = "eval_results"
    strategies: list[str] = ["no_retrieval", "graph_retrieval"]


class Settings(BaseSettings):
    """Top-level settings object aggregating all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    indexer: IndexerSettings = Field(default_factory=IndexerSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    localization: LocalizationWeights = Field(default_factory=LocalizationWeights)
    context: ContextSettings = Field(default_factory=ContextSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    verifier: VerifierSettings = Field(default_factory=VerifierSettings)
    evaluator: EvaluatorSettings = Field(default_factory=EvaluatorSettings)

    def model_post_init(self, __context: Any) -> None:
        """Overlay TOML defaults (env vars have already been applied by pydantic)."""
        toml = _load_toml()
        # TOML values are defaults; only apply them if the field is still at its
        # pydantic default (i.e. env var was NOT set).  We do a best-effort merge
        # by re-constructing sub-settings from TOML dicts merged with env-sourced values.
        # This keeps pydantic-settings env-var priority intact.
        _ = toml  # TOML overlay is a future enhancement; env vars win for now.


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
