"""OpenAI-compatible HTTP adapter.

Works with any endpoint that speaks the OpenAI Chat Completions API:
  - OpenAI (api.openai.com)
  - Ollama (localhost:11434/v1)
  - vLLM, LM Studio, Together, Groq, etc.

Configuration (via .env or environment variables):
  OPENAI_API_BASE  — base URL (default: https://api.openai.com/v1)
  OPENAI_API_KEY   — API key
  OPENAI_MODEL     — model name (default from settings)
"""

from __future__ import annotations

import os
import logging

from graph_agent.llm.base import ModelAdapter
from graph_agent.settings import get_settings

log = logging.getLogger(__name__)


class OpenAICompatibleAdapter(ModelAdapter):
    """Adapter for any OpenAI-compatible REST endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ):
        s = get_settings().llm
        self._base_url = base_url or os.environ.get("OPENAI_API_BASE", s.base_url)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model or os.environ.get("OPENAI_MODEL", s.model)
        self._temperature = temperature if temperature is not None else s.temperature
        self._max_tokens = max_tokens or s.max_tokens
        self._timeout = timeout or s.timeout_seconds

        # Lazy import — openai is optional for non-LLM tests
        try:
            import openai as _openai
            self._client = _openai.OpenAI(
                api_key=self._api_key or "EMPTY",
                base_url=self._base_url,
                timeout=self._timeout,
            )
        except ImportError as exc:
            raise RuntimeError(
                "openai package not installed. Run: pip install openai"
            ) from exc

    def name(self) -> str:
        return f"openai_compat:{self._model}@{self._base_url}"

    def complete(self, messages: list[dict], **kwargs) -> str:
        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        log.debug("Calling %s model=%s tokens=%d", self._base_url, self._model, max_tokens)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            log.debug("Got %d chars from LLM", len(content))
            return content
        except Exception as exc:
            raise RuntimeError(f"LLM call failed: {exc}") from exc


def get_adapter(adapter_name: str | None = None) -> ModelAdapter:
    """Factory: return the configured adapter by name."""
    from graph_agent.llm.mock import MockAdapter

    name = adapter_name or get_settings().llm.adapter
    if name == "mock":
        return MockAdapter()
    elif name == "openai_compat":
        return OpenAICompatibleAdapter()
    else:
        raise ValueError(f"Unknown adapter: {name!r}. Choose 'mock' or 'openai_compat'.")
