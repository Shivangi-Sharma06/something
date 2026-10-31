"""Ollama local model adapter and interactive model selector.

Provides:
- list_ollama_models: queries local Ollama tags API for installed models
- select_ollama_model: renders interactive terminal dropdown for model selection
- OllamaAdapter: ModelAdapter implementation talking to local Ollama (/v1)
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any, Iterator
import urllib.request
import json

from rich.console import Console
from rich.prompt import Prompt

from graph_agent.llm.base import ModelAdapter
from graph_agent.llm.openai_compat import OpenAICompatibleAdapter
from graph_agent.settings import get_settings

log = logging.getLogger(__name__)
console = Console()

DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "gptoss:20b"


def get_ollama_host_url(base_url: str | None = None) -> str:
    """Normalize base URL to host URL (stripping /v1 for native api tags)."""
    raw = (
        base_url
        or os.environ.get("OLLAMA_API_BASE")
        or os.environ.get("OLLAMA_HOST")
        or os.environ.get("OPENAI_API_BASE")
        or DEFAULT_OLLAMA_BASE
    )
    if raw.endswith("/v1"):
        raw = raw[:-3]
    if raw.endswith("/v1/"):
        raw = raw[:-4]
    return raw.rstrip("/")


def list_ollama_models(host_url: str | None = None, timeout: float = 2.0) -> list[str]:
    """Query local Ollama instance for installed models via /api/tags."""
    host = get_ollama_host_url(host_url)
    tags_url = f"{host}/api/tags"

    try:
        req = urllib.request.Request(tags_url, headers={"User-Agent": "graph-agent"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") or m.get("model") for m in data.get("models", []) if (m.get("name") or m.get("model"))]
            return models
    except Exception as exc:
        log.debug("Could not query Ollama models at %s: %s", tags_url, exc)
        return []


def select_ollama_model(
    default_model: str = DEFAULT_OLLAMA_MODEL,
    host_url: str | None = None,
    interactive: bool = True,
) -> str:
    """Renders a terminal dropdown to choose among local Ollama models on the machine.
    
    If gptoss:20b is configured, it is featured as default choice.
    """
    configured_model = (
        os.environ.get("OLLAMA_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or default_model
    )
    installed_models = list_ollama_models(host_url)

    # Build unique ordered choices
    choices: list[str] = []
    if configured_model not in installed_models:
        choices.append(configured_model)
    choices.extend(installed_models)

    # Ensure gptoss:20b is always an available option
    if "gptoss:20b" not in choices:
        choices.append("gptoss:20b")

    # Dedup while preserving order
    seen = set()
    unique_choices = []
    for c in choices:
        if c and c not in seen:
            seen.add(c)
            unique_choices.append(c)

    custom_option = "➕ [Enter custom model name]"
    menu_choices = unique_choices + [custom_option]

    # Non-interactive or piped input: return configured model
    if not interactive or not sys.stdin.isatty():
        return configured_model

    # Interactive dropdown using questionary (with arrow-key navigation)
    try:
        import questionary
        console.print("[bold cyan]Select local Ollama model to use:[/bold cyan]")
        choice = questionary.select(
            "Local Ollama Model:",
            choices=menu_choices,
            default=configured_model if configured_model in unique_choices else unique_choices[0],
            instruction="(Use arrow keys ↑/↓, Enter to confirm)",
        ).ask()

        if choice is None:  # User pressed Ctrl+C or Esc
            return configured_model

        if choice == custom_option:
            typed = questionary.text("Enter model name (e.g. gptoss:20b, llama3):").ask()
            return typed.strip() if typed and typed.strip() else configured_model

        return choice
    except Exception as exc:
        log.debug("questionary selection failed, falling back to rich prompt: %s", exc)

    # Fallback to rich numbered prompt if questionary is unavailable or fails
    console.print("\n[bold cyan]Available Local Ollama Models:[/bold cyan]")
    for i, m in enumerate(unique_choices, 1):
        def_tag = " [green](default)[/green]" if m == configured_model else ""
        console.print(f"  {i}) [bold]{m}[/bold]{def_tag}")
    console.print(f"  {len(unique_choices) + 1}) Custom model name")

    val = Prompt.ask(
        "Enter choice number or model name",
        default="1",
    ).strip()

    if val.isdigit() and 1 <= int(val) <= len(unique_choices):
        return unique_choices[int(val) - 1]
    elif val.isdigit() and int(val) == len(unique_choices) + 1:
        custom = Prompt.ask("Enter model name").strip()
        return custom if custom else configured_model
    elif val in unique_choices:
        return val
    elif val:
        return val

    return configured_model


class OllamaAdapter(OpenAICompatibleAdapter):
    """Model adapter dedicated to local Ollama instance via its /v1 endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ):
        s = get_settings().llm
        url = (
            base_url
            or os.environ.get("OLLAMA_API_BASE")
            or os.environ.get("OLLAMA_HOST")
            or os.environ.get("OPENAI_API_BASE")
            or DEFAULT_OLLAMA_BASE
        )
        if not url.endswith("/v1"):
            url = f"{url.rstrip('/')}/v1"

        mod = (
            model
            or os.environ.get("OLLAMA_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or DEFAULT_OLLAMA_MODEL
        )

        super().__init__(
            base_url=url,
            api_key="ollama",
            model=mod,
            temperature=temperature if temperature is not None else s.temperature,
            max_tokens=max_tokens or s.max_tokens,
            timeout=timeout or s.timeout_seconds,
        )

    def name(self) -> str:
        return f"ollama:{self._model}@{self._base_url}"

    def complete(self, messages: list[dict], **kwargs) -> str:
        try:
            return super().complete(messages, **kwargs)
        except Exception as exc:
            err_msg = str(exc)
            if "Connection refused" in err_msg or "Failed to connect" in err_msg or "ConnectError" in err_msg:
                raise RuntimeError(
                    f"Ollama server is not running at {self._base_url}. "
                    "Please start Ollama with 'ollama serve' or check 'systemctl status ollama'."
                ) from exc
            if "not found" in err_msg.lower() or "404" in err_msg:
                raise RuntimeError(
                    f"Model '{self._model}' was not found in Ollama. "
                    f"Run: ollama pull {self._model} (or ollama create {self._model})"
                ) from exc
            raise

    def stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        try:
            yield from super().stream(messages, **kwargs)
        except Exception as exc:
            err_msg = str(exc)
            if "Connection refused" in err_msg or "Failed to connect" in err_msg or "ConnectError" in err_msg:
                raise RuntimeError(
                    f"Ollama server is not running at {self._base_url}. "
                    "Please start Ollama with 'ollama serve' or check 'systemctl status ollama'."
                ) from exc
            if "not found" in err_msg.lower() or "404" in err_msg:
                raise RuntimeError(
                    f"Model '{self._model}' was not found in Ollama. "
                    f"Run: ollama pull {self._model} (or ollama create {self._model})"
                ) from exc
            raise
