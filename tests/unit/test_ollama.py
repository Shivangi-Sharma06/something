"""Unit tests for Ollama adapter and model selector."""

from __future__ import annotations

import io
import json
from unittest.mock import patch, MagicMock

import pytest

from graph_agent.llm.ollama import (
    OllamaAdapter,
    list_ollama_models,
    select_ollama_model,
    get_ollama_host_url,
    DEFAULT_OLLAMA_MODEL,
)
from graph_agent.llm.openai_compat import get_adapter


def test_get_ollama_host_url():
    assert get_ollama_host_url("http://localhost:11434/v1") == "http://localhost:11434"
    assert get_ollama_host_url("http://localhost:11434/v1/") == "http://localhost:11434"
    assert get_ollama_host_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"


def test_list_ollama_models_offline():
    # If no server is running at bogus port, returns empty list
    models = list_ollama_models(host_url="http://127.0.0.1:59999", timeout=0.1)
    assert models == []


def test_list_ollama_models_mocked():
    fake_response = json.dumps({
        "models": [
            {"name": "gptoss:20b"},
            {"name": "codellama:7b"},
        ]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = list_ollama_models()
        assert models == ["gptoss:20b", "codellama:7b"]


def test_select_ollama_model_non_interactive():
    # In non-interactive mode, select_ollama_model returns the configured / default model
    model = select_ollama_model(default_model="gptoss:20b", interactive=False)
    assert model == "gptoss:20b"


def test_ollama_adapter_initialization():
    adapter = OllamaAdapter(model="gptoss:20b", base_url="http://localhost:11434/v1")
    assert "ollama:gptoss:20b" in adapter.name()
    assert adapter._model == "gptoss:20b"
    assert adapter._base_url == "http://localhost:11434/v1"


def test_get_adapter_factory_for_ollama():
    adapter = get_adapter("ollama", model="gptoss:20b")
    assert isinstance(adapter, OllamaAdapter)
    assert "gptoss:20b" in adapter.name()
