"""Unit tests for ModelAdapter implementations."""

from __future__ import annotations

import pytest
from graph_agent.llm.mock import MockAdapter, DEFAULT_MOCK_DIFF, DEFAULT_MOCK_ROOT_CAUSE
from graph_agent.llm.base import ModelAdapter


class TestModelAdapterInterface:
    def test_mock_is_adapter(self):
        adapter = MockAdapter()
        assert isinstance(adapter, ModelAdapter)

    def test_mock_name(self):
        adapter = MockAdapter()
        assert adapter.name() == "mock"

    def test_mock_returns_string(self):
        adapter = MockAdapter()
        result = adapter.complete([{"role": "user", "content": "fix the bug"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_contains_diff(self):
        adapter = MockAdapter()
        result = adapter.complete([{"role": "user", "content": "fix bug"}])
        assert "```diff" in result or "---" in result

    def test_mock_custom_response(self):
        adapter = MockAdapter(response="CUSTOM_RESPONSE")
        result = adapter.complete([])
        assert result == "CUSTOM_RESPONSE"

    def test_mock_custom_diff(self):
        custom_diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x\n+y"
        adapter = MockAdapter(diff=custom_diff)
        result = adapter.complete([])
        assert custom_diff in result

    def test_mock_ignores_kwargs(self):
        """MockAdapter must not raise on extra kwargs."""
        adapter = MockAdapter()
        result = adapter.complete([], temperature=0.5, max_tokens=100)
        assert isinstance(result, str)

    def test_repr(self):
        adapter = MockAdapter()
        assert "MockAdapter" in repr(adapter)


class TestAdapterFactory:
    def test_get_mock_adapter(self):
        from graph_agent.llm.openai_compat import get_adapter
        adapter = get_adapter("mock")
        assert adapter.name() == "mock"

    def test_unknown_adapter_raises(self):
        from graph_agent.llm.openai_compat import get_adapter
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("nonexistent_adapter")
