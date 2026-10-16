"""ModelAdapter ABC — common interface for all LLM backends.

Every adapter must implement:
  - name() → str
  - complete(messages, **kwargs) → str

ADR-5: Adapters receive raw OpenAI-style message dicts so the prompt
construction layer is adapter-agnostic.  Streaming is not required for
the repair loop (we need the full diff before applying anything).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """Abstract base class for LLM model adapters."""

    @abstractmethod
    def name(self) -> str:
        """Return a human-readable identifier for this adapter/model."""
        ...

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        """Send *messages* to the model and return the response text.

        Args:
            messages: OpenAI-style list of {role, content} dicts.
            **kwargs: Adapter-specific overrides (temperature, max_tokens, etc.)

        Returns:
            The model's text response.

        Raises:
            RuntimeError: on API errors or timeouts.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name()!r})"
