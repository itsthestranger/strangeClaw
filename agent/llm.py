"""LiteLLM wrapper."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ToolCall:
    """Normalized tool call."""

    skill: str
    action: str
    args: dict[str, Any]
    reason: str | None = None


@dataclass(slots=True)
class LLMResponse:
    """Normalized model response."""

    text: str
    action: ToolCall | None
    usage: dict[str, int] | None = None


class LLMClient:
    """Unified model client."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Produce one normalized response."""
        raise NotImplementedError("Backlog task A3 implements LLMClient.")

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token usage for a message list."""
        raise NotImplementedError("Backlog task A3 implements LLMClient.")
