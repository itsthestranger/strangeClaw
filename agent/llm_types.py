"""Shared LLM runtime types with no provider dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ToolCall:
    """Normalized tool call."""

    tool: str
    args: dict[str, Any]
    reason: str | None = None


@dataclass(slots=True)
class LLMResponse:
    """Normalized model response."""

    text: str
    action: ToolCall | None
    usage: dict[str, int] | None = None


class LLMRuntime(Protocol):
    """Minimal runtime contract for agent-facing LLM backends."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

    def count_tokens(self, messages: list[dict[str, Any]]) -> int: ...
