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
    """Normalized model response.

    ``action_error`` states the concrete violation (wrong number of tool calls,
    unparseable arguments, ...) whenever ``action`` is ``None`` and a structured
    decision was requested. It is ``None`` on success and whenever no structured
    decision was asked for.
    """

    text: str
    action: ToolCall | None
    usage: dict[str, int] | None = None
    action_error: str | None = None


class LLMRuntimeError(RuntimeError):
    """Raised when an LLM runtime transport or service call fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class LLMRuntime(Protocol):
    """Minimal runtime contract for agent-facing LLM backends."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

    def count_tokens(self, messages: list[dict[str, Any]]) -> int: ...
