"""Guest-side LLM runtime that forwards calls to a host service."""

from __future__ import annotations

from typing import Any

from agent.broker_client import BrokerClient, HostServiceError
from agent.llm_types import LLMResponse, LLMRuntimeError, ToolCall


class LLMProxyRuntime:
    """Thin LLMRuntime implementation backed by the host-side ``llm`` service."""

    def __init__(self, client: BrokerClient) -> None:
        self._client = client

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        try:
            response = self._client.call(
                "llm",
                {
                    "action": "complete",
                    "messages": messages,
                    "action_schema": action_schema,
                },
            )
        except HostServiceError as exc:
            raise LLMRuntimeError(str(exc)) from exc

        success = response.get("success")
        if success is False:
            raise LLMRuntimeError(str(response.get("error", "llm complete failed")))
        if success is not True:
            raise LLMRuntimeError("llm complete response missing success envelope")
        return LLMResponse(
            text=str(response.get("text", "")),
            action=_deserialize_tool_call(response.get("action")),
            usage=_deserialize_usage(response.get("usage")),
        )

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        try:
            response = self._client.call(
                "llm",
                {"action": "count_tokens", "messages": messages},
            )
        except HostServiceError as exc:
            raise LLMRuntimeError(str(exc)) from exc

        success = response.get("success")
        if success is False:
            raise LLMRuntimeError(str(response.get("error", "llm count_tokens failed")))
        if success is not True:
            raise LLMRuntimeError("llm count_tokens response missing success envelope")
        tokens = response.get("tokens")
        if not isinstance(tokens, int):
            raise LLMRuntimeError("llm count_tokens response missing integer tokens")
        return tokens


def _deserialize_tool_call(value: Any) -> ToolCall | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise LLMRuntimeError("llm complete response action must be an object or null")

    tool = value.get("tool")
    args = value.get("args")
    reason = value.get("reason")
    if not isinstance(tool, str) or not isinstance(args, dict):
        raise LLMRuntimeError("llm complete response action must contain tool and args")
    if reason is not None and not isinstance(reason, str):
        raise LLMRuntimeError("llm complete response action reason must be a string")
    return ToolCall(tool=tool, args=args, reason=reason)


def _deserialize_usage(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise LLMRuntimeError("llm complete response usage must be an object or null")
    usage: dict[str, int] = {}
    for key, raw_count in value.items():
        if not isinstance(key, str) or not isinstance(raw_count, int):
            raise LLMRuntimeError("llm complete response usage values must be integers")
        usage[key] = raw_count
    return usage
