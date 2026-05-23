"""Host-side LLM service for Fire-mode proxy calls."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeGuard, cast

from agent.llm import LLMClient
from agent.llm_types import LLMRuntime, ToolCall

LOGGER = logging.getLogger(__name__)

DEFAULT_LLM_MAX_REQUEST_BYTES = 2 * 1024 * 1024


class LLMService:
    """Host service wrapper around an LLM runtime."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        llm_client: LLMRuntime | None = None,
    ) -> None:
        self._config = config
        self._api_key = _llm_api_key(config)
        self._max_request_bytes = _llm_max_request_bytes(config)
        self._llm = llm_client or LLMClient.from_config(_llm_config(config))

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Handle one LLM host-service request."""
        try:
            request_size = _payload_size_bytes(payload)
            if request_size > self._max_request_bytes:
                return self._failure(
                    f"llm request too large: {request_size} bytes exceeds "
                    f"{self._max_request_bytes} bytes"
                )

            action = payload.get("action")
            if action == "complete":
                return self._handle_complete(payload)
            if action == "count_tokens":
                return self._handle_count_tokens(payload)
            return self._failure(f"unknown action: {action}")
        except Exception as exc:
            error = self._redact(str(exc))
            LOGGER.warning("LLM service request failed: %s", error)
            return self._failure(error)

    def _handle_complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if not _is_message_list(messages):
            return self._failure("invalid_request: messages must be a list of objects")
        action_schema = cast(
            dict[str, Any] | list[dict[str, Any]] | None,
            payload.get("action_schema"),
        )

        response = self._llm.complete(messages, action_schema=action_schema)
        return {
            "success": True,
            "text": response.text,
            "action": _serialize_tool_call(response.action),
            "usage": response.usage,
        }

    def _handle_count_tokens(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        if not _is_message_list(messages):
            return self._failure("invalid_request: messages must be a list of objects")

        return {"success": True, "tokens": self._llm.count_tokens(messages)}

    def _failure(self, error: str) -> dict[str, Any]:
        return {"success": False, "error": self._redact(error)}

    def _redact(self, text: str) -> str:
        if not self._api_key:
            return text
        return text.replace(self._api_key, "[REDACTED]")


def _llm_config(config: dict[str, Any]) -> dict[str, Any]:
    llm_config = config.get("llm", {})
    return llm_config if isinstance(llm_config, dict) else {}


def _llm_api_key(config: dict[str, Any]) -> str:
    api_key = _llm_config(config).get("api_key", "")
    return api_key if isinstance(api_key, str) else str(api_key)


def _llm_max_request_bytes(config: dict[str, Any]) -> int:
    host_services = config.get("host_services", {})
    if not isinstance(host_services, dict):
        return DEFAULT_LLM_MAX_REQUEST_BYTES
    raw_value = host_services.get("llm_max_request_bytes", DEFAULT_LLM_MAX_REQUEST_BYTES)
    return int(raw_value)


def _payload_size_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _is_message_list(value: Any) -> TypeGuard[list[dict[str, Any]]]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _serialize_tool_call(action: ToolCall | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {"tool": action.tool, "args": action.args, "reason": action.reason}
