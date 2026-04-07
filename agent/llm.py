"""LiteLLM wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import litellm


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

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        provider_settings: dict[str, Any] | None = None,
        structured_output: str = "native",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.provider_settings = provider_settings or {}
        self.structured_output = structured_output

        disallowed_keys = {
            "model",
            "messages",
            "api_key",
            "max_tokens",
            "temperature",
            "timeout",
            "num_retries",
        }
        conflicts = sorted(key for key in self.provider_settings if key in disallowed_keys)
        if conflicts:
            conflict_str = ", ".join(conflicts)
            raise ValueError(
                "provider_settings contains reserved keys that must be configured on LLMClient: "
                f"{conflict_str}"
            )
        if self.structured_output not in {"native", "prompt"}:
            raise ValueError("structured_output must be either 'native' or 'prompt'.")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> LLMClient:
        """Construct a client from the `llm` section of config."""
        provider_settings = config.get("provider_settings")
        if provider_settings is None:
            parsed_provider_settings: dict[str, Any] = {}
        elif isinstance(provider_settings, dict):
            parsed_provider_settings = provider_settings
        else:
            raise ValueError("llm.provider_settings must be an object when provided.")

        return cls(
            model=str(config["model"]),
            api_key=str(config.get("api_key", "")),
            max_tokens=int(config.get("max_tokens", 4096)),
            temperature=float(config.get("temperature", 0.2)),
            timeout_seconds=float(config.get("timeout_seconds", 60.0)),
            max_retries=int(config.get("max_retries", 2)),
            provider_settings=parsed_provider_settings,
            structured_output=str(config.get("structured_output", "native")),
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Produce one normalized response."""
        request_messages = messages
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "api_key": self.api_key,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
            "num_retries": self.max_retries,
            **self.provider_settings,
        }

        if action_schema is not None:
            if self.structured_output == "native":
                completion_kwargs.update(_native_tool_call_kwargs(action_schema))
            else:
                request_messages = _inject_prompt_action_schema(messages, action_schema)
                completion_kwargs["messages"] = request_messages

        response = litellm.completion(**completion_kwargs)
        text = _extract_text(response)
        usage = _extract_usage(response)
        action: ToolCall | None = None
        if action_schema is not None:
            if self.structured_output == "native":
                action = _extract_native_tool_call(response)
            else:
                action = _extract_prompt_tool_call(text)
        return LLMResponse(text=text, action=action, usage=usage)

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token usage for a message list."""
        try:
            return int(
                litellm.token_counter(
                    model=self.model,
                    messages=messages,
                    **self.provider_settings,
                )
            )
        except Exception:
            return _heuristic_token_count(messages)


def _extract_text(response: Any) -> str:
    choices = _get_value(response, "choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    message = _get_value(first_choice, "message")
    content = _get_value(message, "content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(part for part in parts if isinstance(part, str))
    return ""


def _native_tool_call_kwargs(action_schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "submit_tool_call",
                    "description": "Emit a structured tool call decision.",
                    "parameters": action_schema,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "submit_tool_call"}},
    }


def _extract_native_tool_call(response: Any) -> ToolCall | None:
    choices = _get_value(response, "choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    message = _get_value(first_choice, "message")
    tool_calls = _get_value(message, "tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        first_call = tool_calls[0]
        function_block = _get_value(first_call, "function")
        raw_args = _get_value(function_block, "arguments")
        payload = _parse_action_payload(raw_args)
        if payload is not None:
            return payload

    content = _get_value(message, "content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "tool_use":
                continue
            payload = _action_from_dict(_get_value(item, "input"))
            if payload is not None:
                return payload
    return None


def _inject_prompt_action_schema(
    messages: list[dict[str, Any]],
    action_schema: dict[str, Any],
) -> list[dict[str, Any]]:
    schema_text = json.dumps(action_schema, separators=(",", ":"), ensure_ascii=True)
    instruction = (
        "When selecting a tool, return ONLY a JSON object matching this schema: "
        f"{schema_text}. If no tool is needed, respond normally."
    )
    return [{"role": "system", "content": instruction}, *messages]


def _extract_prompt_tool_call(text: str) -> ToolCall | None:
    payload = _extract_first_json_object(text)
    if payload is None:
        return None
    return _action_from_dict(payload)


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _parse_action_payload(raw_args: Any) -> ToolCall | None:
    if isinstance(raw_args, dict):
        return _action_from_dict(raw_args)
    if not isinstance(raw_args, str):
        return None
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        return None
    return _action_from_dict(parsed)


def _action_from_dict(payload: Any) -> ToolCall | None:
    if not isinstance(payload, dict):
        return None
    skill = payload.get("skill")
    action = payload.get("action")
    args = payload.get("args")
    reason = payload.get("reason")
    if not isinstance(skill, str) or not isinstance(action, str) or not isinstance(args, dict):
        return None
    if reason is not None and not isinstance(reason, str):
        reason = None
    return ToolCall(skill=skill, action=action, args=args, reason=reason)


def _extract_usage(response: Any) -> dict[str, int] | None:
    usage_obj = _get_value(response, "usage")
    if usage_obj is None:
        return None

    prompt_tokens = _get_value(usage_obj, "prompt_tokens")
    completion_tokens = _get_value(usage_obj, "completion_tokens")
    total_tokens = _get_value(usage_obj, "total_tokens")

    usage: dict[str, int] = {}
    if isinstance(prompt_tokens, int):
        usage["prompt_tokens"] = prompt_tokens
    if isinstance(completion_tokens, int):
        usage["completion_tokens"] = completion_tokens
    if isinstance(total_tokens, int):
        usage["total_tokens"] = total_tokens

    return usage or None


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _heuristic_token_count(messages: list[dict[str, Any]]) -> int:
    total_chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        total_chars += len(text)
    # Rough approximation used only when provider-specific counting isn't available.
    return max(1, total_chars // 4)
