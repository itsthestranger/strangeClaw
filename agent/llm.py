"""LiteLLM wrapper."""

from __future__ import annotations

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
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.provider_settings = provider_settings or {}

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
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Produce one normalized response."""
        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=self.timeout_seconds,
            num_retries=self.max_retries,
            **self.provider_settings,
        )
        text = _extract_text(response)
        usage = _extract_usage(response)
        _ = action_schema
        return LLMResponse(text=text, action=None, usage=usage)

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
