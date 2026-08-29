"""LiteLLM wrapper."""

from __future__ import annotations

import json
from typing import Any

import litellm

from agent.llm_types import LLMResponse, ToolCall

# Appended to every rejection reason so the model is told the rule it broke, not
# just the symptom. The one-decision-per-turn invariant itself never relaxes.
ONE_DECISION_RULE = "Emit exactly one structured decision per turn."

# Longest fragment of a model-authored payload echoed back inside a rejection
# reason. Long enough to be recognisable, short enough not to flood the prompt.
_REASON_PREVIEW_LIMIT = 160


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
        native_tool_choice: str = "required",
        native_fallback_to_prompt: bool = True,
        native_probe: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.provider_settings = provider_settings or {}
        self.structured_output = structured_output
        self.native_tool_choice = native_tool_choice
        self.native_fallback_to_prompt = native_fallback_to_prompt
        self.native_probe = native_probe
        self._native_tool_choice_resolved: str | None = None
        self._native_probe_ran = False

        disallowed_keys = {
            "model",
            "messages",
            "api_key",
            "max_tokens",
            "temperature",
            "timeout",
            "num_retries",
            "tools",
            "tool_choice",
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
        if self.native_tool_choice not in {"forced_function", "required", "auto", "none"}:
            raise ValueError(
                "native_tool_choice must be one of: forced_function, required, auto, none."
            )
        if not isinstance(self.native_fallback_to_prompt, bool):
            raise ValueError("native_fallback_to_prompt must be a boolean.")
        if not isinstance(self.native_probe, bool):
            raise ValueError("native_probe must be a boolean.")

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

        api_base = config.get("api_base")
        if api_base is not None:
            if not isinstance(api_base, str) or not api_base.strip():
                raise ValueError("llm.api_base must be a non-empty string when provided.")
            parsed_provider_settings = dict(parsed_provider_settings)
            parsed_provider_settings.setdefault("api_base", api_base.strip())

        return cls(
            model=str(config["model"]),
            api_key=str(config.get("api_key", "")),
            max_tokens=int(config.get("max_tokens", 4096)),
            temperature=float(config.get("temperature", 0.2)),
            timeout_seconds=float(config.get("timeout_seconds", 60.0)),
            max_retries=int(config.get("max_retries", 2)),
            provider_settings=parsed_provider_settings,
            structured_output=str(config.get("structured_output", "native")),
            native_tool_choice=str(config.get("native_tool_choice", "required")),
            native_fallback_to_prompt=bool(config.get("native_fallback_to_prompt", True)),
            native_probe=bool(config.get("native_probe", True)),
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        """Produce one normalized response."""
        response_action_mode = "none"
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "api_key": self.api_key,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
            "num_retries": self.max_retries,
            **self.provider_settings,
        }

        if action_schema is not None:
            if self.structured_output == "native":
                native_tool_choice = self._resolve_native_tool_choice_mode()
                if native_tool_choice is None:
                    response_action_mode = "prompt"
                    completion_kwargs["messages"] = _inject_prompt_action_schema(
                        messages,
                        action_schema,
                    )
                else:
                    response_action_mode = "native"
                    completion_kwargs.update(
                        _native_tool_call_kwargs(
                            action_schema,
                            tool_choice_mode=native_tool_choice,
                        )
                    )
            else:
                response_action_mode = "prompt"
                completion_kwargs["messages"] = _inject_prompt_action_schema(
                    messages,
                    action_schema,
                )

        response = litellm.completion(**completion_kwargs)
        text = _extract_text(response)
        usage = _extract_usage(response)
        action: ToolCall | None = None
        action_error: str | None = None
        if action_schema is not None:
            if response_action_mode == "native":
                action, action_error = _extract_native_tool_call(response)
            else:
                action, action_error = _extract_prompt_tool_call(text)
        return LLMResponse(
            text=text,
            action=action,
            usage=usage,
            action_error=action_error,
        )

    def _resolve_native_tool_choice_mode(self) -> str | None:
        configured_mode = _required_action_tool_choice_mode(self.native_tool_choice)
        if not self.native_probe:
            return configured_mode
        if self._native_probe_ran:
            return self._native_tool_choice_resolved

        self._native_probe_ran = True
        modes_to_try = _native_tool_choice_probe_order(configured_mode)
        last_client_error: Exception | None = None

        for mode in modes_to_try:
            probe_result = self._probe_native_mode(mode)
            if probe_result is True:
                self._native_tool_choice_resolved = mode
                return mode
            if isinstance(probe_result, Exception):
                last_client_error = probe_result

        self._native_tool_choice_resolved = None
        if self.native_fallback_to_prompt:
            return None
        if last_client_error is not None:
            raise last_client_error
        raise RuntimeError("No compatible native tool-choice mode is available.")

    def _probe_native_mode(self, tool_choice_mode: str) -> bool | Exception:
        probe_messages = [
            {"role": "system", "content": "Tool capability probe. Reply briefly."},
            {"role": "user", "content": "probe"},
        ]
        probe_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": probe_messages,
            "api_key": self.api_key,
            "max_tokens": min(8, self.max_tokens),
            "temperature": 0.0,
            "timeout": min(10.0, self.timeout_seconds),
            "num_retries": 0,
            **self.provider_settings,
            **_native_tool_call_kwargs(
                _probe_action_schema(),
                tool_choice_mode=tool_choice_mode,
            ),
        }
        try:
            response = litellm.completion(**probe_kwargs)
            return _extract_native_tool_call(response)[0] is not None
        except Exception as exc:
            status_code = _exception_status_code(exc)
            if status_code is not None and 400 <= status_code < 500:
                return exc
            raise

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


def _native_tool_call_kwargs(
    action_schema: dict[str, Any] | list[dict[str, Any]],
    *,
    tool_choice_mode: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any]
    if isinstance(action_schema, list):
        tools: list[dict[str, Any]] = []
        for item in action_schema:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            params = item.get("parameters")
            if not isinstance(name, str) or not name.strip() or not isinstance(params, dict):
                continue
            description = item.get("description")
            if not isinstance(description, str):
                description = f"Run tool {name.strip()}."
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name.strip(),
                        "description": description,
                        "parameters": params,
                    },
                }
            )
        kwargs = {"tools": tools}
        if tool_choice_mode == "forced_function":
            kwargs["tool_choice"] = "required"
        else:
            kwargs["tool_choice"] = tool_choice_mode
    else:
        kwargs = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "submit_tool_call",
                        "description": "Emit a structured tool call decision.",
                        "parameters": action_schema,
                    },
                }
            ]
        }
        if tool_choice_mode == "forced_function":
            kwargs["tool_choice"] = {"type": "function", "function": {"name": "submit_tool_call"}}
        else:
            kwargs["tool_choice"] = tool_choice_mode
    return kwargs


def _native_tool_choice_probe_order(configured_mode: str) -> list[str]:
    modes = [configured_mode]
    if configured_mode == "forced_function":
        modes.append("required")
    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def _probe_action_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string"},
            "args": {"type": "object"},
        },
        "required": ["tool", "args"],
        "additionalProperties": False,
    }


def _exception_status_code(exc: Exception) -> int | None:
    for key in ("status_code", "status", "code"):
        value = getattr(exc, key, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    return None


def _extract_native_tool_call(response: Any) -> tuple[ToolCall | None, str | None]:
    """Return the single structured decision, or why the response had none.

    The second element is a model-facing reason naming the concrete violation.
    It is never a substitute for the one-decision-per-turn rule: a response
    carrying several tool calls is still rejected, it is now just told why.
    """
    choices = _get_value(response, "choices")
    if not isinstance(choices, list) or not choices:
        return None, "The model response contained no completion choices."

    first_choice = choices[0]
    message = _get_value(first_choice, "message")
    tool_calls = _get_value(message, "tool_calls")
    # Set when a lone tool call was present but unreadable. The content-block
    # branch below still gets its chance (unchanged behaviour); if it also
    # fails, this more specific reason wins over the generic one.
    pending_reason: str | None = None
    if isinstance(tool_calls, list) and tool_calls:
        if len(tool_calls) != 1:
            return None, _tool_call_count_reason(len(tool_calls))
        first_call = tool_calls[0]
        function_block = _get_value(first_call, "function")
        function_name = _get_value(function_block, "name")
        raw_args = _get_value(function_block, "arguments")
        if isinstance(function_name, str) and function_name and function_name != "submit_tool_call":
            parsed_args = _parse_native_function_args(raw_args)
            if parsed_args is not None:
                return ToolCall(tool=function_name, args=parsed_args, reason=None), None
        payload = _parse_action_payload(raw_args)
        if payload is not None:
            return payload, None
        pending_reason = _unparseable_arguments_reason(function_name, raw_args)

    content = _get_value(message, "content")
    if isinstance(content, list):
        tool_use_items = [
            item
            for item in content
            if isinstance(item, dict) and item.get("type") == "tool_use"
        ]
        if len(tool_use_items) != 1:
            return None, pending_reason or _tool_call_count_reason(len(tool_use_items))
        item = tool_use_items[0]
        input_payload = _get_value(item, "input")
        payload = _action_from_dict(input_payload)
        if payload is None:
            name = item.get("name")
            if isinstance(name, str) and name and isinstance(input_payload, dict):
                payload = ToolCall(tool=name, args=input_payload, reason=None)
        if payload is not None:
            return payload, None
        return None, pending_reason or _unparseable_arguments_reason(
            item.get("name"),
            input_payload,
        )
    return None, pending_reason or _tool_call_count_reason(0)


def _tool_call_count_reason(count: int) -> str:
    if count == 0:
        return f"You emitted no tool calls. {ONE_DECISION_RULE}"
    return f"You emitted {count} tool calls. {ONE_DECISION_RULE}"


def _unparseable_arguments_reason(function_name: Any, raw_args: Any) -> str:
    if isinstance(function_name, str) and function_name:
        subject = f"Your tool call '{function_name}' had"
    else:
        subject = "Your tool call had"
    return (
        f"{subject} arguments that could not be read as a JSON object with a "
        f"string tool and an object args: {_reason_preview(raw_args)}. "
        f"{ONE_DECISION_RULE}"
    )


def _reason_preview(value: Any) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=True, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    text = " ".join(text.split())
    if not text:
        return "(empty)"
    if len(text) > _REASON_PREVIEW_LIMIT:
        return f"{text[:_REASON_PREVIEW_LIMIT]}..."
    return text


def _inject_prompt_action_schema(
    messages: list[dict[str, Any]],
    action_schema: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    schema_text = json.dumps(action_schema, separators=(",", ":"), ensure_ascii=True)
    if isinstance(action_schema, list):
        instruction = (
            "Return exactly one structured action as ONLY a JSON object with keys "
            "tool (string) and args (object) where tool is one of the listed names "
            "and args matches that tool's parameters. Tool definitions: "
            f"{schema_text}. Do not answer in free-form prose."
        )
    else:
        instruction = (
            "Return exactly one structured action as ONLY a JSON object matching this schema: "
            f"{schema_text}. Do not answer in free-form prose."
        )
    return [{"role": "system", "content": instruction}, *messages]


def _extract_prompt_tool_call(text: str) -> tuple[ToolCall | None, str | None]:
    """Return the decision parsed from prose mode, or why there was none."""
    stripped = text.strip()
    if not stripped:
        return None, f"You returned an empty response with no tool call. {ONE_DECISION_RULE}"
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None, (
            "Your response was not valid JSON: "
            f"{_reason_preview(stripped)}. Return only a JSON object with "
            f"a string tool and an object args. {ONE_DECISION_RULE}"
        )
    if not isinstance(payload, dict):
        return None, (
            "Your response parsed as JSON but was not an object with tool and args: "
            f"{_reason_preview(payload)}. {ONE_DECISION_RULE}"
        )
    action = _action_from_dict(payload)
    if action is None:
        return None, (
            "Your JSON response was missing a string tool or an object args: "
            f"{_reason_preview(payload)}. {ONE_DECISION_RULE}"
        )
    return action, None


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


def _parse_native_function_args(raw_args: Any) -> dict[str, Any] | None:
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str):
        return None
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _action_from_dict(payload: Any) -> ToolCall | None:
    if not isinstance(payload, dict):
        return None
    tool = payload.get("tool")
    args = payload.get("args")
    reason = payload.get("reason")
    if not isinstance(tool, str) or not isinstance(args, dict):
        return None
    if reason is not None and not isinstance(reason, str):
        reason = None
    return ToolCall(tool=tool, args=args, reason=reason)


def _required_action_tool_choice_mode(configured_mode: str) -> str:
    if configured_mode in {"required", "forced_function"}:
        return configured_mode
    return "required"


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
