"""Tests for LiteLLM client wrapper."""

from __future__ import annotations

import re
from typing import Any

import pytest

from agent.llm import LLMClient

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["tool", "args"],
}

TOOLS_SCHEMA = [
    {
        "name": "shell",
        "description": "Run shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    }
]

EXECUTION_SURFACE_SCHEMA = [
    {
        "name": "shell",
        "description": "Run shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
    {
        "name": "agent_done",
        "description": "Finish execution.",
        "parameters": {
            "type": "object",
            "properties": {"reply": {"type": "string"}},
            "required": ["reply"],
            "additionalProperties": False,
        },
    },
]


class _FakeClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_complete_normalizes_text_and_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [{"message": {"content": "hello from model"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)

    client = LLMClient(
        model="openai/gpt-4.1-mini",
        api_key="sk-test",
        max_tokens=123,
        temperature=0.4,
        timeout_seconds=12.5,
        max_retries=5,
        provider_settings={"api_base": "http://127.0.0.1:1234/v1"},
    )
    result = client.complete(messages=[{"role": "user", "content": "hello"}])

    assert result.text == "hello from model"
    assert result.action is None
    assert result.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    assert captured["api_key"] == "sk-test"
    assert captured["model"] == "openai/gpt-4.1-mini"
    assert captured["timeout"] == 12.5
    assert captured["num_retries"] == 5
    assert captured["api_base"] == "http://127.0.0.1:1234/v1"


def test_complete_handles_list_content(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**_: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "alpha "},
                            {"type": "text", "text": "beta"},
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(model="anthropic/claude-sonnet-4", api_key="sk-test")

    result = client.complete(messages=[{"role": "user", "content": "hello"}])

    assert result.text == "alpha beta"
    assert result.usage is None


def test_native_structured_output_parses_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "submit_tool_call",
                                    "arguments": (
                                        '{"tool":"shell.run","args":{"command":"pwd"},'
                                        '"reason":"Need cwd"}'
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(
        model="openai/gpt-4.1-mini",
        api_key="sk-test",
        structured_output="native",
        native_tool_choice="forced_function",
        native_probe=False,
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Use a tool"}],
        action_schema=ACTION_SCHEMA,
    )

    assert result.action is not None
    assert result.action.tool == "shell.run"
    assert result.action.args == {"command": "pwd"}
    assert result.action.reason == "Need cwd"
    assert "tools" in captured
    assert captured["tool_choice"]["function"]["name"] == "submit_tool_call"


def test_native_structured_output_parses_direct_tool_function_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"command":"pwd"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(
        model="openai/gpt-4.1-mini",
        api_key="sk-test",
        structured_output="native",
        native_tool_choice="required",
        native_probe=False,
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Use a tool"}],
        action_schema=TOOLS_SCHEMA,
    )

    assert result.action is not None
    assert result.action.tool == "shell"
    assert result.action.args == {"command": "pwd"}
    assert captured["tool_choice"] == "required"
    assert captured["tools"][0]["function"]["name"] == "shell"


def test_native_structured_output_list_schemas_emit_provider_safe_function_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "agent_done",
                                    "arguments": '{"reply":"done"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(
        model="openai/gpt-4.1-mini",
        api_key="sk-test",
        structured_output="native",
        native_tool_choice="required",
        native_probe=False,
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Finish"}],
        action_schema=EXECUTION_SURFACE_SCHEMA,
    )

    assert result.action is not None
    assert result.action.tool == "agent_done"
    tool_defs = captured.get("tools", [])
    function_names = [
        item.get("function", {}).get("name")
        for item in tool_defs
        if isinstance(item, dict)
    ]
    safe_name_pattern = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
    assert all(
        isinstance(name, str) and safe_name_pattern.fullmatch(name) and "." not in name
        for name in function_names
    )


def test_native_structured_output_can_use_string_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "submit_tool_call",
                                    "arguments": (
                                        '{"tool":"shell.run",'
                                        '"args":{"command":"pwd"}}'
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(
        model="openai/gpt-4.1-mini",
        api_key="sk-test",
        structured_output="native",
        native_tool_choice="required",
        native_probe=False,
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Use a tool"}],
        action_schema=ACTION_SCHEMA,
    )

    assert result.action is not None
    assert captured["tool_choice"] == "required"


def test_native_probe_falls_back_from_forced_function_to_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) == 1:
            raise _FakeClientError("bad request", status_code=400)
        if len(calls) == 2:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "submit_tool_call",
                                        "arguments": (
                                            '{"tool":"shell.run",'
                                            '"args":{"command":"echo probe"}}'
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "submit_tool_call",
                                    "arguments": (
                                        '{"tool":"shell.run",'
                                        '"args":{"command":"pwd"}}'
                                    ),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(
        model="lm_studio/local-model",
        api_key="",
        structured_output="native",
        native_tool_choice="forced_function",
        native_fallback_to_prompt=False,
        native_probe=True,
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Use a tool"}],
        action_schema=ACTION_SCHEMA,
    )

    assert result.action is not None
    assert calls[0]["tool_choice"]["function"]["name"] == "submit_tool_call"
    assert calls[1]["tool_choice"] == "required"
    assert calls[2]["tool_choice"] == "required"


def test_native_probe_can_fallback_to_prompt_when_native_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if len(calls) <= 2:
            raise _FakeClientError("bad request", status_code=400)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tool":"http-request.request","args":{"method":"GET"},'
                            '"reason":"Need data"}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(
        model="lm_studio/local-model",
        api_key="",
        structured_output="native",
        native_tool_choice="forced_function",
        native_fallback_to_prompt=True,
        native_probe=True,
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Use a tool"}],
        action_schema=ACTION_SCHEMA,
    )

    assert result.action is not None
    assert result.action.tool == "http-request.request"
    assert "tool_choice" in calls[0]
    assert "tool_choice" in calls[1]
    assert "tool_choice" not in calls[2]
    assert calls[2]["messages"][0]["role"] == "system"
    assert "return ONLY a JSON object matching this schema" in calls[2]["messages"][0]["content"]


def test_native_probe_falls_back_to_prompt_when_probe_returns_no_native_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if "tool_choice" in kwargs:
            return {"choices": [{"message": {"content": "", "tool_calls": []}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tool":"http-request.request","args":{"method":"GET"},'
                            '"reason":"Need data"}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(
        model="local/model",
        api_key="",
        structured_output="native",
        native_tool_choice="required",
        native_fallback_to_prompt=True,
        native_probe=True,
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Use a tool"}],
        action_schema=ACTION_SCHEMA,
    )

    assert result.action is not None
    assert "tool_choice" in calls[0]
    assert "tool_choice" not in calls[1]


def test_prompt_structured_output_parses_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"tool":"http-request.request","args":{"method":"GET"},'
                            '"reason":"Need data"}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(
        model="ollama/llama3.1",
        api_key="",
        structured_output="prompt",
        provider_settings={"api_base": "http://127.0.0.1:11434"},
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Fetch data"}],
        action_schema=ACTION_SCHEMA,
    )

    assert result.action is not None
    assert result.action.tool == "http-request.request"
    assert result.action.args == {"method": "GET"}
    assert result.action.reason == "Need data"
    assert captured["messages"][0]["role"] == "system"
    assert "return ONLY a JSON object matching this schema" in captured["messages"][0]["content"]
    assert "tools" not in captured


def test_prompt_structured_output_rejects_legacy_skill_action_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_completion(**_: Any) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"skill":"http-request","action":"request",'
                            '"args":{"method":"GET"},"reason":"Need data"}'
                        )
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    client = LLMClient(
        model="ollama/llama3.1",
        api_key="",
        structured_output="prompt",
    )

    result = client.complete(
        messages=[{"role": "user", "content": "Fetch data"}],
        action_schema=ACTION_SCHEMA,
    )

    assert result.action is None


def test_count_tokens_uses_litellm_token_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_counter(*, model: str, messages: list[dict[str, Any]]) -> int:
        captured["model"] = model
        captured["messages"] = messages
        return 42

    monkeypatch.setattr("agent.llm.litellm.token_counter", fake_counter)
    client = LLMClient(model="openai/gpt-4.1-mini", api_key="sk-test")

    count = client.count_tokens(messages=[{"role": "user", "content": "x"}])

    assert count == 42
    assert captured["model"] == "openai/gpt-4.1-mini"


def test_count_tokens_falls_back_to_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_counter(*_: Any, **__: Any) -> int:
        raise RuntimeError("unsupported provider")

    monkeypatch.setattr("agent.llm.litellm.token_counter", fake_counter)
    client = LLMClient(model="local/model", api_key="")

    count = client.count_tokens(messages=[{"role": "user", "content": "abcdefgh"}])

    assert count == 2


def test_from_config_accepts_provider_settings() -> None:
    client = LLMClient.from_config(
        {
            "model": "openrouter/deepseek/deepseek-r1",
            "api_key": "sk-test",
            "max_tokens": 2048,
            "temperature": 0.1,
            "timeout_seconds": 30,
            "max_retries": 4,
            "structured_output": "prompt",
            "native_tool_choice": "auto",
            "native_fallback_to_prompt": False,
            "native_probe": False,
            "provider_settings": {"api_base": "https://openrouter.ai/api/v1"},
        }
    )

    assert client.model == "openrouter/deepseek/deepseek-r1"
    assert client.provider_settings["api_base"] == "https://openrouter.ai/api/v1"
    assert client.structured_output == "prompt"
    assert client.native_tool_choice == "auto"
    assert client.native_fallback_to_prompt is False
    assert client.native_probe is False


def test_from_config_maps_api_base_into_provider_settings() -> None:
    client = LLMClient.from_config(
        {
            "model": "lm_studio/local-model",
            "api_key": "",
            "max_tokens": 1024,
            "temperature": 0.1,
            "api_base": "http://127.0.0.1:11434/v1",
        }
    )

    assert client.provider_settings["api_base"] == "http://127.0.0.1:11434/v1"


def test_from_config_rejects_invalid_api_base() -> None:
    with pytest.raises(ValueError, match=r"llm\.api_base"):
        LLMClient.from_config(
            {
                "model": "lm_studio/local-model",
                "api_key": "",
                "max_tokens": 1024,
                "temperature": 0.1,
                "api_base": "",
            }
        )


def test_provider_settings_reject_reserved_keys() -> None:
    with pytest.raises(ValueError, match="reserved keys"):
        LLMClient(model="x", api_key="y", provider_settings={"model": "override"})


def test_structured_output_mode_must_be_valid() -> None:
    with pytest.raises(ValueError, match="structured_output"):
        LLMClient(model="x", api_key="y", structured_output="invalid")


def test_native_tool_choice_must_be_valid() -> None:
    with pytest.raises(ValueError, match="native_tool_choice"):
        LLMClient(model="x", api_key="y", native_tool_choice="bad-value")
