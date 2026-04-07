"""Tests for LiteLLM client wrapper."""

from __future__ import annotations

from typing import Any

import pytest

from agent.llm import LLMClient

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "skill": {"type": "string"},
        "action": {"type": "string"},
        "args": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["skill", "action", "args"],
}


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
                                        '{"skill":"shell","action":"run","args":{"command":"pwd"},'
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
    client = LLMClient(model="openai/gpt-4.1-mini", api_key="sk-test", structured_output="native")

    result = client.complete(
        messages=[{"role": "user", "content": "Use a tool"}],
        action_schema=ACTION_SCHEMA,
    )

    assert result.action is not None
    assert result.action.skill == "shell"
    assert result.action.action == "run"
    assert result.action.args == {"command": "pwd"}
    assert result.action.reason == "Need cwd"
    assert "tools" in captured
    assert captured["tool_choice"]["function"]["name"] == "submit_tool_call"


def test_prompt_structured_output_parses_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"skill":"http-request","action":"request","args":{"method":"GET"},'
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
    assert result.action.skill == "http-request"
    assert result.action.action == "request"
    assert result.action.args == {"method": "GET"}
    assert result.action.reason == "Need data"
    assert captured["messages"][0]["role"] == "system"
    assert "return ONLY a JSON object matching this schema" in captured["messages"][0]["content"]
    assert "tools" not in captured


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
            "provider_settings": {"api_base": "https://openrouter.ai/api/v1"},
        }
    )

    assert client.model == "openrouter/deepseek/deepseek-r1"
    assert client.provider_settings["api_base"] == "https://openrouter.ai/api/v1"
    assert client.structured_output == "prompt"


def test_provider_settings_reject_reserved_keys() -> None:
    with pytest.raises(ValueError, match="reserved keys"):
        LLMClient(model="x", api_key="y", provider_settings={"model": "override"})


def test_structured_output_mode_must_be_valid() -> None:
    with pytest.raises(ValueError, match="structured_output"):
        LLMClient(model="x", api_key="y", structured_output="invalid")
