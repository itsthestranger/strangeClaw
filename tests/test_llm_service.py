"""Tests for the host-side LLM service."""

from __future__ import annotations

from typing import Any

import pytest

from agent.broker_client import BrokerClient
from agent.llm_types import LLMResponse, ToolCall
from agent.transport import DEFAULT_MAX_FRAME_BYTES
from sandbox.host_services import HostServiceServer
from sandbox.llm_service import DEFAULT_LLM_MAX_REQUEST_BYTES, LLMService


class FakeLLMClient:
    def __init__(self) -> None:
        self.complete_calls: list[tuple[list[dict[str, Any]], Any]] = []
        self.count_token_calls: list[list[dict[str, Any]]] = []
        self.fail_with: Exception | None = None
        self.action_error: str | None = None

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.complete_calls.append((messages, action_schema))
        if self.fail_with is not None:
            raise self.fail_with
        if self.action_error is not None:
            return LLMResponse(text="", action=None, usage=None, action_error=self.action_error)
        return LLMResponse(
            text="use shell",
            action=ToolCall(tool="shell", args={"command": "pwd"}, reason="inspect"),
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        self.count_token_calls.append(messages)
        if self.fail_with is not None:
            raise self.fail_with
        return 42


def _config(api_key: str = "sk-test-key", max_request_bytes: int = 4096) -> dict[str, Any]:
    return {
        "llm": {
            "model": "anthropic/claude-sonnet-4-20250514",
            "api_key": api_key,
            "api_base": "http://127.0.0.1:1234/v1" if not api_key else None,
        },
        "host_services": {"llm_max_request_bytes": max_request_bytes},
    }


def test_llm_service_complete_returns_normalized_shape() -> None:
    fake = FakeLLMClient()
    service = LLMService(_config(), llm_client=fake)
    messages = [{"role": "user", "content": "hello"}]
    action_schema = [{"name": "shell", "parameters": {"type": "object"}}]

    result = service.handle(
        {"action": "complete", "messages": messages, "action_schema": action_schema}
    )

    assert result == {
        "success": True,
        "text": "use shell",
        "action": {"tool": "shell", "args": {"command": "pwd"}, "reason": "inspect"},
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        "action_error": None,
    }
    assert fake.complete_calls == [(messages, action_schema)]


def test_llm_service_complete_forwards_decision_rejection_reason() -> None:
    fake = FakeLLMClient()
    fake.action_error = "You emitted 2 tool calls. Emit exactly one structured decision per turn."
    service = LLMService(_config(), llm_client=fake)

    result = service.handle(
        {"action": "complete", "messages": [{"role": "user", "content": "hi"}]}
    )

    assert result["action_error"] == (
        "You emitted 2 tool calls. Emit exactly one structured decision per turn."
    )


def test_llm_service_redacts_api_key_in_decision_rejection_reason() -> None:
    fake = FakeLLMClient()
    fake.action_error = 'Could not parse arguments: {"token":"sk-test-key"}'
    service = LLMService(_config(api_key="sk-test-key"), llm_client=fake)

    result = service.handle(
        {"action": "complete", "messages": [{"role": "user", "content": "hi"}]}
    )

    assert result["action_error"] == 'Could not parse arguments: {"token":"[REDACTED]"}'
    assert "sk-test-key" not in str(result)


def test_llm_service_count_tokens_returns_token_count() -> None:
    fake = FakeLLMClient()
    service = LLMService(_config(), llm_client=fake)
    messages = [{"role": "user", "content": "hello"}]

    result = service.handle({"action": "count_tokens", "messages": messages})

    assert result == {"success": True, "tokens": 42}
    assert fake.count_token_calls == [messages]


def test_llm_service_rejects_oversized_payload_before_client_call() -> None:
    fake = FakeLLMClient()
    service = LLMService(_config(max_request_bytes=90), llm_client=fake)

    result = service.handle(
        {
            "action": "complete",
            "messages": [{"role": "user", "content": "x" * 200}],
            "action_schema": None,
        }
    )

    assert result["success"] is False
    assert "too large" in str(result["error"])
    assert fake.complete_calls == []
    assert fake.count_token_calls == []


def test_llm_service_redacts_api_key_from_provider_errors_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    key = "sk-secret-sentinel"
    fake = FakeLLMClient()
    fake.fail_with = RuntimeError(f"provider rejected {key}")
    service = LLMService(_config(api_key=key), llm_client=fake)

    with caplog.at_level("WARNING"):
        result = service.handle(
            {
                "action": "complete",
                "messages": [{"role": "user", "content": "hello"}],
                "action_schema": None,
            }
        )

    assert result["success"] is False
    assert key not in str(result)
    assert "[REDACTED]" in str(result["error"])
    assert key not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_llm_service_forwards_messages_and_schema_without_mutation() -> None:
    fake = FakeLLMClient()
    service = LLMService(_config(), llm_client=fake)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    action_schema = [{"name": "agent_done", "parameters": {"type": "object"}}]

    service.handle({"action": "complete", "messages": messages, "action_schema": action_schema})

    forwarded_messages, forwarded_schema = fake.complete_calls[0]
    assert forwarded_messages is messages
    assert forwarded_schema is action_schema
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    assert action_schema == [{"name": "agent_done", "parameters": {"type": "object"}}]


def test_llm_service_handles_local_llm_config_with_empty_api_key() -> None:
    fake = FakeLLMClient()
    service = LLMService(_config(api_key=""), llm_client=fake)

    result = service.handle(
        {
            "action": "complete",
            "messages": [{"role": "user", "content": "hello"}],
            "action_schema": None,
        }
    )

    assert result["success"] is True
    assert fake.complete_calls


def test_llm_service_registers_with_host_service_server() -> None:
    fake = FakeLLMClient()
    server = HostServiceServer()
    server.register("llm", LLMService(_config(), llm_client=fake).handle)
    client = BrokerClient(server)

    result = client.call("llm", {"action": "count_tokens", "messages": []})

    assert result == {"success": True, "tokens": 42}


def test_frame_cap_exceeds_llm_request_cap() -> None:
    """The transport frame cap is a resource guard that must clear the policy cap.

    ``DEFAULT_LLM_MAX_REQUEST_BYTES`` is checked only after a frame has been
    buffered and parsed, so it cannot bound buffering. ``DEFAULT_MAX_FRAME_BYTES``
    does that, and must leave room for the largest payload the policy allows plus
    its JSON event envelope.
    """
    assert DEFAULT_MAX_FRAME_BYTES > DEFAULT_LLM_MAX_REQUEST_BYTES
