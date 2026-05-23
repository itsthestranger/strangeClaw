"""Tests for guest-side LLM proxy runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from agent.broker_client import BrokerClient, HostServiceError
from agent.llm_proxy import LLMProxyRuntime
from agent.llm_types import LLMRuntimeError


class _FakeClient:
    def __init__(
        self,
        handler: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._handler = handler

    def call(self, service: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((service, payload))
        return self._handler(service, payload)


def test_llm_proxy_complete_sends_payload_and_deserializes_response() -> None:
    def handler(service: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert service == "llm"
        assert payload == {
            "action": "complete",
            "messages": [{"role": "user", "content": "hi"}],
            "action_schema": [{"name": "agent_done"}],
        }
        return {
            "text": "",
            "action": {"tool": "agent_done", "args": {"reply": "ok"}, "reason": "done"},
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }

    fake = _FakeClient(handler)
    runtime = LLMProxyRuntime(cast(BrokerClient, fake))

    response = runtime.complete(
        [{"role": "user", "content": "hi"}],
        action_schema=[{"name": "agent_done"}],
    )

    assert response.action is not None
    assert response.action.tool == "agent_done"
    assert response.action.args == {"reply": "ok"}
    assert response.action.reason == "done"
    assert response.usage == {"input_tokens": 1, "output_tokens": 2}
    assert fake.calls


def test_llm_proxy_count_tokens_sends_payload_and_returns_int() -> None:
    def handler(service: str, payload: dict[str, Any]) -> dict[str, Any]:
        assert service == "llm"
        assert payload == {
            "action": "count_tokens",
            "messages": [{"role": "user", "content": "hi"}],
        }
        return {"tokens": 42}

    runtime = LLMProxyRuntime(cast(BrokerClient, _FakeClient(handler)))

    assert runtime.count_tokens([{"role": "user", "content": "hi"}]) == 42


def test_llm_proxy_reraises_host_service_error_as_runtime_error() -> None:
    class FailingClient:
        def call(self, service: str, payload: dict[str, Any]) -> dict[str, Any]:
            del service
            del payload
            raise HostServiceError("sentinel failure")

    runtime = LLMProxyRuntime(cast(BrokerClient, FailingClient()))

    with pytest.raises(LLMRuntimeError, match="sentinel failure"):
        runtime.complete([])


def test_llm_proxy_service_failure_payload_raises_runtime_error() -> None:
    runtime = LLMProxyRuntime(
        cast(
            BrokerClient,
            _FakeClient(lambda _service, _payload: {"success": False, "error": "too large"}),
        )
    )

    with pytest.raises(LLMRuntimeError, match="too large"):
        runtime.complete([])


def test_llm_proxy_rejects_malformed_complete_action() -> None:
    runtime = LLMProxyRuntime(
        cast(BrokerClient, _FakeClient(lambda _service, _payload: {"action": {"tool": "x"}}))
    )

    with pytest.raises(LLMRuntimeError, match="tool and args"):
        runtime.complete([])
