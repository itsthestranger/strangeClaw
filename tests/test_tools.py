"""Unit tests for built-in tools module."""

from __future__ import annotations

import json
from typing import Any, cast

from agent.broker_client import HostServiceError
from agent.llm import ToolCall
from agent.tools import Tools


class _FakeBroker:
    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self.response = response or {"success": True}
        self.calls: list[dict[str, Any]] = []
        self.error: HostServiceError | None = None

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


def _tools_with_broker(
    *,
    config: dict[str, Any] | None = None,
    broker_response: dict[str, Any] | None = None,
) -> tuple[Tools, _FakeBroker]:
    broker = _FakeBroker(response=broker_response)
    tools = Tools(config=config or {}, broker=cast(Any, broker))
    return tools, broker


def test_tools_shell_execute_success() -> None:
    tools = Tools(config={})

    result = tools.execute(ToolCall(tool="shell", args={"command": "printf 'hello'"}))

    assert result.exit_code == 0
    assert result.stdout == "hello"
    assert result.stderr == ""


def test_tools_schema_excludes_disabled_shell() -> None:
    tools = Tools(config={"tools": {"shell": False}})

    schema = tools.schema()

    assert all(item["name"] != "shell" for item in schema)
    assert "shell" not in tools.list_enabled()


def test_tools_execute_rejects_disabled_tool() -> None:
    tools = Tools(config={"tools": {"shell": False}})

    result = tools.execute(ToolCall(tool="shell", args={"command": "echo should-not-run"}))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "tool shell is not enabled."


def test_tools_shell_output_truncates_long_text() -> None:
    tools = Tools(config={})
    command = "python - <<'PY'\nprint('a' * 9005)\nPY"

    result = tools.execute(ToolCall(tool="shell", args={"command": command}))

    assert result.exit_code == 0
    assert "...[truncated" in result.stdout
    assert len(result.stdout) < 9005


def test_tools_web_search_calls_broker_with_config_default_max_results() -> None:
    tools, broker = _tools_with_broker(
        config={"web_search": {"max_results": 7}},
        broker_response={
            "success": True,
            "query": "llm",
            "results": [{"title": "T1", "url": "https://example.com/1", "snippet": "S1"}],
        },
    )

    result = tools.execute(ToolCall(tool="web_search", args={"query": "llm"}))

    assert result.exit_code == 0
    assert broker.calls == [{"action": "web_search", "query": "llm", "max_results": 7}]
    assert _unwrap_data(result.stdout)["results"] == [
        {"title": "T1", "url": "https://example.com/1", "snippet": "S1"}
    ]


def test_tools_web_search_args_override_max_results() -> None:
    tools, broker = _tools_with_broker(config={"web_search": {"max_results": 7}})

    result = tools.execute(
        ToolCall(tool="web_search", args={"query": "llm", "max_results": 2})
    )

    assert result.exit_code == 0
    assert broker.calls == [{"action": "web_search", "query": "llm", "max_results": 2}]


def test_tools_web_search_requires_broker() -> None:
    tools = Tools(config={})

    result = tools.execute(ToolCall(tool="web_search", args={"query": "llm"}))

    assert result.exit_code == 1
    assert result.stderr == "web_search requires a host broker connection."


def test_tools_web_search_rejects_invalid_max_results() -> None:
    tools, _ = _tools_with_broker()

    result = tools.execute(
        ToolCall(tool="web_search", args={"query": "llm", "max_results": 0})
    )

    assert result.exit_code == 1
    assert result.stderr == "web_search.max_results must be a positive integer."


def test_tools_web_fetch_calls_broker() -> None:
    tools, broker = _tools_with_broker(
        broker_response={
            "success": True,
            "url": "https://example.com",
            "text": "Hello",
        }
    )

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 0
    assert broker.calls == [{"action": "web_fetch", "url": "https://example.com"}]
    assert _unwrap_data(result.stdout)["success"] is True


def test_tools_web_fetch_requires_broker() -> None:
    tools = Tools(config={})

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 1
    assert result.stderr == "web_fetch requires a host broker connection."


def test_tools_http_request_calls_broker_with_payload() -> None:
    tools, broker = _tools_with_broker(
        broker_response={"success": True, "status_code": 201, "body": '{"ok":true}'}
    )

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={
                "method": "post",
                "url": "https://api.example.com/items",
                "integration": "notion",
                "headers": {"Accept": "application/json"},
                "body": '{"name":"x"}',
            },
        )
    )

    assert result.exit_code == 0
    assert broker.calls == [
        {
            "action": "http_request",
            "integration": "notion",
            "method": "POST",
            "url": "https://api.example.com/items",
            "headers": {"Accept": "application/json"},
            "body": '{"name":"x"}',
        }
    ]
    assert _unwrap_data(result.stdout)["status_code"] == 201


def test_tools_http_request_requires_broker() -> None:
    tools = Tools(config={})

    result = tools.execute(
        ToolCall(tool="http_request", args={"method": "GET", "url": "https://api.example.com"})
    )

    assert result.exit_code == 1
    assert result.stderr == "http_request requires a host broker connection."


def test_tools_http_request_invalid_method_rejected() -> None:
    tools, _ = _tools_with_broker()

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={"method": "OPTIONS", "url": "https://api.example.com"},
        )
    )

    assert result.exit_code == 1
    assert "http_request.method must be one of" in result.stderr


def test_tools_http_request_invalid_headers_rejected() -> None:
    tools, _ = _tools_with_broker()

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={
                "method": "GET",
                "url": "https://api.example.com",
                "headers": {"X-Test": 1},
            },
        )
    )

    assert result.exit_code == 1
    assert result.stderr == "http_request.headers must contain only string keys and values."


def test_tools_http_request_schema_integration_description_mentions_secrets_yaml() -> None:
    tools, _ = _tools_with_broker()

    schema = next(item for item in tools.schema() if item["name"] == "http_request")
    integration = schema["parameters"]["properties"]["integration"]
    assert "secrets.yaml" in integration["description"]


def test_tools_broker_denial_returns_structured_payload() -> None:
    tools, _ = _tools_with_broker(
        broker_response={
            "success": False,
            "error": "policy_denied",
            "reason": "method POST is not allowed",
        }
    )

    result = tools.execute(
        ToolCall(tool="http_request", args={"method": "POST", "url": "https://api.example.com"})
    )

    assert result.exit_code == 1
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is False
    assert payload["error"] == "policy_denied"
    assert payload["reason"] == "method POST is not allowed"
    assert result.stderr == ""


def test_tools_broker_transport_error_is_stderr() -> None:
    tools, broker = _tools_with_broker()
    broker.error = HostServiceError("transport down")

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "transport down"


def _unwrap_data(text: str) -> dict[str, Any]:
    prefix = "--- BEGIN DATA ---\n"
    suffix = "\n--- END DATA ---"
    assert text.startswith(prefix)
    assert text.endswith(suffix)
    payload = text[len(prefix) : -len(suffix)]
    loaded = json.loads(payload)
    assert isinstance(loaded, dict)
    return loaded
