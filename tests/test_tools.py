"""Unit tests for built-in tools module."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.broker_client import HostServiceError
from agent.llm_types import ToolCall
from agent.tools import SHELL_ENV_ALLOWLIST, Tools


class _RecordingBroker:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, service: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((service, payload))
        return dict(self.result)


class _RaisingBroker:
    def __init__(self, message: str) -> None:
        self.message = message

    def call(self, service: str, payload: dict[str, Any]) -> dict[str, Any]:
        del service
        del payload
        raise HostServiceError(self.message)


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


def test_tools_shell_does_not_leak_unallowlisted_parent_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "sk-do-not-leak")
    tools = Tools(config={})

    result = tools.execute(
        ToolCall(tool="shell", args={"command": 'echo "seen=[${FAKE_API_KEY-unset}]"'})
    )

    assert result.exit_code == 0
    assert "sk-do-not-leak" not in result.stdout
    assert result.stdout.strip() == "seen=[unset]"


def test_tools_shell_runs_real_command_under_stripped_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FAKE_API_KEY", "sk-do-not-leak")
    tools = Tools(config={})

    # `cat` is resolved through PATH, so this fails if the allowlist strips the
    # environment down to something bash cannot work in.
    result = tools.execute(ToolCall(tool="shell", args={"command": "echo ready | cat"}))

    assert result.exit_code == 0
    assert result.stdout.strip() == "ready"
    assert result.stderr == ""


def test_tools_shell_passes_allowlisted_parent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TZ", "UTC")
    tools = Tools(config={})

    result = tools.execute(ToolCall(tool="shell", args={"command": 'echo "tz=${TZ-unset}"'}))

    assert result.exit_code == 0
    assert result.stdout.strip() == "tz=UTC"


def test_tools_shell_forwards_configured_env_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC_EXTRA_PASSTHROUGH", "forwarded")
    tools = Tools(config={"shell": {"env_passthrough": ["SC_EXTRA_PASSTHROUGH"]}})

    result = tools.execute(
        ToolCall(tool="shell", args={"command": 'echo "extra=${SC_EXTRA_PASSTHROUGH-unset}"'})
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "extra=forwarded"


def test_shell_env_omits_allowlisted_variable_absent_from_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("FAKE_API_KEY", "sk-do-not-leak")
    tools = Tools(config={})

    env = tools._shell_env()

    # Absent means unset in the child, not set to an empty string.
    assert "TZ" not in env
    assert env["PATH"] == "/usr/bin"
    assert "FAKE_API_KEY" not in env
    assert set(env) <= set(SHELL_ENV_ALLOWLIST)


def test_shell_env_ignores_malformed_passthrough_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SC_EXTRA_PASSTHROUGH", "forwarded")
    tools = Tools(config={"shell": {"env_passthrough": "SC_EXTRA_PASSTHROUGH"}})

    assert "SC_EXTRA_PASSTHROUGH" not in tools._shell_env()


def test_tools_web_search_calls_broker_with_expected_payload() -> None:
    broker = _RecordingBroker(
        {"success": True, "results": [{"title": "A", "url": "u", "snippet": "s"}]}
    )
    tools = Tools(config={"web_search": {"max_results": 7}}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(ToolCall(tool="web_search", args={"query": "llm"}))

    assert result.exit_code == 0
    assert result.stderr == ""
    assert broker.calls == [
        (
            "broker",
            {
                "action": "web_search",
                "query": "llm",
                "max_results": 7,
            },
        )
    ]
    payload = _unwrap_data(result.stdout)
    assert payload["results"] == [{"title": "A", "url": "u", "snippet": "s"}]


def test_tools_web_search_broker_denial_returns_wrapped_error_payload() -> None:
    broker = _RecordingBroker({"success": False, "error": "policy_denied", "reason": "nope"})
    tools = Tools(config={}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(ToolCall(tool="web_search", args={"query": "llm"}))

    assert result.exit_code == 1
    assert result.stderr == ""
    payload = _unwrap_data(result.stdout)
    assert payload["error"] == "policy_denied"


def test_tools_web_search_handles_host_service_error() -> None:
    tools = Tools(config={}, broker=_RaisingBroker("broker down"))  # type: ignore[arg-type]

    result = tools.execute(ToolCall(tool="web_search", args={"query": "llm"}))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "broker down"


def test_tools_web_search_rejects_missing_success_envelope() -> None:
    broker = _RecordingBroker({"results": [{"title": "A", "url": "u", "snippet": "s"}]})
    tools = Tools(config={}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(ToolCall(tool="web_search", args={"query": "llm"}))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "invalid broker response for web_search: missing success envelope."


def test_tools_web_fetch_calls_broker_with_expected_payload() -> None:
    broker = _RecordingBroker(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "text/plain"},
            "body": "hello",
            "truncated": False,
        }
    )
    tools = Tools(config={}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 0
    assert broker.calls == [
        ("broker", {"action": "web_fetch", "url": "https://example.com"})
    ]
    payload = _unwrap_data(result.stdout)
    assert payload["body"] == "hello"


def test_tools_web_fetch_handles_host_service_error() -> None:
    tools = Tools(config={}, broker=_RaisingBroker("timeout"))  # type: ignore[arg-type]

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "timeout"


def test_tools_web_fetch_rejects_missing_success_envelope() -> None:
    broker = _RecordingBroker({"status_code": 200, "body": "hello"})
    tools = Tools(config={}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "invalid broker response for web_fetch: missing success envelope."


def test_tools_http_request_schema_includes_integration_description() -> None:
    tools = Tools(config={})

    schema = next(item for item in tools.schema() if item["name"] == "http_request")

    integration = schema["parameters"]["properties"]["integration"]
    assert integration["type"] == ["string", "null"]
    assert "Named integration from secrets.yaml" in integration["description"]


def test_tools_http_request_calls_broker_with_expected_payload() -> None:
    broker = _RecordingBroker(
        {
            "success": True,
            "status_code": 201,
            "headers": {"Content-Type": "application/json"},
            "body": '{"ok":true}',
            "truncated": False,
        }
    )
    tools = Tools(config={}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={
                "method": "post",
                "url": "https://api.example.com/items",
                "integration": "github",
                "headers": {"Accept": "application/json"},
                "body": '{"name":"x"}',
            },
        )
    )

    assert result.exit_code == 0
    assert broker.calls == [
        (
            "broker",
            {
                "action": "http_request",
                "integration": "github",
                "method": "POST",
                "url": "https://api.example.com/items",
                "headers": {"Accept": "application/json"},
                "body": '{"name":"x"}',
            },
        )
    ]
    payload = _unwrap_data(result.stdout)
    assert payload["status_code"] == 201
    assert payload["body"] == '{"ok":true}'
    assert _wrapper_count(result.stdout) == 1


def test_tools_http_request_broker_denial_returns_wrapped_payload() -> None:
    broker = _RecordingBroker(
        {
            "success": False,
            "error": "policy_denied",
            "reason": "path not allowed",
            "requested_url": "https://api.example.com/admin",
        }
    )
    tools = Tools(config={}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={"method": "GET", "url": "https://api.example.com/admin"},
        )
    )

    assert result.exit_code == 1
    assert result.stderr == ""
    payload = _unwrap_data(result.stdout)
    assert payload["error"] == "policy_denied"
    assert _wrapper_count(result.stdout) == 1


def test_tools_web_search_output_uses_single_data_wrapper() -> None:
    broker = _RecordingBroker({"success": True, "results": []})
    tools = Tools(config={"web_search": {"max_results": 10}}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(ToolCall(tool="web_search", args={"query": "test"}))

    assert result.exit_code == 0
    assert _wrapper_count(result.stdout) == 1


def test_tools_web_fetch_output_uses_single_data_wrapper() -> None:
    broker = _RecordingBroker(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "text/plain"},
            "body": "hello",
            "truncated": False,
        }
    )
    tools = Tools(config={}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 0
    assert _wrapper_count(result.stdout) == 1


def test_tools_http_request_body_preserves_raw_broker_value() -> None:
    broker = _RecordingBroker(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "body": "raw-upstream-body",
            "truncated": False,
        }
    )
    tools = Tools(config={}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={"method": "GET", "url": "https://api.example.com"},
        )
    )

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["body"] == "raw-upstream-body"
    assert isinstance(payload["body"], str)
    assert not payload["body"].startswith("--- BEGIN DATA ---")
    assert _wrapper_count(result.stdout) == 1


def test_tools_http_request_invalid_method_rejected() -> None:
    tools = Tools(config={})

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={"method": "OPTIONS", "url": "https://api.example.com"},
        )
    )

    assert result.exit_code == 1
    assert "http_request.method must be one of" in result.stderr


def test_tools_http_request_handles_host_service_error() -> None:
    tools = Tools(config={}, broker=_RaisingBroker("offline"))  # type: ignore[arg-type]

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={"method": "GET", "url": "https://api.example.com"},
        )
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "offline"


def test_tools_http_request_rejects_missing_success_envelope() -> None:
    broker = _RecordingBroker({"status_code": 200, "body": "ok", "headers": {}, "truncated": False})
    tools = Tools(config={}, broker=broker)  # type: ignore[arg-type]

    result = tools.execute(
        ToolCall(tool="http_request", args={"method": "GET", "url": "https://api.example.com"})
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert (
        result.stderr
        == "invalid broker response for http_request: missing success envelope."
    )


def test_tools_does_not_emit_legacy_web_search_key_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING")

    _ = Tools(config={"web_search": {"api_key": ""}})

    assert "web_search.api_key" not in caplog.text


def _unwrap_data(text: str) -> dict[str, Any]:
    prefix = "--- BEGIN DATA ---\n"
    suffix = "\n--- END DATA ---"
    assert text.startswith(prefix)
    assert text.endswith(suffix)
    payload = text[len(prefix) : -len(suffix)]
    loaded = json.loads(payload)
    assert isinstance(loaded, dict)
    return loaded


def _wrapper_count(text: str) -> int:
    return text.count("--- BEGIN DATA ---")
