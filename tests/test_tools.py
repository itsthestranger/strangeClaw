"""Unit tests for built-in tools module."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import requests

from agent.llm import ToolCall
from agent.request_broker_client import InProcessRequestBrokerClient
from agent.tools import Tools
from broker.credentials import HostCredential, HostCredentialRegistry
from broker.request_broker import RequestBroker


class _FakeBrokerHTTPResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text


class _FakeMetadata:
    def __init__(self, title: str | None) -> None:
        self.title = title


class _FakeTrafilatura:
    @staticmethod
    def extract(html: str, include_links: bool, include_tables: bool) -> str:
        assert include_links is True
        assert include_tables is True
        assert "<html>" in html
        return "main extracted text"

    @staticmethod
    def extract_metadata(html: str) -> _FakeMetadata:
        assert "<title>Example</title>" in html
        return _FakeMetadata("Example")


class _FakeBrokerClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = dict(response)
        self.requests: list[dict[str, Any]] = []

    def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.requests.append(dict(request))
        return dict(self.response)


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


def test_tools_web_search_normalizes_brave(monkeypatch: Any) -> None:
    del monkeypatch
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "web": {
                        "results": [
                            {
                                "title": "Result One",
                                "url": "https://example.com/1",
                                "description": "Snippet 1",
                            },
                            {
                                "title": "Result Two",
                                "url": "https://example.com/2",
                                "description": "Snippet 2",
                            },
                        ]
                    }
                }
            ),
            "truncated": False,
            "integration": "brave_search",
        }
    )
    tools = Tools(
        config={
            "web_search": {
                "endpoint": "https://api.search.brave.com/res/v1/web/search",
                "format": "brave",
                "integration": "brave_search",
                "max_results": 1,
            },
        },
        request_broker_client=broker,
    )

    result = tools.execute(ToolCall(tool="web_search", args={"query": "llm"}))

    assert result.exit_code == 0
    assert result.stderr == ""
    assert broker.requests == [
        {
            "method": "GET",
            "url": "https://api.search.brave.com/res/v1/web/search?q=llm",
            "integration": "brave_search",
            "headers": {
                "User-Agent": "strangeclaw/0.1 (+local)",
                "Accept": "application/json",
            },
            "body": None,
        }
    ]

    body = _unwrap_data(result.stdout)
    assert body["query"] == "llm"
    assert body["results"] == [
        {"title": "Result One", "url": "https://example.com/1", "snippet": "Snippet 1"}
    ]


def test_tools_web_search_normalizes_searxng(monkeypatch: Any) -> None:
    del monkeypatch
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "results": [
                        {"title": "S1", "url": "https://s1", "content": "C1"},
                        {"title": "S2", "url": "https://s2", "content": "C2"},
                    ]
                }
            ),
            "truncated": False,
            "integration": None,
        }
    )
    tools = Tools(
        config={
            "web_search": {
                "endpoint": "http://localhost:8080/search",
                "format": "searxng",
                "max_results": 10,
            },
        },
        request_broker_client=broker,
    )

    result = tools.execute(ToolCall(tool="web_search", args={"query": "solid state batteries"}))

    assert result.exit_code == 0
    assert broker.requests == [
        {
            "method": "GET",
            "url": "http://localhost:8080/search?q=solid+state+batteries&format=json",
            "integration": None,
            "headers": {
                "User-Agent": "strangeclaw/0.1 (+local)",
                "Accept": "application/json",
            },
            "body": None,
        }
    ]
    assert _unwrap_data(result.stdout)["results"] == [
        {"title": "S1", "url": "https://s1", "snippet": "C1"},
        {"title": "S2", "url": "https://s2", "snippet": "C2"},
    ]


def test_tools_web_search_brave_requires_integration() -> None:
    tools = Tools(
        config={
            "web_search": {
                "endpoint": "https://api.search.brave.com/res/v1/web/search",
                "format": "brave",
            }
        }
    )

    result = tools.execute(ToolCall(tool="web_search", args={"query": "test"}))

    assert result.exit_code == 1
    assert "web_search.integration is required" in result.stderr


def test_tools_web_search_does_not_use_direct_requests_path(monkeypatch: Any) -> None:
    def fail_direct_get(*args: Any, **kwargs: Any) -> Any:
        del args
        del kwargs
        raise AssertionError("tools web_search must not call requests.get directly")

    monkeypatch.setattr("requests.get", fail_direct_get)
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"results": []}),
            "truncated": False,
            "integration": None,
        }
    )
    tools = Tools(
        config={"web_search": {"endpoint": "https://example.com/search", "format": "searxng"}},
        request_broker_client=broker,
    )

    result = tools.execute(ToolCall(tool="web_search", args={"query": "q"}))

    assert result.exit_code == 0
    assert len(broker.requests) == 1


def test_tools_web_fetch_html_uses_trafilatura(monkeypatch: Any) -> None:
    html = "<html><head><title>Example</title></head><body><p>Hello</p></body></html>"
    monkeypatch.setattr("agent.tools.trafilatura", _FakeTrafilatura())
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "text/html; charset=utf-8"},
            "body": html,
            "truncated": False,
            "integration": None,
        }
    )
    tools = Tools(config={"web_fetch": {"max_chars": 20000}}, request_broker_client=broker)

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 0
    assert broker.requests == [
        {
            "method": "GET",
            "url": "https://example.com",
            "integration": None,
            "headers": {"User-Agent": "strangeclaw/0.1 (+fetch)"},
            "body": None,
            "response_body_max_chars": 5242880,
        }
    ]
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is True
    assert payload["content_type"] == "text/html"
    assert payload["title"] == "Example"
    assert payload["text"] == "main extracted text"
    assert payload["truncated"] is False


def test_tools_web_fetch_json_returns_body(monkeypatch: Any) -> None:
    del monkeypatch
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "application/json"},
            "body": '{"ok":true}',
            "truncated": False,
            "integration": None,
        }
    )
    tools = Tools(config={}, request_broker_client=broker)

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://api.example.com/data"}))

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["content_type"] == "application/json"
    assert payload["text"] == '{"ok":true}'


def test_tools_web_fetch_pdf_returns_metadata_hint(monkeypatch: Any) -> None:
    del monkeypatch
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "application/pdf"},
            "body": "%PDF-1.4",
            "truncated": False,
            "integration": None,
        }
    )
    tools = Tools(config={}, request_broker_client=broker)

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com/doc.pdf"}))

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["content_type"] == "application/pdf"
    assert "Use shell tool with pdftotext" in str(payload["text"])


def test_tools_web_fetch_truncates_at_max_chars(monkeypatch: Any) -> None:
    del monkeypatch
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "text/plain"},
            "body": "abcdefghijklmnopqrstuvwxyz",
            "truncated": False,
            "integration": None,
        }
    )
    tools = Tools(config={"web_fetch": {"max_chars": 10}}, request_broker_client=broker)

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com/text"}))

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["truncated"] is True
    assert str(payload["text"]).startswith("abcdefghij")
    assert "truncated, original" in str(payload["text"])


def test_tools_web_fetch_handles_request_error(monkeypatch: Any) -> None:
    del monkeypatch
    broker = _FakeBrokerClient(
        {
            "success": False,
            "error_code": "request_failed",
            "message": "HTTP request failed: boom",
            "integration": None,
        }
    )
    tools = Tools(config={}, request_broker_client=broker)

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 1
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is False
    assert payload["error_code"] == "request_failed"
    assert "web_fetch request failed" in str(payload["error"])


def test_tools_web_fetch_applies_5mb_cap(monkeypatch: Any) -> None:
    del monkeypatch
    body = "a" * (5 * 1024 * 1024)
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "text/plain"},
            "body": body,
            "truncated": True,
            "integration": None,
        }
    )
    tools = Tools(config={"web_fetch": {"max_chars": 6000000}}, request_broker_client=broker)

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com/huge"}))

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["truncated"] is True
    assert "response body capped at 5242880 bytes" in str(payload["text"])


def test_tools_web_fetch_rejects_private_target_before_network(
    monkeypatch: Any,
) -> None:
    def fail_request(**kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("network request should be blocked by broker policy")

    monkeypatch.setattr("broker.request_broker.requests.request", fail_request)
    broker_client = InProcessRequestBrokerClient(
        RequestBroker(HostCredentialRegistry(credentials={}))
    )
    tools = Tools(config={}, request_broker_client=broker_client)

    result = tools.execute(
        ToolCall(tool="web_fetch", args={"url": "http://127.0.0.1:8000/private"})
    )

    assert result.exit_code == 1
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is False
    assert payload["error_code"] == "policy_denied"


def test_tools_web_fetch_redirect_policy_is_enforced(
    monkeypatch: Any,
) -> None:
    call_count = 0

    def fake_request(**kwargs: Any) -> _FakeBrokerHTTPResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeBrokerHTTPResponse(
                status_code=302,
                headers={"Location": "http://127.0.0.1/internal"},
                text="",
            )
        raise AssertionError(f"unexpected second request: {kwargs}")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker_client = InProcessRequestBrokerClient(
        RequestBroker(HostCredentialRegistry(credentials={}))
    )
    tools = Tools(config={}, request_broker_client=broker_client)

    result = tools.execute(
        ToolCall(tool="web_fetch", args={"url": "https://example.com/start"})
    )

    assert result.exit_code == 1
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is False
    assert payload["error_code"] == "policy_denied"
    assert "Redirect denied by policy" in payload["error"]


def test_tools_web_fetch_does_not_use_direct_requests_path(monkeypatch: Any) -> None:
    def fail_direct_get(*args: Any, **kwargs: Any) -> Any:
        del args
        del kwargs
        raise AssertionError("tools web_fetch must not call requests.get directly")

    monkeypatch.setattr("requests.get", fail_direct_get)
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {"Content-Type": "text/plain"},
            "body": "ok",
            "truncated": False,
            "integration": None,
        }
    )
    tools = Tools(config={}, request_broker_client=broker)

    result = tools.execute(ToolCall(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 0
    assert len(broker.requests) == 1


def test_tools_http_request_uses_broker_client_and_captures_response() -> None:
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 201,
            "headers": {"Content-Type": "application/json"},
            "body": '{"ok":true}',
            "truncated": False,
            "integration": "notion",
        }
    )
    tools = Tools(config={}, request_broker_client=broker)

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={
                "method": "post",
                "url": "https://api.example.com/items",
                "headers": {"Content-Type": "application/json"},
                "integration": "notion",
                "body": '{"name":"x"}',
            },
        )
    )

    assert result.exit_code == 0
    assert broker.requests == [
        {
            "method": "POST",
            "url": "https://api.example.com/items",
            "integration": "notion",
            "headers": {
                "Content-Type": "application/json",
                "User-Agent": "strangeclaw/0.1 (+http)",
            },
            "body": '{"name":"x"}',
        }
    ]
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is True
    assert payload["status_code"] == 201
    assert payload["body"] == '{"ok":true}'
    assert payload["truncated"] is False
    assert payload["integration"] == "notion"


def test_tools_http_request_schema_uses_request_broker_metadata() -> None:
    tools = Tools(
        config={
            "request_broker": {
                "integration_metadata": [
                    {"name": "notion"},
                    {"name": "github"},
                    {"name": "notion"},
                ]
            }
        },
        request_broker_client=_FakeBrokerClient(
            {
                "success": True,
                "status_code": 200,
                "headers": {},
                "body": "",
                "truncated": False,
                "integration": None,
            }
        ),
    )

    schema = next(item for item in tools.schema() if item["name"] == "http_request")
    assert schema["parameters"]["properties"]["integration"]["enum"] == ["github", "notion", None]


def test_tools_http_request_rejects_protected_header_override_before_network(
    monkeypatch: Any,
) -> None:
    def fail_request(**kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("network request should be blocked by broker policy")

    monkeypatch.setattr("broker.request_broker.requests.request", fail_request)
    registry = HostCredentialRegistry(
        credentials={
            "github": HostCredential(
                name="github",
                credential_type="bearer",
                token="github-secret",
                allowed_hosts=("api.github.com",),
                allowed_methods=("GET",),
                allowed_paths=("/repos/*",),
                default_headers={"Accept": "application/vnd.github+json"},
            )
        }
    )
    broker_client = InProcessRequestBrokerClient(RequestBroker(registry))
    tools = Tools(config={}, request_broker_client=broker_client)

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={
                "method": "GET",
                "url": "https://api.github.com/repos/o/r",
                "integration": "github",
                "headers": {"Authorization": "Bearer attacker"},
            },
        )
    )

    assert result.exit_code == 1
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is False
    assert payload["error_code"] == "policy_denied"
    assert "protected header" in payload["message"]


def test_tools_http_request_rejects_unknown_integration() -> None:
    broker_client = InProcessRequestBrokerClient(
        RequestBroker(HostCredentialRegistry(credentials={}))
    )
    tools = Tools(config={}, request_broker_client=broker_client)

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={
                "method": "GET",
                "url": "https://api.github.com/repos/o/r",
                "integration": "jira",
            },
        )
    )

    assert result.exit_code == 1
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is False
    assert payload["error_code"] == "policy_denied"
    assert "Integration 'jira' is not configured" in payload["message"]


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


def test_tools_http_request_request_error(monkeypatch: Any) -> None:
    def fail_request(**kwargs: Any) -> Any:
        del kwargs
        raise requests.ConnectionError("offline")

    monkeypatch.setattr("broker.request_broker.requests.request", fail_request)
    broker_client = InProcessRequestBrokerClient(
        RequestBroker(HostCredentialRegistry(credentials={}))
    )
    tools = Tools(config={}, request_broker_client=broker_client)

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={"method": "GET", "url": "https://api.example.com"},
        )
    )

    assert result.exit_code == 1
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is False
    assert payload["error_code"] == "request_failed"
    assert "HTTP request failed" in payload["message"]


def test_tools_http_request_does_not_use_direct_requests_path(monkeypatch: Any) -> None:
    def fail_direct_request(*args: Any, **kwargs: Any) -> Any:
        del args
        del kwargs
        raise AssertionError("tools http_request must not call requests.request directly")

    monkeypatch.setattr("requests.request", fail_direct_request)
    broker = _FakeBrokerClient(
        {
            "success": True,
            "status_code": 200,
            "headers": {},
            "body": "ok",
            "truncated": False,
            "integration": None,
        }
    )
    tools = Tools(config={}, request_broker_client=broker)

    result = tools.execute(
        ToolCall(
            tool="http_request",
            args={"method": "GET", "url": "https://example.com"},
        )
    )

    assert result.exit_code == 0
    assert len(broker.requests) == 1


def _unwrap_data(text: str) -> dict[str, Any]:
    prefix = "--- BEGIN DATA ---\n"
    suffix = "\n--- END DATA ---"
    assert text.startswith(prefix)
    assert text.endswith(suffix)
    payload = text[len(prefix) : -len(suffix)]
    loaded = json.loads(payload)
    assert isinstance(loaded, dict)
    return loaded
