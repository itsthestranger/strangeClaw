"""Unit tests for built-in tools module."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import requests

from agent.tools import Tools


@dataclass(slots=True)
class _Call:
    tool: str
    args: dict[str, object]


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self) -> Any:
        return self._payload


class _FakeStreamingResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"http {self.status_code}")

    def iter_content(self, chunk_size: int = 8192) -> Any:
        del chunk_size
        yield self._body

    def close(self) -> None:
        return


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


def test_tools_shell_execute_success() -> None:
    tools = Tools(config={})

    result = tools.execute(_Call(tool="shell", args={"command": "printf 'hello'"}))

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

    result = tools.execute(_Call(tool="shell", args={"command": "echo should-not-run"}))

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "tool shell is not enabled."


def test_tools_shell_output_truncates_long_text() -> None:
    tools = Tools(config={})
    command = "python - <<'PY'\nprint('a' * 9005)\nPY"

    result = tools.execute(_Call(tool="shell", args={"command": command}))

    assert result.exit_code == 0
    assert "...[truncated" in result.stdout
    assert len(result.stdout) < 9005


def test_tools_web_search_normalizes_brave(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse(
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
        )

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    tools = Tools(
        config={
            "web_search": {
                "endpoint": "https://api.search.brave.com/res/v1/web/search",
                "format": "brave",
                "api_key": "test-token",
                "max_results": 1,
            }
        }
    )

    result = tools.execute(_Call(tool="web_search", args={"query": "llm"}))

    assert result.exit_code == 0
    assert result.stderr == ""
    assert captured["url"] == "https://api.search.brave.com/res/v1/web/search"
    assert captured["kwargs"]["headers"] == {
        "User-Agent": "strangeclaw/0.1 (+local)",
        "X-Subscription-Token": "test-token",
    }
    assert captured["kwargs"]["params"] == {"q": "llm"}

    body = _unwrap_data(result.stdout)
    assert body["query"] == "llm"
    assert body["results"] == [
        {"title": "Result One", "url": "https://example.com/1", "snippet": "Snippet 1"}
    ]


def test_tools_web_search_normalizes_searxng(monkeypatch: Any) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeResponse:
        assert url == "http://localhost:8080/search"
        assert kwargs["headers"] == {
            "User-Agent": "strangeclaw/0.1 (+local)",
            "Accept": "application/json",
        }
        assert kwargs["params"] == {"q": "solid state batteries", "format": "json"}
        return _FakeResponse(
            {
                "results": [
                    {"title": "S1", "url": "https://s1", "content": "C1"},
                    {"title": "S2", "url": "https://s2", "content": "C2"},
                ]
            }
        )

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    tools = Tools(
        config={
            "web_search": {
                "endpoint": "http://localhost:8080/search",
                "format": "searxng",
                "max_results": 10,
            }
        }
    )

    result = tools.execute(_Call(tool="web_search", args={"query": "solid state batteries"}))

    assert result.exit_code == 0
    assert _unwrap_data(result.stdout)["results"] == [
        {"title": "S1", "url": "https://s1", "snippet": "C1"},
        {"title": "S2", "url": "https://s2", "snippet": "C2"},
    ]


def test_tools_web_search_brave_requires_api_key() -> None:
    tools = Tools(
        config={
            "web_search": {
                "endpoint": "https://api.search.brave.com/res/v1/web/search",
                "format": "brave",
                "api_key": "",
            }
        }
    )

    result = tools.execute(_Call(tool="web_search", args={"query": "test"}))

    assert result.exit_code == 1
    assert result.stderr == "web_search.api_key is required when web_search.format is brave."


def test_tools_web_fetch_html_uses_trafilatura(monkeypatch: Any) -> None:
    html = b"<html><head><title>Example</title></head><body><p>Hello</p></body></html>"

    def fake_get(url: str, **kwargs: Any) -> _FakeStreamingResponse:
        assert url == "https://example.com"
        assert kwargs["stream"] is True
        return _FakeStreamingResponse(
            headers={"Content-Type": "text/html; charset=utf-8"},
            body=html,
        )

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    monkeypatch.setattr("agent.tools.trafilatura", _FakeTrafilatura())
    tools = Tools(config={"web_fetch": {"max_chars": 20000}})

    result = tools.execute(_Call(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is True
    assert payload["content_type"] == "text/html"
    assert payload["title"] == "Example"
    assert payload["text"] == "main extracted text"
    assert payload["truncated"] is False


def test_tools_web_fetch_json_returns_body(monkeypatch: Any) -> None:
    body = b'{"ok":true}'

    def fake_get(url: str, **kwargs: Any) -> _FakeStreamingResponse:
        del kwargs
        assert url == "https://api.example.com/data"
        return _FakeStreamingResponse(
            headers={"Content-Type": "application/json"},
            body=body,
        )

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    tools = Tools(config={})

    result = tools.execute(_Call(tool="web_fetch", args={"url": "https://api.example.com/data"}))

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["content_type"] == "application/json"
    assert payload["text"] == '{"ok":true}'


def test_tools_web_fetch_pdf_returns_metadata_hint(monkeypatch: Any) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeStreamingResponse:
        del kwargs
        assert url == "https://example.com/doc.pdf"
        return _FakeStreamingResponse(
            headers={"Content-Type": "application/pdf"},
            body=b"%PDF-1.4",
        )

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    tools = Tools(config={})

    result = tools.execute(_Call(tool="web_fetch", args={"url": "https://example.com/doc.pdf"}))

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["content_type"] == "application/pdf"
    assert "Use shell tool with pdftotext" in str(payload["text"])


def test_tools_web_fetch_truncates_at_max_chars(monkeypatch: Any) -> None:
    def fake_get(url: str, **kwargs: Any) -> _FakeStreamingResponse:
        del kwargs
        assert url == "https://example.com/text"
        return _FakeStreamingResponse(
            headers={"Content-Type": "text/plain"},
            body=b"abcdefghijklmnopqrstuvwxyz",
        )

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    tools = Tools(config={"web_fetch": {"max_chars": 10}})

    result = tools.execute(_Call(tool="web_fetch", args={"url": "https://example.com/text"}))

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["truncated"] is True
    assert str(payload["text"]).startswith("abcdefghij")
    assert "truncated, original" in str(payload["text"])


def test_tools_web_fetch_handles_request_error(monkeypatch: Any) -> None:
    def fake_get(url: str, **kwargs: Any) -> Any:
        del url
        del kwargs
        raise requests.Timeout("boom")

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    tools = Tools(config={})

    result = tools.execute(_Call(tool="web_fetch", args={"url": "https://example.com"}))

    assert result.exit_code == 1
    payload = _unwrap_data(result.stdout)
    assert payload["success"] is False
    assert "web_fetch request failed" in str(payload["error"])


def test_tools_web_fetch_applies_5mb_cap(monkeypatch: Any) -> None:
    body = b"a" * (6 * 1024 * 1024)

    def fake_get(url: str, **kwargs: Any) -> _FakeStreamingResponse:
        del kwargs
        assert url == "https://example.com/huge"
        return _FakeStreamingResponse(
            headers={"Content-Type": "text/plain"},
            body=body,
        )

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    tools = Tools(config={"web_fetch": {"max_chars": 6000000}})

    result = tools.execute(_Call(tool="web_fetch", args={"url": "https://example.com/huge"}))

    assert result.exit_code == 0
    payload = _unwrap_data(result.stdout)
    assert payload["truncated"] is True
    assert "response body capped at 5242880 bytes" in str(payload["text"])


def _unwrap_data(text: str) -> dict[str, Any]:
    prefix = "--- BEGIN DATA ---\n"
    suffix = "\n--- END DATA ---"
    assert text.startswith(prefix)
    assert text.endswith(suffix)
    payload = text[len(prefix) : -len(suffix)]
    loaded = json.loads(payload)
    assert isinstance(loaded, dict)
    return loaded
