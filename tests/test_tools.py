"""Unit tests for built-in tools module."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

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


def _unwrap_data(text: str) -> dict[str, Any]:
    prefix = "--- BEGIN DATA ---\n"
    suffix = "\n--- END DATA ---"
    assert text.startswith(prefix)
    assert text.endswith(suffix)
    payload = text[len(prefix) : -len(suffix)]
    loaded = json.loads(payload)
    assert isinstance(loaded, dict)
    return loaded
