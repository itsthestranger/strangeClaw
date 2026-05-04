"""Tests for the host-side request broker policy engine."""

from __future__ import annotations

from typing import Any

import pytest
import requests

from broker.credentials import HostCredential, HostCredentialRegistry
from broker.request_broker import RequestBroker, RequestBrokerConfig


class _FakeResponse:
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


def _registry(token: str = "sentinel-non-pattern-token") -> HostCredentialRegistry:
    return HostCredentialRegistry(
        credentials={
            "notion": HostCredential(
                name="notion",
                credential_type="bearer",
                token=token,
                allowed_hosts=("api.notion.com",),
                allowed_methods=("GET", "POST", "PATCH"),
                allowed_paths=("/v1/pages", "/v1/data_sources/*"),
                default_headers={"Notion-Version": "2026-03-11"},
            )
        }
    )


def test_anonymous_get_to_public_host_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_request(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(status_code=200, text='{"ok":true}')

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(HostCredentialRegistry(credentials={}))

    result = broker.execute(
        {
            "method": "GET",
            "url": "https://example.com/resource",
            "headers": {"Accept": "application/json"},
            "body": None,
        }
    )

    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["integration"] is None
    assert captured["url"] == "https://example.com/resource"
    assert captured["headers"] == {"Accept": "application/json"}


def test_integration_request_injects_auth_only_in_outbound_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_request(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(status_code=200, text='{"created":true}')

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(_registry(token="SENTINEL-TOKEN-123"))

    result = broker.execute(
        {
            "method": "POST",
            "url": "https://api.notion.com/v1/pages",
            "integration": "notion",
            "headers": {"Content-Type": "application/json"},
            "body": "{}",
        }
    )

    assert result["success"] is True
    assert captured["headers"]["Authorization"] == "Bearer SENTINEL-TOKEN-123"
    assert captured["headers"]["Notion-Version"] == "2026-03-11"
    assert "SENTINEL-TOKEN-123" not in repr(result)


@pytest.mark.parametrize(
    ("method", "url", "match"),
    [
        ("POST", "https://api.notion.com/v1/blocks", "Path"),
        ("DELETE", "https://api.notion.com/v1/pages", "Method"),
        ("GET", "https://api.github.com/repos/x/y", "Host"),
    ],
)
def test_disallowed_integration_policy_denied_before_network(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    url: str,
    match: str,
) -> None:
    called = False

    def fake_request(**kwargs: Any) -> _FakeResponse:
        nonlocal called
        called = True
        raise AssertionError(f"network should not be called: {kwargs}")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(_registry())

    result = broker.execute(
        {
            "method": method,
            "url": url,
            "integration": "notion",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error_code"] == "policy_denied"
    assert match in str(result["message"])
    assert called is False


def test_exact_and_wildcard_path_allowlists_work(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_request(**kwargs: Any) -> _FakeResponse:
        calls.append(dict(kwargs))
        return _FakeResponse(status_code=200, text="ok")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(_registry())

    exact = broker.execute(
        {
            "method": "GET",
            "url": "https://api.notion.com/v1/pages",
            "integration": "notion",
            "headers": {},
            "body": None,
        }
    )
    wildcard = broker.execute(
        {
            "method": "GET",
            "url": "https://api.notion.com/v1/data_sources/abc123",
            "integration": "notion",
            "headers": {},
            "body": None,
        }
    )

    assert exact["success"] is True
    assert wildcard["success"] is True
    assert len(calls) == 2


def test_protected_header_override_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_request(**kwargs: Any) -> _FakeResponse:
        nonlocal called
        called = True
        raise AssertionError(f"network should not be called: {kwargs}")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(_registry())

    result = broker.execute(
        {
            "method": "POST",
            "url": "https://api.notion.com/v1/pages",
            "integration": "notion",
            "headers": {"Authorization": "Bearer pasted-token"},
            "body": "{}",
        }
    )

    assert result["success"] is False
    assert result["error_code"] == "policy_denied"
    assert "protected header" in str(result["message"]) 
    assert called is False


def test_redirect_to_disallowed_host_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_request(**kwargs: Any) -> _FakeResponse:
        calls.append(dict(kwargs))
        return _FakeResponse(
            status_code=302,
            headers={"Location": "https://evil.example.com/hijack"},
            text="",
        )

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(_registry())

    result = broker.execute(
        {
            "method": "GET",
            "url": "https://api.notion.com/v1/pages",
            "integration": "notion",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error_code"] == "policy_denied"
    assert "Redirect denied by policy" in str(result["message"])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/test",
        "http://127.0.0.1:8000/test",
        "http://10.0.0.8/private",
        "http://169.254.169.254/latest/meta-data",
    ],
)
def test_private_local_metadata_targets_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    url: str,
) -> None:
    called = False

    def fake_request(**kwargs: Any) -> _FakeResponse:
        nonlocal called
        called = True
        raise AssertionError(f"network should not be called: {kwargs}")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(HostCredentialRegistry(credentials={}))

    result = broker.execute(
        {
            "method": "GET",
            "url": url,
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error_code"] == "policy_denied"
    assert called is False


def test_synthetic_integration_uses_same_generic_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_request(**kwargs: Any) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(status_code=200, text="{\"ok\":true}")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    registry = HostCredentialRegistry(
        credentials={
            "linear": HostCredential(
                name="linear",
                credential_type="bearer",
                token="linear-sentinel-token",
                allowed_hosts=("api.linear.app",),
                allowed_methods=("POST",),
                allowed_paths=("/graphql",),
                default_headers={"Content-Type": "application/json"},
            )
        }
    )
    broker = RequestBroker(registry)

    result = broker.execute(
        {
            "method": "POST",
            "url": "https://api.linear.app/graphql",
            "integration": "linear",
            "headers": {},
            "body": '{"query":"{ viewer { id } }"}',
        }
    )

    assert result["success"] is True
    assert captured["headers"]["Authorization"] == "Bearer linear-sentinel-token"


def test_response_body_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(**kwargs: Any) -> _FakeResponse:
        del kwargs
        return _FakeResponse(status_code=200, text="abcdefghijklmnopqrstuvwxyz")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(
        HostCredentialRegistry(credentials={}),
        config=RequestBrokerConfig(max_response_body_chars=10),
    )

    result = broker.execute(
        {
            "method": "GET",
            "url": "https://example.com/large",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is True
    assert result["truncated"] is True
    assert str(result["body"]).startswith("abcdefghij")


def test_redirect_policy_for_anonymous_blocks_private_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_request(**kwargs: Any) -> _FakeResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _FakeResponse(status_code=302, headers={"Location": "http://127.0.0.1/internal"})
        raise AssertionError(f"second request should not be executed: {kwargs}")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(HostCredentialRegistry(credentials={}))

    result = broker.execute(
        {
            "method": "GET",
            "url": "https://example.com/start",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error_code"] == "policy_denied"
    assert call_count == 1


def test_redaction_context_applies_to_success_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "totally-unique-sentinel-token"
    registry = _registry(token=sentinel)

    def fake_request(**kwargs: Any) -> _FakeResponse:
        if kwargs["url"].endswith("/ok"):
            return _FakeResponse(
                status_code=200,
                headers={"Set-Cookie": f"auth={sentinel}"},
                text=f"response-body-contains-{sentinel}",
            )
        raise requests.RequestException(f"boom {sentinel}")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(registry)

    success = broker.execute(
        {
            "method": "GET",
            "url": "https://api.notion.com/v1/data_sources/ok",
            "integration": "notion",
            "headers": {},
            "body": None,
        }
    )

    failure = broker.execute(
        {
            "method": "GET",
            "url": "https://api.notion.com/v1/data_sources/fail",
            "integration": "notion",
            "headers": {},
            "body": None,
        }
    )

    assert success["success"] is True
    assert failure["success"] is False
    assert sentinel not in repr(success)
    assert sentinel not in repr(failure)

    redacted = broker.redact_for_logging({"payload": f"x-{sentinel}-x"})
    assert redacted == {"payload": "x-[REDACTED]-x"}
    assert sentinel in broker.known_secret_values


def test_broker_rejects_request_body_over_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake_request(**kwargs: Any) -> _FakeResponse:
        nonlocal called
        called = True
        raise AssertionError(f"network should not be called: {kwargs}")

    monkeypatch.setattr("broker.request_broker.requests.request", fake_request)
    broker = RequestBroker(
        HostCredentialRegistry(credentials={}),
        config=RequestBrokerConfig(max_request_body_chars=4),
    )

    result = broker.execute(
        {
            "method": "POST",
            "url": "https://example.com/upload",
            "headers": {},
            "body": "12345",
        }
    )

    assert result["success"] is False
    assert result["error_code"] == "policy_denied"
    assert called is False
