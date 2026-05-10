"""Tests for request-broker policy validation primitives."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest
import requests
import responses

from sandbox.broker import PolicyResult, RequestBroker


def _broker() -> RequestBroker:
    return RequestBroker(credentials={}, config={})


def _policy() -> dict[str, object]:
    return {
        "name": "notion",
        "allowed_methods": ["GET"],
        "allowed_hosts": ["api.notion.com"],
        "allowed_paths": ["/v1/*"],
        "protected_headers": ["Authorization"],
    }


def test_validate_denies_method_mismatch() -> None:
    broker = _broker()
    policy = _policy()

    result = broker._validate(policy, "POST", "https://api.notion.com/v1/pages", {})

    assert result == PolicyResult(
        allowed=False,
        reason="method POST not in allowed_methods ['GET'] for integration 'notion'",
    )


def test_validate_denies_host_mismatch() -> None:
    broker = _broker()
    policy = _policy()

    result = broker._validate(policy, "GET", "https://evil.example.com/v1/pages", {})

    assert result == PolicyResult(
        allowed=False,
        reason="host evil.example.com not in allowed_hosts for integration 'notion'",
    )


def test_validate_denies_path_mismatch() -> None:
    broker = _broker()
    policy = _policy()

    result = broker._validate(policy, "GET", "https://api.notion.com/admin", {})

    assert result == PolicyResult(
        allowed=False,
        reason="path /admin not matched by allowed_paths ['/v1/*'] for integration 'notion'",
    )


def test_validate_denies_protected_header_case_insensitive() -> None:
    broker = _broker()
    policy = _policy()

    result = broker._validate(
        policy,
        "GET",
        "https://api.notion.com/v1/pages",
        {"authorization": "Bearer abc"},
    )

    assert result == PolicyResult(
        allowed=False,
        reason="header 'Authorization' is protected for integration 'notion'",
    )


def test_validate_allows_valid_request() -> None:
    broker = _broker()
    policy = _policy()

    result = broker._validate(policy, "GET", "https://api.notion.com/v1/pages", {"X-Trace": "1"})

    assert result == PolicyResult(allowed=True, reason=None)


def test_deny_shape_and_token_safety() -> None:
    broker = _broker()
    token = "should-never-appear"
    reason = (
        "path /v1/databases not matched by allowed_paths ['/v1/pages/*'] "
        "for integration 'notion'"
    )

    denied = broker._deny("notion", "post", "https://api.notion.com/v1/databases", reason)

    assert denied == {
        "success": False,
        "error": "policy_denied",
        "reason": reason,
        "integration": "notion",
        "requested_method": "POST",
        "requested_url": "https://api.notion.com/v1/databases",
    }
    for value in denied.values():
        assert token not in str(value)


def _fake_getaddrinfo(ip: str) -> list[tuple[object, object, object, object, tuple[str, int]]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


def test_ssrf_check_denies_10_slash_8(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: _fake_getaddrinfo("10.1.2.3"))

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to reserved address 10.1.2.3",
    )


def test_ssrf_check_denies_172_16_slash_12(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo("172.16.12.34"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to reserved address 172.16.12.34",
    )


def test_ssrf_check_denies_192_168_slash_16(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo("192.168.55.9"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to reserved address 192.168.55.9",
    )


def test_ssrf_check_denies_127_slash_8(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo("127.0.0.1"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to reserved address 127.0.0.1",
    )


def test_ssrf_check_denies_169_254_slash_16(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo("169.254.22.7"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to reserved address 169.254.22.7",
    )


def test_ssrf_check_denies_ipv6_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: _fake_getaddrinfo("::1"))

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to reserved address ::1",
    )


def test_ssrf_check_denies_ipv6_fc00(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo("fc00::1234"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to reserved address fc00::1234",
    )


def test_ssrf_check_allows_public_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo("93.184.216.34"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(allowed=True, reason=None)


def test_ssrf_check_dns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()

    def _raise_gaierror(host: str, port: object) -> list[object]:
        raise socket.gaierror("no address")

    monkeypatch.setattr(socket, "getaddrinfo", _raise_gaierror)

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="DNS resolution failed for example.com",
    )


@responses.activate
def test_execute_returns_status_headers_body_without_truncation() -> None:
    broker = _broker()
    responses.add(
        responses.GET,
        "https://example.com/data",
        body="hello world",
        status=200,
        headers={"Content-Type": "text/plain", "X-Trace": "abc123"},
    )

    result = broker._execute("GET", "https://example.com/data", {"X-Test": "1"}, None, 1024)

    assert result["status_code"] == 200
    assert result["headers"]["Content-Type"] == "text/plain"
    assert result["body"] == "hello world"
    assert result["truncated"] is False


@responses.activate
def test_execute_truncates_when_body_exceeds_max_bytes() -> None:
    broker = _broker()
    responses.add(
        responses.GET,
        "https://example.com/long",
        body="abcdefghij",
        status=200,
        headers={"Content-Type": "text/plain"},
    )

    result = broker._execute("GET", "https://example.com/long", {}, None, 5)

    assert result["status_code"] == 200
    assert result["body"] == "abcde"
    assert result["truncated"] is True


def test_execute_returns_structured_error_on_request_exception() -> None:
    broker = _broker()
    with patch.object(
        requests.Session,
        "request",
        side_effect=requests.ConnectionError("connection failed"),
    ):
        result = broker._execute("GET", "https://example.com/down", {}, None, 1024)

    assert result == {
        "success": False,
        "error": "ConnectionError",
        "detail": "connection failed",
    }


class _FakeResponse:
    status_code = 200
    headers = {"Content-Type": "text/plain"}
    encoding = "utf-8"

    def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
        _ = chunk_size
        return [b"ok"]


def test_inject_bearer_captures_outbound_header_key_and_no_token_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = _broker()
    token = "token-bearer-secret"
    policy = {
        "name": "notion",
        "auth_type": "bearer",
        "token": token,
        "default_headers": {"X-Default": "yes"},
    }

    captured: dict[str, object] = {}

    def _fake_request(*args: object, **kwargs: object) -> _FakeResponse:
        _ = args
        captured.update(kwargs)
        return _FakeResponse()

    with caplog.at_level("DEBUG"):
        final_headers, final_url = broker._inject(policy, {"X-Client": "1"}, "https://example.com")

    with patch.object(requests.Session, "request", side_effect=_fake_request):
        broker._execute("GET", final_url, final_headers, None, 1024)

    outbound_headers = captured["headers"]
    assert isinstance(outbound_headers, dict)
    assert "Authorization" in outbound_headers
    assert "X-Default" in outbound_headers
    assert "X-Client" in outbound_headers
    assert token not in caplog.text


def test_inject_custom_header_captures_outbound_header_key_and_no_token_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = _broker()
    token = "token-header-secret"
    policy = {
        "name": "github",
        "auth_type": "header",
        "header_name": "X-API-Key",
        "token": token,
    }

    captured: dict[str, object] = {}

    def _fake_request(*args: object, **kwargs: object) -> _FakeResponse:
        _ = args
        captured.update(kwargs)
        return _FakeResponse()

    with caplog.at_level("DEBUG"):
        final_headers, final_url = broker._inject(policy, {}, "https://example.com")

    with patch.object(requests.Session, "request", side_effect=_fake_request):
        broker._execute("GET", final_url, final_headers, None, 1024)

    outbound_headers = captured["headers"]
    assert isinstance(outbound_headers, dict)
    assert "X-API-Key" in outbound_headers
    assert token not in caplog.text


def test_inject_query_appends_token_param_and_no_token_in_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = _broker()
    token = "token-query-secret"
    policy = {
        "name": "search",
        "auth_type": "query",
        "query_param": "api_key",
        "token": token,
    }

    captured: dict[str, object] = {}

    def _fake_request(*args: object, **kwargs: object) -> _FakeResponse:
        _ = args
        captured.update(kwargs)
        return _FakeResponse()

    with caplog.at_level("DEBUG"):
        final_headers, final_url = broker._inject(
            policy,
            {"X-Client": "1"},
            "https://example.com/search?q=test",
        )

    with patch.object(requests.Session, "request", side_effect=_fake_request):
        broker._execute("GET", final_url, final_headers, None, 1024)

    outbound_url = captured["url"]
    assert isinstance(outbound_url, str)
    assert "api_key=" in outbound_url
    assert outbound_url.startswith("https://example.com/search?")
    assert token not in caplog.text
