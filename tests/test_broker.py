"""Tests for request-broker policy validation primitives."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
import requests
import responses

from agent.broker_client import BrokerClient
from sandbox.broker import Policy, PolicyResult, RequestBroker
from sandbox.host_services import HostServiceServer


def _broker() -> RequestBroker:
    return RequestBroker(credentials={}, config={})


def _policy() -> Policy:
    return Policy(
        name="notion",
        auth_type="bearer",
        token="notion-secret-token",
        header_name="Authorization",
        allowed_methods=("GET",),
        allowed_hosts=("api.notion.com",),
        allowed_paths=("/v1/*",),
        allowed_schemes=("https",),
        protected_headers=("Authorization",),
        default_headers={},
        max_response_bytes=4096,
        rate_limit_requests=None,
        rate_limit_period_seconds=None,
    )


def _broker_config() -> dict[str, object]:
    return {
        "broker": {
            "public_policy": {
                "enabled": True,
                "allowed_methods": ["GET"],
                "max_response_bytes": 4096,
            }
        },
        "web_fetch": {"max_chars": 20000},
        "web_search": {
            "endpoint": "https://search.example.com/search",
            "format": "brave",
            "max_results": 10,
        },
    }


def _credentials() -> dict[str, dict[str, object]]:
    return {
        "notion": {
            "name": "notion",
            "auth_type": "bearer",
            "token": "notion-secret-token",
            "allowed_hosts": ["api.notion.com"],
            "allowed_methods": ["GET", "POST"],
            "allowed_paths": ["/v1/*"],
            "allowed_schemes": ["https"],
            "protected_headers": ["Authorization"],
            "default_headers": {"Notion-Version": "2022-06-28"},
            "max_response_bytes": 4096,
            "rate_limit": {"requests": 10, "per_seconds": 1},
        },
        "_web_search": {
            "name": "_web_search",
            "auth_type": "header",
            "header_name": "X-Subscription-Token",
            "token": "search-secret-token",
            "allowed_hosts": ["search.example.com"],
            "allowed_methods": ["GET"],
            "allowed_paths": ["/*"],
            "allowed_schemes": ["https"],
            "protected_headers": ["Authorization"],
            "default_headers": {},
            "max_response_bytes": 4096,
            "rate_limit": None,
        },
        "_internal": {
            "name": "_internal",
            "auth_type": "header",
            "token": "hidden-token",
            "allowed_hosts": ["internal.example.com"],
            "allowed_methods": ["GET"],
            "allowed_paths": ["/*"],
            "allowed_schemes": ["https"],
            "protected_headers": ["Authorization"],
            "default_headers": {},
            "max_response_bytes": 4096,
            "rate_limit": None,
        },
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


def _fake_getaddrinfo_v6(
    ip: str,
) -> list[tuple[object, object, object, object, tuple[str, int, int, int]]]:
    return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0))]


def test_ssrf_check_denies_unsupported_scheme() -> None:
    broker = _broker()

    result = broker._ssrf_check("ftp://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="unsupported URL scheme 'ftp' for public URL policy",
    )


def test_ssrf_check_denies_10_slash_8(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: _fake_getaddrinfo("10.1.2.3"))

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to blocked address 10.1.2.3",
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
        reason="SSRF: example.com resolves to blocked address 172.16.12.34",
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
        reason="SSRF: example.com resolves to blocked address 192.168.55.9",
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
        reason="SSRF: example.com resolves to blocked address 127.0.0.1",
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
        reason="SSRF: example.com resolves to blocked address 169.254.22.7",
    )


def test_ssrf_check_denies_ipv4_unspecified(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: _fake_getaddrinfo("0.0.0.0"))

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to blocked address 0.0.0.0",
    )


def test_ssrf_check_denies_ipv4_multicast(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo("224.0.0.1"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to blocked address 224.0.0.1",
    )


def test_ssrf_check_denies_ipv6_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: _fake_getaddrinfo_v6("::1"))

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to blocked address ::1",
    )


def test_ssrf_check_denies_ipv6_unique_local(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo_v6("fc00::1234"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to blocked address fc00::1234",
    )


def test_ssrf_check_denies_ipv6_link_local(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo_v6("fe80::1"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to blocked address fe80::1",
    )


def test_ssrf_check_denies_ipv6_multicast(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo_v6("ff02::1"),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to blocked address ff02::1",
    )


def test_ssrf_check_denies_ipv6_unspecified(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port: _fake_getaddrinfo_v6("::"))

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason="SSRF: example.com resolves to blocked address ::",
    )


@pytest.mark.parametrize(
    ("mapped_ipv6", "expected_ipv4"),
    [
        ("::ffff:127.0.0.1", "127.0.0.1"),
        ("::ffff:10.0.0.1", "10.0.0.1"),
        ("::ffff:169.254.1.1", "169.254.1.1"),
    ],
)
def test_ssrf_check_denies_ipv4_mapped_ipv6(
    monkeypatch: pytest.MonkeyPatch,
    mapped_ipv6: str,
    expected_ipv4: str,
) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo_v6(mapped_ipv6),
    )

    result = broker._ssrf_check("https://example.com/path")

    assert result == PolicyResult(
        allowed=False,
        reason=f"SSRF: example.com resolves to blocked address {expected_ipv4}",
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


def test_ssrf_check_allows_public_ipv6(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = _broker()
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo_v6("2606:2800:220:1:248:1893:25c8:1946"),
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

    assert result["success"] is True
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

    assert result["success"] is True
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


def test_execute_disables_env_proxy_inheritance() -> None:
    broker = _broker()

    def _fake_request(session: requests.Session, *args: object, **kwargs: object) -> _FakeResponse:
        _ = args, kwargs
        assert session.trust_env is False
        return _FakeResponse()

    with patch.object(requests.Session, "request", autospec=True, side_effect=_fake_request):
        result = broker._execute("GET", "https://example.com/data", {}, None, 1024)

    assert result["success"] is True


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
    policy = Policy(
        name="notion",
        auth_type="bearer",
        token=token,
        header_name="Authorization",
        allowed_methods=("GET",),
        allowed_hosts=("api.notion.com",),
        allowed_paths=("/*",),
        allowed_schemes=("https",),
        protected_headers=("Authorization",),
        default_headers={"X-Default": "yes"},
        max_response_bytes=4096,
        rate_limit_requests=None,
        rate_limit_period_seconds=None,
    )

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
    policy = Policy(
        name="github",
        auth_type="header",
        token=token,
        header_name="X-API-Key",
        allowed_methods=("GET",),
        allowed_hosts=("api.github.com",),
        allowed_paths=("/*",),
        allowed_schemes=("https",),
        protected_headers=("Authorization", "X-API-Key"),
        default_headers={},
        max_response_bytes=4096,
        rate_limit_requests=None,
        rate_limit_period_seconds=None,
    )

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


def test_inject_unknown_auth_type_does_not_inject_token_or_modify_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broker = _broker()
    token = "token-unknown-secret"
    policy = Policy(
        name="search",
        auth_type="unsupported",
        token=token,
        header_name="Authorization",
        allowed_methods=("GET",),
        allowed_hosts=("search.example.com",),
        allowed_paths=("/*",),
        allowed_schemes=("https",),
        protected_headers=("Authorization",),
        default_headers={},
        max_response_bytes=4096,
        rate_limit_requests=None,
        rate_limit_period_seconds=None,
    )

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
    outbound_headers = captured["headers"]
    assert isinstance(outbound_url, str)
    assert outbound_url == "https://example.com/search?q=test"
    assert isinstance(outbound_headers, dict)
    assert "X-Client" in outbound_headers
    assert "Authorization" not in outbound_headers
    assert token not in caplog.text


def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: _fake_getaddrinfo("93.184.216.34"),
    )


def _mapped_dns(
    monkeypatch: pytest.MonkeyPatch,
    mapping: dict[str, str],
) -> None:
    def _resolver(
        host: str,
        port: object,
    ) -> list[tuple[object, object, object, object, tuple[str, int]]]:
        _ = port
        ip = mapping.get(host, "93.184.216.34")
        return _fake_getaddrinfo(ip)

    monkeypatch.setattr(socket, "getaddrinfo", _resolver)


def test_handle_unknown_action() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())

    result = broker.handle({"action": "nope"})

    assert result == {"success": False, "error": "unknown action: nope"}


def test_handle_wraps_internal_handler_exception_with_redacted_envelope() -> None:
    creds = _credentials()
    secret = str(creds["notion"]["token"])
    broker = RequestBroker(credentials=creds, config=_broker_config())

    def _explode(payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        raise RuntimeError(f"upstream exploded with token={secret}")

    broker._handlers["explode"] = _explode  # noqa: SLF001

    result = broker.handle({"action": "explode"})

    assert result["success"] is False
    assert result["error"] == "internal_error"
    detail = str(result.get("detail", ""))
    assert secret not in detail
    assert "[REDACTED]" in detail


def test_handle_rejects_missing_success_envelope_for_handler_result() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    broker._handlers["bad_payload"] = lambda payload: {"echo": payload}  # noqa: SLF001

    result = broker.handle({"action": "bad_payload", "x": 1})

    assert result == {
        "success": False,
        "error": "internal_error",
        "detail": "invalid broker response: missing success envelope",
    }


def test_handle_list_integrations_skips_invalid_policy_records() -> None:
    credentials = _credentials()
    credentials["broken"] = {
        "name": "broken",
        "auth_type": "query",
        "token": "ignored",
        "allowed_hosts": ["api.invalid.local"],
        "allowed_methods": ["GET"],
        "allowed_paths": ["/*"],
        "allowed_schemes": ["https"],
        "protected_headers": ["Authorization"],
        "default_headers": {},
        "max_response_bytes": 4096,
        "rate_limit": None,
    }
    broker = RequestBroker(credentials=credentials, config=_broker_config())

    listed = broker.handle({"action": "list_integrations"})
    denied = broker.handle(
        {
            "action": "http_request",
            "integration": "broken",
            "method": "GET",
            "url": "https://api.invalid.local/v1/check",
        }
    )

    assert listed == {"success": True, "integrations": ["notion"]}
    assert denied["success"] is False
    assert denied["error"] == "policy_denied"
    assert "not found in secrets.yaml" in str(denied.get("reason", ""))


@responses.activate
def test_handle_http_request_public_policy_custom_allowed_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    config = _broker_config()
    config["broker"] = {
        "public_policy": {
            "enabled": True,
            "allowed_methods": ["GET", "POST"],
            "max_response_bytes": 4096,
        }
    }
    broker = RequestBroker(credentials=_credentials(), config=config)
    responses.add(
        responses.POST,
        "https://public.example.com/submit",
        body='{"ok":true}',
        status=200,
        headers={"Content-Type": "application/json"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "method": "POST",
            "url": "https://public.example.com/submit",
            "headers": {"Content-Type": "application/json"},
            "body": '{"x":1}',
        }
    )

    assert result["success"] is True
    assert result["status_code"] == 200


def test_execute_policy_request_helper_denies_disallowed_redirect_host() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    policy = broker._policies["notion"]

    def _fake_execute_with_redirects(
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: str | None,
        max_bytes: int,
        redirect_guard: Callable[[str, str], dict[str, object] | None] | None,
    ) -> dict[str, object]:
        _ = (method, url, headers, body, max_bytes)
        assert redirect_guard is not None
        denied = redirect_guard("https://evil.example.com/v1/pages", "GET")
        assert denied is not None
        return denied

    with patch.object(broker, "_execute_with_redirects", side_effect=_fake_execute_with_redirects):
        result = broker._execute_policy_request(
            policy=policy,
            integration_name="notion",
            method="GET",
            url="https://api.notion.com/v1/pages",
            model_headers={},
            body=None,
        )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert result["integration"] == "notion"
    assert result["requested_method"] == "GET"
    assert result["requested_url"] == "https://evil.example.com/v1/pages"


@responses.activate
def test_handle_http_request_named_integration_success() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.POST,
        "https://api.notion.com/v1/pages",
        json={"id": "abc123"},
        status=200,
        headers={"Content-Type": "application/json"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "POST",
            "url": "https://api.notion.com/v1/pages",
            "headers": {"X-Trace": "1"},
            "body": "{\"title\":\"Test\"}",
        }
    )

    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["truncated"] is False


def test_handle_http_request_named_integration_denies_http_by_default() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "GET",
            "url": "http://api.notion.com/v1/pages",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "allowed_schemes" in str(result.get("reason", ""))


@responses.activate
def test_handle_http_request_named_integration_allows_http_when_explicit() -> None:
    creds = _credentials()
    creds["notion"]["allowed_schemes"] = ["http"]
    broker = RequestBroker(credentials=creds, config=_broker_config())
    responses.add(
        responses.GET,
        "http://api.notion.com/v1/pages",
        json={"ok": True},
        status=200,
        headers={"Content-Type": "application/json"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "GET",
            "url": "http://api.notion.com/v1/pages",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is True
    assert result["status_code"] == 200


@responses.activate
def test_handle_http_request_public_policy_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _public_dns(monkeypatch)
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://public.example.com/data",
        body="hello",
        status=200,
        headers={"Content-Type": "text/plain"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "method": "GET",
            "url": "https://public.example.com/data",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["body"] == "hello"


def test_handle_http_request_public_policy_disabled_denial() -> None:
    config = _broker_config()
    config["broker"] = {"public_policy": {"enabled": False}}
    broker = RequestBroker(credentials=_credentials(), config=config)

    result = broker.handle(
        {"action": "http_request", "method": "GET", "url": "https://public.example.com/"}
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "public requests disabled" in str(result.get("reason", ""))


def test_handle_http_request_public_policy_denies_unsupported_scheme() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())

    result = broker.handle(
        {
            "action": "http_request",
            "method": "GET",
            "url": "ftp://public.example.com/data",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "allowed_schemes" in str(result.get("reason", ""))


def test_handle_web_fetch_denies_unsupported_scheme() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())

    result = broker.handle({"action": "web_fetch", "url": "file:///etc/passwd"})

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "unsupported URL scheme" in str(result.get("reason", ""))


def test_handle_http_request_path_denied_and_no_token_leak() -> None:
    creds = _credentials()
    creds["notion"]["allowed_paths"] = ["/v1/pages/*"]
    broker = RequestBroker(credentials=creds, config=_broker_config())
    result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "POST",
            "url": "https://api.notion.com/v1/databases",
            "headers": {},
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    secret_tokens = {"notion-secret-token", "search-secret-token", "hidden-token"}
    for value in result.values():
        rendered = str(value)
        for token in secret_tokens:
            assert token not in rendered


def test_handle_http_request_redacts_reflected_bearer_token() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())

    class _EchoResponse:
        status_code = 200
        headers = {"Content-Type": "application/json"}
        encoding = "utf-8"

        def __init__(self, body: str) -> None:
            self._body = body

        def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
            _ = chunk_size
            return [self._body.encode("utf-8")]

    def _fake_request(*args: object, **kwargs: object) -> _EchoResponse:
        _ = args
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        auth = str(headers.get("Authorization", ""))
        return _EchoResponse(f'{{"echo":"{auth}"}}')

    with patch.object(requests.Session, "request", side_effect=_fake_request):
        result = broker.handle(
            {
                "action": "http_request",
                "integration": "notion",
                "method": "POST",
                "url": "https://api.notion.com/v1/pages",
                "headers": {},
                "body": '{"title":"x"}',
            }
        )

    rendered = json.dumps(result, ensure_ascii=True, sort_keys=True)
    assert "notion-secret-token" not in rendered
    assert "[REDACTED]" in rendered


def test_handle_http_request_redacts_reflected_custom_header_token() -> None:
    creds = _credentials()
    creds["custom"] = {
        "name": "custom",
        "auth_type": "header",
        "header_name": "X-API-Key",
        "token": "custom-header-secret-token",
        "allowed_hosts": ["api.custom.local"],
        "allowed_methods": ["GET"],
        "allowed_paths": ["/v1/*"],
        "allowed_schemes": ["https"],
        "protected_headers": ["Authorization", "X-API-Key"],
        "default_headers": {},
        "max_response_bytes": 4096,
        "rate_limit": None,
    }
    broker = RequestBroker(credentials=creds, config=_broker_config())

    class _EchoResponse:
        status_code = 200
        headers = {"Content-Type": "application/json"}
        encoding = "utf-8"

        def __init__(self, body: str) -> None:
            self._body = body

        def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
            _ = chunk_size
            return [self._body.encode("utf-8")]

    def _fake_request(*args: object, **kwargs: object) -> _EchoResponse:
        _ = args
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        token_header = str(headers.get("X-API-Key", ""))
        return _EchoResponse(f'{{"echo":"{token_header}"}}')

    with patch.object(requests.Session, "request", side_effect=_fake_request):
        result = broker.handle(
            {
                "action": "http_request",
                "integration": "custom",
                "method": "GET",
                "url": "https://api.custom.local/v1/check",
                "headers": {},
                "body": None,
            }
        )

    rendered = json.dumps(result, ensure_ascii=True, sort_keys=True)
    assert "custom-header-secret-token" not in rendered
    assert "[REDACTED]" in rendered


def test_handle_http_request_redacts_token_in_exception_detail() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())

    with patch.object(
        requests.Session,
        "request",
        side_effect=requests.ConnectionError("upstream echoed notion-secret-token"),
    ):
        result = broker.handle(
            {
                "action": "http_request",
                "integration": "notion",
                "method": "GET",
                "url": "https://api.notion.com/v1/pages",
                "headers": {},
                "body": None,
            }
        )

    rendered = json.dumps(result, ensure_ascii=True, sort_keys=True)
    assert "notion-secret-token" not in rendered
    assert "[REDACTED]" in rendered


@responses.activate
def test_handle_web_fetch_redacts_token_in_body(monkeypatch: pytest.MonkeyPatch) -> None:
    _public_dns(monkeypatch)
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://example.com/page",
        body="token leak search-secret-token in content",
        status=200,
        headers={"Content-Type": "text/plain"},
    )

    result = broker.handle({"action": "web_fetch", "url": "https://example.com/page"})

    rendered = json.dumps(result, ensure_ascii=True, sort_keys=True)
    assert "search-secret-token" not in rendered
    assert "[REDACTED]" in rendered


@responses.activate
def test_handle_http_request_rate_limited_on_n_plus_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _public_dns(monkeypatch)
    creds = _credentials()
    creds["notion"]["rate_limit"] = {"requests": 1, "per_seconds": 60}
    broker = RequestBroker(credentials=creds, config=_broker_config())
    responses.add(
        responses.POST,
        "https://api.notion.com/v1/pages",
        json={"ok": True},
        status=200,
        headers={"Content-Type": "application/json"},
    )

    first = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "POST",
            "url": "https://api.notion.com/v1/pages",
            "headers": {},
        }
    )
    second = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "POST",
            "url": "https://api.notion.com/v1/pages",
            "headers": {},
        }
    )

    assert first["status_code"] == 200
    assert second == {
        "success": False,
        "error": "rate_limited",
        "reason": "rate limit exceeded for integration 'notion'",
    }


@responses.activate
def test_handle_web_fetch_html_runs_trafilatura(monkeypatch: pytest.MonkeyPatch) -> None:
    _public_dns(monkeypatch)
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://example.com/page",
        body="<html><body><article>Main text</article></body></html>",
        status=200,
        headers={"Content-Type": "text/html; charset=utf-8"},
    )

    class _Meta:
        title = "Sample"

    monkeypatch.setattr("sandbox.broker.trafilatura.extract", lambda *args, **kwargs: "MAIN")
    monkeypatch.setattr(
        "sandbox.broker.trafilatura.extract_metadata",
        lambda *args, **kwargs: _Meta(),
    )

    result = broker.handle({"action": "web_fetch", "url": "https://example.com/page"})

    assert result["success"] is True
    assert result["content_type"] == "text/html"
    assert result["title"] == "Sample"
    assert result["text"] == "MAIN"


@responses.activate
def test_handle_web_search_normalizes_brave() -> None:
    config = _broker_config()
    config["web_search"] = {
        "endpoint": "https://search.example.com/search",
        "format": "brave",
        "max_results": 10,
    }
    broker = RequestBroker(credentials=_credentials(), config=config)
    responses.add(
        responses.GET,
        "https://search.example.com/search",
        json={
            "web": {
                "results": [
                    {"title": "A", "url": "https://a.example", "description": "Snippet A"},
                    {"title": "B", "url": "https://b.example", "description": "Snippet B"},
                ]
            }
        },
        status=200,
        headers={"Content-Type": "application/json"},
    )

    result = broker.handle({"action": "web_search", "query": "test"})

    assert result == {
        "success": True,
        "results": [
            {"title": "A", "url": "https://a.example", "snippet": "Snippet A"},
            {"title": "B", "url": "https://b.example", "snippet": "Snippet B"},
        ],
    }


@responses.activate
def test_handle_web_search_normalizes_searxng() -> None:
    config = _broker_config()
    config["web_search"] = {
        "endpoint": "https://search.example.com/search",
        "format": "searxng",
        "max_results": 10,
    }
    broker = RequestBroker(credentials=_credentials(), config=config)
    responses.add(
        responses.GET,
        "https://search.example.com/search",
        json={
            "results": [
                {"title": "X", "url": "https://x.example", "content": "Snippet X"},
                {"title": "Y", "url": "https://y.example", "content": "Snippet Y"},
            ]
        },
        status=200,
        headers={"Content-Type": "application/json"},
    )

    result = broker.handle({"action": "web_search", "query": "test"})

    assert result == {
        "success": True,
        "results": [
            {"title": "X", "url": "https://x.example", "snippet": "Snippet X"},
            {"title": "Y", "url": "https://y.example", "snippet": "Snippet Y"},
        ],
    }


@responses.activate
def test_handle_web_search_denies_allowed_hosts_mismatch() -> None:
    config = _broker_config()
    config["web_search"] = {
        "endpoint": "https://wrong.example.com/search",
        "format": "brave",
        "max_results": 10,
    }
    broker = RequestBroker(credentials=_credentials(), config=config)

    result = broker.handle({"action": "web_search", "query": "test"})

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "allowed_hosts" in str(result.get("reason", ""))
    assert len(responses.calls) == 0


@responses.activate
def test_handle_web_search_denies_allowed_paths_mismatch() -> None:
    config = _broker_config()
    config["web_search"] = {
        "endpoint": "https://search.example.com/private",
        "format": "brave",
        "max_results": 10,
    }
    credentials = _credentials()
    credentials["_web_search"] = {
        **credentials["_web_search"],
        "allowed_paths": ["/search"],
    }
    broker = RequestBroker(credentials=credentials, config=config)

    result = broker.handle({"action": "web_search", "query": "test"})

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "allowed_paths" in str(result.get("reason", ""))
    assert len(responses.calls) == 0


def test_handle_web_search_denies_http_by_default() -> None:
    config = _broker_config()
    config["web_search"] = {
        "endpoint": "http://search.example.com/search",
        "format": "brave",
        "max_results": 10,
    }
    broker = RequestBroker(credentials=_credentials(), config=config)

    result = broker.handle({"action": "web_search", "query": "test"})

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "allowed_schemes" in str(result.get("reason", ""))


@responses.activate
def test_handle_web_search_allows_http_when_explicit() -> None:
    config = _broker_config()
    config["web_search"] = {
        "endpoint": "http://search.example.com/search",
        "format": "brave",
        "max_results": 10,
    }
    creds = _credentials()
    creds["_web_search"]["allowed_schemes"] = ["http"]
    broker = RequestBroker(credentials=creds, config=config)
    responses.add(
        responses.GET,
        "http://search.example.com/search",
        json={
            "web": {
                "results": [
                    {"title": "A", "url": "https://a.example", "description": "Snippet A"},
                ]
            }
        },
        status=200,
        headers={"Content-Type": "application/json"},
    )

    result = broker.handle({"action": "web_search", "query": "test"})

    assert result == {
        "success": True,
        "results": [
            {"title": "A", "url": "https://a.example", "snippet": "Snippet A"},
        ],
    }


def test_handle_http_request_header_auth_auto_protects_header_name() -> None:
    creds = _credentials()
    creds["custom"] = {
        "auth_type": "header",
        "header_name": "X-API-Key",
        "token": "custom-token",
        "allowed_hosts": ["api.custom.local"],
        "allowed_methods": ["GET"],
        "allowed_paths": ["/v1/*"],
        "allowed_schemes": ["https"],
        "protected_headers": ["Authorization"],
        "default_headers": {},
        "max_response_bytes": 4096,
        "rate_limit": None,
    }
    broker = RequestBroker(credentials=creds, config=_broker_config())

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "custom",
            "method": "GET",
            "url": "https://api.custom.local/v1/check",
            "headers": {"X-API-Key": "model-value"},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "X-API-Key" in str(result.get("reason", ""))


def test_broker_logs_invalid_policy_records_without_token_leak(
    caplog: pytest.LogCaptureFixture,
) -> None:
    creds = _credentials()
    secret = "should-not-leak-in-logs"
    creds["bad"] = {
        "auth_type": "header",
        "token": secret,
        "allowed_hosts": ["api.bad.local"],
        "allowed_methods": ["GET"],
        "allowed_paths": ["/*"],
        "allowed_schemes": ["gopher"],
        "protected_headers": ["Authorization"],
        "default_headers": {},
        "max_response_bytes": 4096,
        "rate_limit": None,
    }

    with caplog.at_level("WARNING"):
        broker = RequestBroker(credentials=creds, config=_broker_config())

    result = broker.handle({"action": "list_integrations"})
    assert result == {"success": True, "integrations": ["notion"]}
    assert "skipping integration 'bad'" in caplog.text
    assert secret not in caplog.text


def test_web_search_header_name_is_auto_protected_in_normalized_policy() -> None:
    creds = _credentials()
    creds["_web_search"] = {
        **creds["_web_search"],
        "header_name": "X-Subscription-Token",
        "protected_headers": ["Authorization"],
    }
    broker = RequestBroker(credentials=creds, config=_broker_config())

    policy = broker._policies["_web_search"]
    assert policy.header_name == "X-Subscription-Token"
    assert "X-Subscription-Token" in policy.protected_headers


@responses.activate
def test_handle_web_search_redirect_to_disallowed_host_is_denied() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://search.example.com/search",
        status=302,
        headers={"Location": "https://evil.example.com/search"},
    )

    result = broker.handle({"action": "web_search", "query": "test"})

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "allowed_hosts" in str(result.get("reason", ""))
    assert len(responses.calls) == 1
    request_headers = responses.calls[0].request.headers
    assert "X-Subscription-Token" in request_headers


@responses.activate
def test_redirect_denials_match_between_http_request_and_web_search() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://api.notion.com/v1/pages",
        status=302,
        headers={"Location": "https://evil.example.com/v1/pages"},
    )
    responses.add(
        responses.GET,
        "https://search.example.com/search",
        status=302,
        headers={"Location": "https://evil.example.com/search"},
    )

    http_result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "GET",
            "url": "https://api.notion.com/v1/pages",
            "headers": {},
            "body": None,
        }
    )
    search_result = broker.handle({"action": "web_search", "query": "test"})

    assert http_result["success"] is False
    assert search_result["success"] is False
    assert http_result["error"] == "policy_denied"
    assert search_result["error"] == "policy_denied"
    assert http_result["requested_method"] == "GET"
    assert search_result["requested_method"] == "GET"
    assert http_result["integration"] == "notion"
    assert search_result["integration"] == "_web_search"
    assert "allowed_hosts" in str(http_result.get("reason", ""))
    assert "allowed_hosts" in str(search_result.get("reason", ""))


def test_handle_list_integrations_via_host_service_registration() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    server = HostServiceServer()
    server.register("broker", broker.handle)
    client = BrokerClient(server)

    result = client.call("broker", {"action": "list_integrations"})

    assert result == {"success": True, "integrations": ["notion"]}


@responses.activate
def test_handle_http_request_public_redirect_to_loopback_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mapped_dns(
        monkeypatch,
        {
            "public.example.com": "93.184.216.34",
            "127.0.0.1": "127.0.0.1",
        },
    )
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://public.example.com/start",
        status=302,
        headers={"Location": "http://127.0.0.1/private"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "method": "GET",
            "url": "https://public.example.com/start",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "SSRF" in str(result.get("reason", ""))
    assert len(responses.calls) == 1


@responses.activate
def test_handle_http_request_public_redirect_to_unsupported_scheme_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mapped_dns(monkeypatch, {"public.example.com": "93.184.216.34"})
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://public.example.com/start",
        status=302,
        headers={"Location": "ftp://public.example.com/private"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "method": "GET",
            "url": "https://public.example.com/start",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "allowed_schemes" in str(result.get("reason", ""))
    assert len(responses.calls) == 1


@responses.activate
def test_handle_web_fetch_redirect_to_reserved_address_is_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mapped_dns(
        monkeypatch,
        {
            "example.com": "93.184.216.34",
            "192.168.1.10": "192.168.1.10",
        },
    )
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://example.com/start",
        status=302,
        headers={"Location": "http://192.168.1.10/internal"},
    )

    result = broker.handle({"action": "web_fetch", "url": "https://example.com/start"})

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "SSRF" in str(result.get("reason", ""))
    assert len(responses.calls) == 1


@responses.activate
def test_handle_http_request_named_integration_redirect_to_disallowed_host_is_denied() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://api.notion.com/v1/pages",
        status=302,
        headers={"Location": "https://evil.example.com/v1/pages"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "GET",
            "url": "https://api.notion.com/v1/pages",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "allowed_hosts" in str(result.get("reason", ""))
    assert len(responses.calls) == 1


@responses.activate
def test_handle_http_request_named_integration_redirect_to_disallowed_path_is_denied() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://api.notion.com/v1/pages",
        status=302,
        headers={"Location": "https://api.notion.com/admin"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "GET",
            "url": "https://api.notion.com/v1/pages",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert result["error"] == "policy_denied"
    assert "allowed_paths" in str(result.get("reason", ""))
    assert len(responses.calls) == 1


@responses.activate
def test_handle_http_request_named_integration_no_credential_forward_to_disallowed_redirect_host(
) -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://api.notion.com/v1/pages",
        status=302,
        headers={"Location": "https://evil.example.com/v1/pages"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "GET",
            "url": "https://api.notion.com/v1/pages",
            "headers": {},
            "body": None,
        }
    )

    assert result["success"] is False
    assert len(responses.calls) == 1
    first_request_headers = responses.calls[0].request.headers
    assert "Authorization" in first_request_headers


@responses.activate
def test_handle_http_request_named_integration_redirect_within_policy_succeeds() -> None:
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://api.notion.com/v1/pages",
        status=302,
        headers={"Location": "https://api.notion.com/v1/pages/next"},
    )
    responses.add(
        responses.GET,
        "https://api.notion.com/v1/pages/next",
        json={"ok": True},
        status=200,
        headers={"Content-Type": "application/json"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "integration": "notion",
            "method": "GET",
            "url": "https://api.notion.com/v1/pages",
            "headers": {},
            "body": None,
        }
    )

    assert result["status_code"] == 200
    assert len(responses.calls) == 2


@responses.activate
def test_handle_http_request_redirect_loop_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mapped_dns(monkeypatch, {"public.example.com": "93.184.216.34"})
    broker = RequestBroker(credentials=_credentials(), config=_broker_config())
    responses.add(
        responses.GET,
        "https://public.example.com/r1",
        status=302,
        headers={"Location": "https://public.example.com/r2"},
    )
    responses.add(
        responses.GET,
        "https://public.example.com/r2",
        status=302,
        headers={"Location": "https://public.example.com/r3"},
    )
    responses.add(
        responses.GET,
        "https://public.example.com/r3",
        status=302,
        headers={"Location": "https://public.example.com/r4"},
    )
    responses.add(
        responses.GET,
        "https://public.example.com/r4",
        status=302,
        headers={"Location": "https://public.example.com/r5"},
    )
    responses.add(
        responses.GET,
        "https://public.example.com/r5",
        status=302,
        headers={"Location": "https://public.example.com/r6"},
    )
    responses.add(
        responses.GET,
        "https://public.example.com/r6",
        status=302,
        headers={"Location": "https://public.example.com/r7"},
    )

    result = broker.handle(
        {
            "action": "http_request",
            "method": "GET",
            "url": "https://public.example.com/r1",
            "headers": {},
            "body": None,
        }
    )

    assert result == {
        "success": False,
        "error": "too_many_redirects",
        "detail": "redirect limit exceeded (5) for https://public.example.com/r1",
    }
    assert len(responses.calls) == 6
