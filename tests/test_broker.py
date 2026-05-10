"""Tests for request-broker policy validation primitives."""

from __future__ import annotations

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
