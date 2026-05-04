"""Tests for shared redaction helpers."""

from __future__ import annotations

from typing import Any

from broker.credentials import HostCredential, HostCredentialRegistry
from broker.redaction import redact_sensitive, redact_text, secret_values_from_registry


def test_redact_sensitive_redacts_keys_recursively_without_mutation() -> None:
    original: dict[str, Any] = {
        "integration": "notion",
        "headers": {
            "Authorization": "Bearer abc123",
            "Content-Type": "application/json",
        },
        "items": [
            {"api_key": "sk-secret-value"},
            {"refresh_token": "refresh-secret"},
            {"password": "pw"},
        ],
        "usage": {"prompt_tokens": 123, "max_tokens": 456},
    }

    redacted = redact_sensitive(original)

    assert original["headers"]["Authorization"] == "Bearer abc123"
    assert redacted["integration"] == "notion"
    assert redacted["headers"]["Authorization"] == "[REDACTED]"
    assert redacted["headers"]["Content-Type"] == "application/json"
    assert redacted["items"][0]["api_key"] == "[REDACTED]"
    assert redacted["items"][1]["refresh_token"] == "[REDACTED]"
    assert redacted["items"][2]["password"] == "[REDACTED]"
    assert redacted["usage"] == {"prompt_tokens": 123, "max_tokens": 456}


def test_redact_sensitive_redacts_exact_secret_values_in_strings() -> None:
    payload = {
        "body": "request failed with token notion-secret-token in response body",
        "nested": ["github-secret-token"],
    }

    redacted = redact_sensitive(
        payload,
        secrets=("notion-secret-token", "github-secret-token"),
    )

    assert "notion-secret-token" not in repr(redacted)
    assert "github-secret-token" not in repr(redacted)
    assert redacted["body"] == "request failed with token [REDACTED] in response body"
    assert redacted["nested"] == ["[REDACTED]"]


def test_redact_text_redacts_key_values_bearer_and_openai_like_tokens() -> None:
    text = (
        "token=abc123 Authorization: Bearer super-secret "
        "api_key=sk-test-secret-123456 password=hunter2"
    )

    redacted = redact_text(text)

    assert "abc123" not in redacted
    assert "super-secret" not in redacted
    assert "sk-test-secret-123456" not in redacted
    assert "hunter2" not in redacted
    assert redacted.count("[REDACTED]") >= 4


def test_secret_values_from_registry_extracts_tokens_only() -> None:
    registry = HostCredentialRegistry(
        credentials={
            "notion": HostCredential(
                name="notion",
                credential_type="bearer",
                token="notion-secret-token",
                allowed_hosts=("api.notion.com",),
                allowed_methods=("GET",),
                allowed_paths=("/v1/pages",),
                default_headers={"Notion-Version": "2026-03-11"},
            )
        }
    )

    assert secret_values_from_registry(registry) == ("notion-secret-token",)
    assert "notion-secret-token" not in repr(registry.safe_metadata())
