"""Shared redaction helpers for persisted and diagnostic surfaces."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTION = "[REDACTED]"

_SENSITIVE_KEYS = {
    "authorization",
    "cookie",
    "set-cookie",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "apikey",
    "api_key",
    "secret",
    "password",
    "token",
}
_KEY_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(authorization)\b(\s*[:=]\s*)bearer\s+[^\s,;]+"),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|token|secret|password|x-api-key)"
        r"\b(\s*[:=]\s*)([^\s,;]+)"
    ),
)
_BEARER_PATTERN = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+")
_SK_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b")


def redact_sensitive(value: Any, *, secrets: Iterable[str] = ()) -> Any:
    """Return a redacted deep copy of value.

    Redaction is structural for mappings/lists and textual for scalar strings. The
    original object is never mutated.
    """
    secret_values = tuple(_normalize_secret_values(secrets))
    return _redact_value(value, secret_values=secret_values)


def redact_text(text: str, *, secrets: Iterable[str] = ()) -> str:
    """Redact secret-bearing substrings from plain text."""
    redacted = text
    for secret in _normalize_secret_values(secrets):
        redacted = redacted.replace(secret, REDACTION)
    for pattern in _KEY_VALUE_PATTERNS:
        redacted = pattern.sub(
            lambda match: f"{match.group(1)}{match.group(2)}{REDACTION}",
            redacted,
        )
    redacted = _BEARER_PATTERN.sub(lambda match: f"{match.group(1)} {REDACTION}", redacted)
    return _SK_PATTERN.sub(REDACTION, redacted)


def secret_values_from_registry(registry: Any) -> tuple[str, ...]:
    """Extract known secret values from a credential registry-like object."""
    raw_credentials = getattr(registry, "credentials", None)
    if not isinstance(raw_credentials, Mapping):
        return ()

    secrets: list[str] = []
    for credential in raw_credentials.values():
        token = getattr(credential, "token", None)
        if isinstance(token, str) and token:
            secrets.append(token)
    return tuple(secrets)


def _redact_value(value: Any, *, secret_values: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _is_sensitive_key(key):
                redacted[key] = REDACTION
            else:
                redacted[key] = _redact_value(item, secret_values=secret_values)
        return redacted
    if isinstance(value, list):
        return [_redact_value(item, secret_values=secret_values) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item, secret_values=secret_values) for item in value)
    if isinstance(value, str):
        return redact_text(value, secrets=secret_values)
    return value


def _is_sensitive_key(key: str) -> bool:
    lowered = key.strip().lower()
    normalized = lowered.replace("_", "-")
    underscore = lowered.replace("-", "_")
    return (
        lowered in _SENSITIVE_KEYS
        or normalized in _SENSITIVE_KEYS
        or underscore in _SENSITIVE_KEYS
        or underscore.endswith("_token")
        or underscore.endswith("_secret")
        or "password" in underscore
        or "api_key" in underscore
        or "apikey" in underscore
    )


def _normalize_secret_values(secrets: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in secrets:
        if not isinstance(item, str):
            continue
        secret = item.strip()
        if not secret or secret in seen:
            continue
        seen.add(secret)
        normalized.append(secret)
    return tuple(normalized)
