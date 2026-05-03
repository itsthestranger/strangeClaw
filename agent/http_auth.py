"""Generic integration auth for brokerless HTTP requests."""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

_INTEGRATION_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_AUTH_TYPES = {"bearer", "header", "basic", "none"}
_PROTECTED_HEADER_NAMES = {"authorization", "cookie"}


@dataclass(frozen=True, slots=True)
class HttpIntegration:
    """Configured HTTP integration metadata and secret material."""

    name: str
    auth_type: str
    token: str
    default_headers: dict[str, str]
    header: str | None = None
    prefix: str = ""
    username: str | None = None


class HttpAuthResolver:
    """Resolve named HTTP integrations into request headers."""

    def __init__(self, integrations: dict[str, HttpIntegration]) -> None:
        self._integrations = dict(integrations)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HttpAuthResolver:
        raw_integrations = config.get("integrations", {})
        if not isinstance(raw_integrations, dict):
            return cls({})

        integrations: dict[str, HttpIntegration] = {}
        for name, raw in raw_integrations.items():
            if not isinstance(name, str) or not _INTEGRATION_NAME_RE.fullmatch(name):
                continue
            if not isinstance(raw, dict):
                continue
            integration = _integration_from_mapping(name, raw)
            if integration is not None:
                integrations[name] = integration
        return cls(integrations)

    def available_names(self) -> list[str]:
        """Return integrations that can currently be used by the model."""
        names: list[str] = []
        for name, integration in sorted(self._integrations.items()):
            if integration.auth_type == "none" or integration.token:
                names.append(name)
        return names

    def apply(
        self,
        *,
        integration_name: str | None,
        headers: dict[str, str],
    ) -> tuple[dict[str, str], str | None]:
        """Apply one integration to request headers.

        Returns a new headers dict and an optional error string. The original headers
        mapping is not modified.
        """
        if integration_name is None:
            return dict(headers), None
        if not integration_name.strip():
            return dict(headers), "http_request.integration must be a non-empty string or null."

        name = integration_name.strip()
        integration = self._integrations.get(name)
        if integration is None:
            available = ", ".join(self.available_names()) or "<none>"
            return (
                dict(headers),
                f"http_request integration '{name}' is not configured. "
                f"Available integrations: {available}.",
            )
        if integration.auth_type != "none" and not integration.token:
            return (
                dict(headers),
                f"http_request integration '{name}' has no token configured.",
            )

        normalized_user_headers = {_header_key(key): key for key in headers}
        protected_headers = _protected_headers_for(integration)
        for lowered in protected_headers:
            if lowered in normalized_user_headers:
                original = normalized_user_headers[lowered]
                return (
                    dict(headers),
                    f"http_request.headers must not include '{original}' when "
                    f"integration '{name}' is used.",
                )

        merged = dict(headers)
        for key, value in integration.default_headers.items():
            merged.setdefault(key, value)
        auth_header = _auth_header(integration)
        if auth_header is not None:
            key, value = auth_header
            merged[key] = value
        return merged, None


def validate_integrations_config(raw: Any) -> dict[str, dict[str, Any]]:
    """Validate and normalize the public config representation."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Config field integrations must be a mapping.")

    normalized: dict[str, dict[str, Any]] = {}
    for name, integration in raw.items():
        if not isinstance(name, str) or not _INTEGRATION_NAME_RE.fullmatch(name):
            raise ValueError(
                "Config field integrations keys must match ^[A-Za-z0-9_-]{1,64}$."
            )
        if not isinstance(integration, dict):
            raise ValueError(f"Config field integrations.{name} must be a mapping.")
        normalized[name] = _normalize_integration_config(name, integration)
    return normalized


def _normalize_integration_config(name: str, integration: dict[str, Any]) -> dict[str, Any]:
    auth = integration.get("auth", {"type": "bearer"})
    if auth is None:
        auth = {"type": "none"}
    if not isinstance(auth, dict):
        raise ValueError(f"Config field integrations.{name}.auth must be a mapping.")

    auth_type = auth.get("type", "bearer")
    if not isinstance(auth_type, str):
        raise ValueError(f"Config field integrations.{name}.auth.type must be a string.")
    auth_type = auth_type.strip().lower()
    if auth_type not in _AUTH_TYPES:
        allowed = ", ".join(sorted(_AUTH_TYPES))
        raise ValueError(
            f"Config field integrations.{name}.auth.type must be one of: {allowed}."
        )

    token = integration.get("token", "")
    if not isinstance(token, str):
        raise ValueError(f"Config field integrations.{name}.token must be a string.")

    normalized_auth: dict[str, str] = {"type": auth_type}
    if auth_type == "header":
        header = auth.get("header")
        if not isinstance(header, str) or not _valid_header_name(header):
            raise ValueError(
                f"Config field integrations.{name}.auth.header must be a valid header name."
            )
        normalized_auth["header"] = header.strip()
        prefix = auth.get("prefix", "")
        if not isinstance(prefix, str):
            raise ValueError(f"Config field integrations.{name}.auth.prefix must be a string.")
        normalized_auth["prefix"] = prefix
    elif auth_type == "basic":
        username = auth.get("username")
        if not isinstance(username, str) or not username.strip():
            raise ValueError(
                f"Config field integrations.{name}.auth.username must be a non-empty string."
            )
        normalized_auth["username"] = username.strip()

    default_headers = integration.get("default_headers", {})
    if default_headers is None:
        default_headers = {}
    if not isinstance(default_headers, dict):
        raise ValueError(
            f"Config field integrations.{name}.default_headers must be a mapping."
        )
    normalized_headers: dict[str, str] = {}
    for key, value in default_headers.items():
        if not isinstance(key, str) or not _valid_header_name(key):
            raise ValueError(
                f"Config field integrations.{name}.default_headers keys must be valid header names."
            )
        if _header_key(key) in _PROTECTED_HEADER_NAMES:
            raise ValueError(
                f"Config field integrations.{name}.default_headers must not include "
                f"protected header '{key}'."
            )
        if not isinstance(value, str):
            raise ValueError(
                f"Config field integrations.{name}.default_headers.{key} must be a string."
            )
        normalized_headers[key.strip()] = value

    return {
        "token": token.strip(),
        "auth": normalized_auth,
        "default_headers": normalized_headers,
    }


def _integration_from_mapping(name: str, raw: dict[str, Any]) -> HttpIntegration | None:
    try:
        normalized = _normalize_integration_config(name, raw)
    except ValueError:
        return None

    auth = normalized["auth"]
    return HttpIntegration(
        name=name,
        auth_type=auth["type"],
        token=normalized["token"],
        default_headers=normalized["default_headers"],
        header=auth.get("header"),
        prefix=auth.get("prefix", ""),
        username=auth.get("username"),
    )


def _auth_header(integration: HttpIntegration) -> tuple[str, str] | None:
    if integration.auth_type == "none":
        return None
    if integration.auth_type == "bearer":
        return "Authorization", f"Bearer {integration.token}"
    if integration.auth_type == "header":
        if integration.header is None:
            return None
        return integration.header, f"{integration.prefix}{integration.token}"
    if integration.auth_type == "basic":
        if integration.username is None:
            return None
        raw = f"{integration.username}:{integration.token}".encode()
        encoded = base64.b64encode(raw).decode("ascii")
        return "Authorization", f"Basic {encoded}"
    return None


def _protected_headers_for(integration: HttpIntegration) -> set[str]:
    protected = set(_PROTECTED_HEADER_NAMES)
    auth_header = _auth_header(integration)
    if auth_header is not None:
        protected.add(_header_key(auth_header[0]))
    return protected


def _header_key(value: str) -> str:
    return value.strip().lower()


def _valid_header_name(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and all(char.isalnum() or char in "!#$%&'*+-.^_`|~" for char in stripped)
