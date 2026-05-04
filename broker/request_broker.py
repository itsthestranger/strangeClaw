"""Host-side generic request broker policy engine."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlsplit

import requests
from requests import Response

from broker.credentials import HostCredential, HostCredentialRegistry
from broker.redaction import redact_sensitive, redact_text, secret_values_from_registry

_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_METHODS = {"DELETE", "GET", "PATCH", "POST", "PUT"}
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_PROTECTED_HEADER_NAMES = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "set-cookie",
    "x-api-key",
}
_LOCAL_HOSTS = {"localhost", "localhost.localdomain"}
_METADATA_IPS = {
    ipaddress.ip_address("169.254.169.254"),
    ipaddress.ip_address("fe80::a9fe:a9fe"),
}


@dataclass(frozen=True, slots=True)
class RequestBrokerConfig:
    """Broker-level execution and guardrail config."""

    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 30.0
    max_request_body_chars: int = 100_000
    max_response_body_chars: int = 200_000
    max_redirects: int = 5


@dataclass(frozen=True, slots=True)
class _NormalizedRequest:
    method: str
    url: str
    integration: str | None
    headers: dict[str, str]
    body: str | None


class RequestBroker:
    """Validate and execute generic HTTP requests under host-side policy."""

    def __init__(
        self,
        credential_registry: HostCredentialRegistry,
        config: RequestBrokerConfig | None = None,
    ) -> None:
        self._credential_registry = credential_registry
        self._config = config or RequestBrokerConfig()
        self._known_secret_values = secret_values_from_registry(credential_registry)

    @property
    def known_secret_values(self) -> tuple[str, ...]:
        """Return redaction context derived from loaded host credentials."""
        return self._known_secret_values

    def redact_for_logging(self, value: Any) -> Any:
        """Return a redacted copy safe for logs/journals/diagnostics."""
        return redact_sensitive(value, secrets=self._known_secret_values)

    def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and execute one brokered HTTP request."""
        normalized, error = self._normalize_request(request)
        if error is not None:
            return error
        assert normalized is not None

        credential = self._resolve_credential(normalized.integration)
        if isinstance(credential, dict):
            return credential

        policy_error = self._validate_policy(
            method=normalized.method,
            url=normalized.url,
            credential=credential,
        )
        if policy_error is not None:
            return self._error(
                "policy_denied",
                policy_error,
                integration=normalized.integration,
            )

        if (
            normalized.body is not None
            and len(normalized.body) > self._config.max_request_body_chars
        ):
            return self._error(
                "policy_denied",
                (
                    "Request body exceeds max_request_body_chars "
                    f"({self._config.max_request_body_chars})."
                ),
                integration=normalized.integration,
            )

        prepared_headers, header_error = self._prepare_headers(
            user_headers=normalized.headers,
            credential=credential,
            integration=normalized.integration,
        )
        if header_error is not None:
            return header_error

        method = normalized.method
        url = normalized.url
        body = normalized.body
        redirect_count = 0

        while True:
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=prepared_headers,
                    data=body,
                    timeout=(
                        self._config.connect_timeout_seconds,
                        self._config.read_timeout_seconds,
                    ),
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                return self._error(
                    "request_failed",
                    f"HTTP request failed: {exc}",
                    integration=normalized.integration,
                )

            redirect_target = self._redirect_target(url=url, response=response)
            if redirect_target is None:
                return self._success_response(
                    response=response,
                    integration=normalized.integration,
                )

            redirect_count += 1
            if redirect_count > self._config.max_redirects:
                return self._error(
                    "policy_denied",
                    (
                        "Too many redirects. Maximum allowed redirects: "
                        f"{self._config.max_redirects}."
                    ),
                    integration=normalized.integration,
                )

            redirect_policy_error = self._validate_policy(
                method=method,
                url=redirect_target,
                credential=credential,
            )
            if redirect_policy_error is not None:
                return self._error(
                    "policy_denied",
                    (
                        "Redirect denied by policy: "
                        f"{redirect_policy_error}"
                    ),
                    integration=normalized.integration,
                )

            method, body = self._redirect_method_and_body(
                status_code=int(getattr(response, "status_code", 0)),
                method=method,
                body=body,
            )
            url = redirect_target

    def _normalize_request(
        self,
        request: Mapping[str, Any],
    ) -> tuple[_NormalizedRequest | None, dict[str, Any] | None]:
        if not isinstance(request, Mapping):
            return None, self._error("invalid_request", "Request payload must be an object.")

        method_raw = request.get("method")
        if not isinstance(method_raw, str) or not method_raw.strip():
            return None, self._error("invalid_request", "Request method must be a string.")
        method = method_raw.strip().upper()
        if method not in _ALLOWED_METHODS:
            allowed = ", ".join(sorted(_ALLOWED_METHODS))
            return None, self._error(
                "invalid_request",
                f"Request method must be one of: {allowed}.",
            )

        url_raw = request.get("url")
        if not isinstance(url_raw, str) or not url_raw.strip():
            return None, self._error("invalid_request", "Request url must be a string.")
        url = url_raw.strip()

        integration_raw = request.get("integration")
        integration: str | None
        if integration_raw is None:
            integration = None
        elif isinstance(integration_raw, str) and integration_raw.strip():
            integration = integration_raw.strip()
        else:
            return None, self._error(
                "invalid_request",
                "Request integration must be a non-empty string or null.",
            )

        headers_raw = request.get("headers", {})
        if headers_raw is None:
            headers_raw = {}
        if not isinstance(headers_raw, Mapping):
            return None, self._error("invalid_request", "Request headers must be an object.")
        headers: dict[str, str] = {}
        for key, value in headers_raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return None, self._error(
                    "invalid_request",
                    "Request headers must contain only string keys and values.",
                )
            stripped_key = key.strip()
            if not stripped_key:
                return None, self._error(
                    "invalid_request",
                    "Request header names must be non-empty.",
                )
            headers[stripped_key] = value

        body_raw = request.get("body")
        body: str | None
        if body_raw is None:
            body = None
        elif isinstance(body_raw, str):
            body = body_raw
        else:
            return None, self._error(
                "invalid_request",
                "Request body must be a string or null.",
            )

        return (
            _NormalizedRequest(
                method=method,
                url=url,
                integration=integration,
                headers=headers,
                body=body,
            ),
            None,
        )

    def _resolve_credential(
        self,
        integration: str | None,
    ) -> HostCredential | dict[str, Any] | None:
        if integration is None:
            return None
        credential = self._credential_registry.get(integration)
        if credential is None:
            available = ", ".join(self._credential_registry.names()) or "<none>"
            return self._error(
                "policy_denied",
                (
                    f"Integration '{integration}' is not configured. "
                    f"Available integrations: {available}."
                ),
                integration=integration,
            )
        return credential

    def _validate_policy(
        self,
        *,
        method: str,
        url: str,
        credential: HostCredential | None,
    ) -> str | None:
        split = urlsplit(url)
        scheme = split.scheme.strip().lower()
        if scheme not in _ALLOWED_SCHEMES:
            return "URL scheme must be http or https."

        if split.username is not None or split.password is not None:
            return "URL must not include embedded credentials."

        host = split.hostname
        if host is None:
            return "URL must include a host."
        normalized_host = host.strip().lower()
        if not normalized_host:
            return "URL host must be non-empty."

        forbidden_reason = _forbidden_host_reason(normalized_host)
        if forbidden_reason is not None:
            return forbidden_reason

        path = split.path or "/"

        if credential is None:
            return None

        if normalized_host not in set(credential.allowed_hosts):
            return (
                f"Host '{normalized_host}' is not allowed for integration "
                f"'{credential.name}'."
            )
        if method not in set(credential.allowed_methods):
            return (
                f"Method '{method}' is not allowed for integration "
                f"'{credential.name}'."
            )
        if not _path_allowed(path, credential.allowed_paths):
            return (
                f"Path '{path}' is not allowed for integration "
                f"'{credential.name}'."
            )

        return None

    def _prepare_headers(
        self,
        *,
        user_headers: dict[str, str],
        credential: HostCredential | None,
        integration: str | None,
    ) -> tuple[dict[str, str], dict[str, Any] | None]:
        protected = set(_PROTECTED_HEADER_NAMES)
        integration_injected_headers: dict[str, str] = {}

        if credential is not None:
            for key, value in credential.default_headers.items():
                integration_injected_headers[key] = value
            if credential.credential_type == "bearer":
                integration_injected_headers["Authorization"] = f"Bearer {credential.token}"

        protected.update(key.strip().lower() for key in integration_injected_headers)
        for key in user_headers:
            lowered = key.strip().lower()
            if lowered in protected:
                return {}, self._error(
                    "policy_denied",
                    (
                        f"Request headers must not include protected header '{key}'"
                        + (
                            f" when integration '{integration}' is used."
                            if integration is not None
                            else "."
                        )
                    ),
                    integration=integration,
                )

        merged: dict[str, str] = {}
        lowered_to_key: dict[str, str] = {}

        for key, value in user_headers.items():
            merged[key] = value
            lowered_to_key[key.strip().lower()] = key

        for key, value in integration_injected_headers.items():
            lowered = key.strip().lower()
            if lowered in lowered_to_key:
                continue
            merged[key] = value
            lowered_to_key[lowered] = key

        return merged, None

    def _redirect_target(self, *, url: str, response: Response) -> str | None:
        status_code = int(getattr(response, "status_code", 0))
        if status_code not in _REDIRECT_STATUS_CODES:
            return None

        headers = getattr(response, "headers", {})
        location_value = None
        if isinstance(headers, Mapping):
            raw_location = headers.get("Location")
            if isinstance(raw_location, str):
                location_value = raw_location
        else:
            get_header = getattr(headers, "get", None)
            if callable(get_header):
                raw_location = get_header("Location")
                if isinstance(raw_location, str):
                    location_value = raw_location

        if location_value is None or not location_value.strip():
            return None
        return urljoin(url, location_value.strip())

    def _redirect_method_and_body(
        self,
        *,
        status_code: int,
        method: str,
        body: str | None,
    ) -> tuple[str, str | None]:
        if status_code == 303 and method != "HEAD":
            return "GET", None
        if status_code in {301, 302} and method not in {"GET", "HEAD"}:
            return "GET", None
        return method, body

    def _success_response(
        self,
        *,
        response: Response,
        integration: str | None,
    ) -> dict[str, Any]:
        headers = _headers_to_dict(getattr(response, "headers", {}))

        response_text = getattr(response, "text", "")
        if not isinstance(response_text, str):
            response_text = str(response_text)
        truncated_body, truncated = _truncate_text(
            response_text,
            limit=self._config.max_response_body_chars,
        )

        payload = {
            "success": True,
            "status_code": int(getattr(response, "status_code", 0)),
            "headers": headers,
            "body": truncated_body,
            "truncated": truncated,
            "integration": integration,
        }
        redacted = self.redact_for_logging(payload)
        if isinstance(redacted, dict):
            return redacted
        return {
            "success": True,
            "status_code": int(getattr(response, "status_code", 0)),
            "headers": headers,
            "body": truncated_body,
            "truncated": truncated,
            "integration": integration,
        }

    def _error(
        self,
        error_code: str,
        message: str,
        *,
        integration: str | None = None,
    ) -> dict[str, Any]:
        return {
            "success": False,
            "error_code": error_code,
            "message": redact_text(message, secrets=self._known_secret_values),
            "integration": integration,
        }


def _forbidden_host_reason(host: str) -> str | None:
    lowered = host.strip().lower()
    if lowered in _LOCAL_HOSTS or lowered.endswith(".localhost"):
        return f"Host '{host}' is not allowed by broker policy."

    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        return None

    if ip in _METADATA_IPS:
        return f"Host '{host}' is metadata-service scoped and blocked by broker policy."
    if ip.is_loopback or ip.is_link_local or ip.is_private:
        return f"Host '{host}' is private/loopback/link-local and blocked by broker policy."
    if ip.is_multicast or ip.is_unspecified or ip.is_reserved:
        return f"Host '{host}' is not allowed by broker policy."
    return None


def _path_allowed(path: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.endswith("*"):
            if path.startswith(pattern[:-1]):
                return True
            continue
        if path == pattern:
            return True
    return False


def _truncate_text(text: str, *, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return (
        text[:limit] + f"\n\n[... truncated, original {len(text)} chars ...]",
        True,
    )


def _headers_to_dict(raw_headers: Any) -> dict[str, str]:
    if isinstance(raw_headers, Mapping):
        return {str(key): str(value) for key, value in raw_headers.items()}

    items = getattr(raw_headers, "items", None)
    if callable(items):
        try:
            return {str(key): str(value) for key, value in items()}
        except Exception:
            return {}
    return {}
