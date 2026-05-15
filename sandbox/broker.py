"""Host-side request broker for policy-validated HTTP/search/fetch execution."""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlsplit, urlunsplit

import requests
import trafilatura

from host_secrets import list_integration_names

LOGGER = logging.getLogger(__name__)
_DEFAULT_PUBLIC_MAX_RESPONSE_BYTES = 524288
_DEFAULT_WEB_FETCH_MAX_CHARS = 20000
_WEB_FETCH_BODY_MULTIPLIER = 4
_MAX_REDIRECT_HOPS = 5
_REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
_PUBLIC_ALLOWED_SCHEMES = {"http", "https"}
_Handler = Callable[[dict[str, Any]], dict[str, Any]]
_RedirectGuard = Callable[[str, str], dict[str, Any] | None]


@dataclass
class PolicyResult:
    """Result of validating a broker request against policy."""

    allowed: bool
    reason: str | None


@dataclass(frozen=True)
class Policy:
    """Normalized broker policy record used by all security-sensitive paths."""

    name: str
    auth_type: str
    token: str
    header_name: str
    allowed_hosts: tuple[str, ...]
    allowed_methods: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    protected_headers: tuple[str, ...]
    default_headers: dict[str, str]
    max_response_bytes: int
    rate_limit_requests: int | None
    rate_limit_period_seconds: float | None


class _TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, requests_count: int, per_seconds: float) -> None:
        self._capacity = float(requests_count)
        self._tokens = float(requests_count)
        self._refill_rate = float(requests_count) / float(per_seconds)
        self._last_refill = time.monotonic()

    def consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
            self._last_refill = now
        if self._tokens < 1.0:
            return False
        self._tokens -= 1.0
        return True


class RequestBroker:
    """Host-side broker that validates, executes, and redacts outbound HTTP actions."""

    def __init__(self, credentials: dict[str, Any], config: dict[str, Any]) -> None:
        self._credentials = credentials
        self._policies: dict[str, Policy] = {}
        self._config = config
        self._redaction_tokens = _collect_credential_tokens(credentials)
        self._handlers: dict[str, _Handler] = {
            "http_request": self._handle_http_request,
            "web_fetch": self._handle_web_fetch,
            "web_search": self._handle_web_search,
            "list_integrations": self._handle_list_integrations,
        }
        self._rate_limiters: dict[str, _TokenBucket] = {}
        for integration, policy in credentials.items():
            normalized = _normalize_policy(integration, policy)
            if normalized is None:
                continue
            self._policies[integration] = normalized
            if (
                normalized.rate_limit_requests is None
                or normalized.rate_limit_period_seconds is None
            ):
                continue
            self._rate_limiters[integration] = _TokenBucket(
                normalized.rate_limit_requests,
                normalized.rate_limit_period_seconds,
            )

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        action_raw = payload.get("action")
        action = str(action_raw) if isinstance(action_raw, str) else ""
        handler = self._handlers.get(action)
        if handler is None:
            result = {"success": False, "error": f"unknown action: {action_raw}"}
        else:
            result = handler(payload)

        redacted = self._redact_value(result)
        if isinstance(redacted, dict):
            return redacted
        return {"success": False, "error": "internal_error", "detail": "invalid broker response"}

    def _validate(
        self,
        policy: Policy,
        method: str,
        url: str,
        headers: dict[str, str],
    ) -> PolicyResult:
        integration = policy.name
        method_upper = method.upper()
        if method_upper not in policy.allowed_methods:
            return PolicyResult(
                allowed=False,
                reason=(
                    f"method {method_upper} not in allowed_methods {list(policy.allowed_methods)} "
                    f"for integration '{integration}'"
                ),
            )

        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not any(fnmatch(host, pattern) for pattern in policy.allowed_hosts):
            return PolicyResult(
                allowed=False,
                reason=f"host {host} not in allowed_hosts for integration '{integration}'",
            )

        path = parsed.path or "/"
        if not any(fnmatch(path, pattern) for pattern in policy.allowed_paths):
            return PolicyResult(
                allowed=False,
                reason=(
                    f"path {path} not matched by allowed_paths {list(policy.allowed_paths)} "
                    f"for integration '{integration}'"
                ),
            )

        protected_by_lower = {name.lower(): name for name in policy.protected_headers}
        for header_name in headers:
            canonical = protected_by_lower.get(header_name.lower())
            if canonical is not None:
                return PolicyResult(
                    allowed=False,
                    reason=(
                        f"header '{canonical}' is protected for integration '{integration}'"
                    ),
                )

        return PolicyResult(allowed=True, reason=None)

    def _deny(self, integration: str, method: str, url: str, reason: str) -> dict[str, Any]:
        return {
            "success": False,
            "error": "policy_denied",
            "reason": reason,
            "integration": integration,
            "requested_method": method.upper(),
            "requested_url": url,
        }

    def _inject(
        self,
        policy: Policy,
        headers: dict[str, str],
        url: str,
    ) -> tuple[dict[str, str], str]:
        final_headers: dict[str, str] = {}
        for key, value in policy.default_headers.items():
            final_headers[key] = value
        for key, value in headers.items():
            final_headers[key] = value

        auth_type = policy.auth_type
        integration = policy.name
        token = policy.token
        LOGGER.debug("injecting %s credential for integration %s", auth_type, integration)

        if auth_type == "bearer":
            final_headers["Authorization"] = f"Bearer {token}"
            return final_headers, url

        if auth_type == "header":
            final_headers[policy.header_name] = token
            return final_headers, url

        return final_headers, url

    def _ssrf_check(self, url: str) -> PolicyResult:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in _PUBLIC_ALLOWED_SCHEMES:
            label = scheme or "<none>"
            return PolicyResult(
                allowed=False,
                reason=f"unsupported URL scheme '{label}' for public URL policy",
            )

        hostname = parsed.hostname
        if not hostname:
            return PolicyResult(allowed=False, reason="DNS resolution failed for <unknown>")

        try:
            # DNS rebinding remains possible between resolution and connect; this
            # helper is a best-effort policy gate and not DNS pinning.
            infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return PolicyResult(allowed=False, reason=f"DNS resolution failed for {hostname}")

        for info in infos:
            sockaddr = info[4]
            if not isinstance(sockaddr, tuple) or not sockaddr:
                continue
            ip_text = str(sockaddr[0])
            try:
                ip_obj = ipaddress.ip_address(ip_text)
            except ValueError:
                continue
            normalized_ip = self._normalize_public_ip(ip_obj)
            if self._is_blocked_public_ip(normalized_ip):
                return PolicyResult(
                    allowed=False,
                    reason=f"SSRF: {hostname} resolves to blocked address {normalized_ip}",
                )

        return PolicyResult(allowed=True, reason=None)

    def _execute(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: str | None,
        max_bytes: int,
    ) -> dict[str, Any]:
        session = requests.Session()
        session.trust_env = False
        try:
            response = session.request(
                method=method,
                url=url,
                headers=headers,
                data=body,
                timeout=(10, 30),
                allow_redirects=False,
                stream=True,
            )

            chunks: list[bytes] = []
            total = 0
            truncated = False
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                if total + len(chunk) > max_bytes:
                    keep = max_bytes - total
                    if keep > 0:
                        chunks.append(chunk[:keep])
                        total += keep
                    truncated = True
                    break
                chunks.append(chunk)
                total += len(chunk)

            raw = b"".join(chunks)
            body_text = raw.decode(response.encoding or "utf-8", errors="replace")
            return {
                "success": True,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body_text,
                "truncated": truncated,
            }
        except requests.RequestException as exc:
            return {"success": False, "error": exc.__class__.__name__, "detail": str(exc)}
        finally:
            session.close()

    def _execute_with_redirects(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: str | None,
        max_bytes: int,
        redirect_guard: Callable[[str, str], dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        current_method = method.upper()
        current_url = url
        redirects_followed = 0

        while True:
            result = self._execute(
                current_method,
                current_url,
                headers,
                body,
                max_bytes,
            )
            if result.get("success") is False:
                return result

            status_code = _to_int(result.get("status_code"))
            next_url = _redirect_target_from_result(current_url, status_code, result)
            if next_url is None:
                return result

            if current_method != "GET":
                return {
                    "success": False,
                    "error": "unsupported_redirect",
                    "detail": (
                        "redirect handling for non-GET methods is disabled "
                        f"(method={current_method}, status={status_code})"
                    ),
                }

            redirects_followed += 1
            if redirects_followed > _MAX_REDIRECT_HOPS:
                return {
                    "success": False,
                    "error": "too_many_redirects",
                    "detail": f"redirect limit exceeded ({_MAX_REDIRECT_HOPS}) for {url}",
                }

            if redirect_guard is not None:
                denial = redirect_guard(next_url, current_method)
                if denial is not None:
                    return denial

            current_url = next_url

    @staticmethod
    def _normalize_public_ip(
        ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
            return ip_obj.ipv4_mapped
        return ip_obj

    @staticmethod
    def _is_blocked_public_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return bool(
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_multicast
            or ip_obj.is_unspecified
        )

    def _handle_http_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        integration = payload.get("integration")
        method, url, model_headers, body = _parse_http_request_payload(payload)

        if isinstance(integration, str) and integration:
            policy = self._policies.get(integration)
            if policy is None:
                reason = f"integration '{integration}' not found in secrets.yaml"
                return self._deny(integration, method, url, reason)
            return self._execute_policy_request(
                policy=policy,
                integration_name=integration,
                method=method,
                url=url,
                model_headers=model_headers,
                body=body,
                rate_limit_key=integration,
            )
        else:
            public_cfg = self._public_policy_config()
            if not bool(public_cfg.get("enabled", True)):
                reason = "public requests disabled; specify an integration name"
                return self._deny("_public", method, url, reason)
            policy = self._build_public_policy(public_cfg)
            return self._execute_public_request(
                policy=policy,
                integration_name="_public",
                method=method,
                url=url,
                model_headers=model_headers,
                body=body,
                max_bytes=policy.max_response_bytes,
                enforce_validation=True,
                enforce_initial_ssrf=True,
                enforce_redirect_ssrf=True,
            )

    def _execute_policy_request(
        self,
        *,
        policy: Policy,
        integration_name: str,
        method: str,
        url: str,
        model_headers: dict[str, str],
        body: str | None,
        rate_limit_key: str | None = None,
    ) -> dict[str, Any]:
        validation_denial = self._validate_or_deny(
            policy=policy,
            integration_name=integration_name,
            method=method,
            url=url,
            model_headers=model_headers,
        )
        if validation_denial is not None:
            return validation_denial

        if rate_limit_key is not None and not self._consume_rate_limit(rate_limit_key):
            return {
                "success": False,
                "error": "rate_limited",
                "reason": f"rate limit exceeded for integration '{integration_name}'",
            }

        final_headers, final_url = self._inject(policy, model_headers, url)
        redirect_guard = self._make_policy_redirect_guard(
            policy=policy,
            integration_name=integration_name,
            model_headers=model_headers,
        )

        return self._execute_with_redirects(
            method=method,
            url=final_url,
            headers=final_headers,
            body=body,
            max_bytes=policy.max_response_bytes,
            redirect_guard=redirect_guard,
        )

    def _execute_public_request(
        self,
        *,
        policy: Policy,
        integration_name: str,
        method: str,
        url: str,
        model_headers: dict[str, str],
        body: str | None,
        max_bytes: int,
        enforce_validation: bool,
        enforce_initial_ssrf: bool,
        enforce_redirect_ssrf: bool,
    ) -> dict[str, Any]:
        if enforce_validation:
            validation_denial = self._validate_or_deny(
                policy=policy,
                integration_name=integration_name,
                method=method,
                url=url,
                model_headers=model_headers,
            )
            if validation_denial is not None:
                return validation_denial

        if enforce_initial_ssrf:
            ssrf_denial = self._ssrf_or_deny(
                integration_name=integration_name,
                method=method,
                url=url,
            )
            if ssrf_denial is not None:
                return ssrf_denial

        final_headers, final_url = self._inject(policy, model_headers, url)
        redirect_guard = self._make_public_redirect_guard(
            policy=policy,
            integration_name=integration_name,
            model_headers=model_headers,
            enforce_validation=enforce_validation,
            enforce_redirect_ssrf=enforce_redirect_ssrf,
        )

        return self._execute_with_redirects(
            method=method,
            url=final_url,
            headers=final_headers,
            body=body,
            max_bytes=max_bytes,
            redirect_guard=redirect_guard,
        )

    def _validate_or_deny(
        self,
        *,
        policy: Policy,
        integration_name: str,
        method: str,
        url: str,
        model_headers: dict[str, str],
    ) -> dict[str, Any] | None:
        validation = self._validate(policy, method, url, model_headers)
        if validation.allowed:
            return None
        return self._deny(integration_name, method, url, validation.reason or "policy denied")

    def _ssrf_or_deny(
        self,
        *,
        integration_name: str,
        method: str,
        url: str,
    ) -> dict[str, Any] | None:
        ssrf = self._ssrf_check(url)
        if ssrf.allowed:
            return None
        return self._deny(integration_name, method, url, ssrf.reason or "SSRF denied")

    def _make_policy_redirect_guard(
        self,
        *,
        policy: Policy,
        integration_name: str,
        model_headers: dict[str, str],
    ) -> _RedirectGuard:
        def guard(redirect_url: str, redirect_method: str) -> dict[str, Any] | None:
            return self._validate_or_deny(
                policy=policy,
                integration_name=integration_name,
                method=redirect_method,
                url=redirect_url,
                model_headers=model_headers,
            )

        return guard

    def _make_public_redirect_guard(
        self,
        *,
        policy: Policy,
        integration_name: str,
        model_headers: dict[str, str],
        enforce_validation: bool,
        enforce_redirect_ssrf: bool,
    ) -> _RedirectGuard:
        def guard(redirect_url: str, redirect_method: str) -> dict[str, Any] | None:
            if enforce_validation:
                validation_denial = self._validate_or_deny(
                    policy=policy,
                    integration_name=integration_name,
                    method=redirect_method,
                    url=redirect_url,
                    model_headers=model_headers,
                )
                if validation_denial is not None:
                    return validation_denial
            if enforce_redirect_ssrf:
                return self._ssrf_or_deny(
                    integration_name=integration_name,
                    method=redirect_method,
                    url=redirect_url,
                )
            return None

        return guard

    def _handle_web_fetch(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = str(payload.get("url", ""))
        max_chars = self._web_fetch_max_chars()
        execute_result = self._execute_public_request(
            policy=self._build_public_policy({}),
            integration_name="_web_fetch",
            method="GET",
            url=url,
            model_headers={},
            body=None,
            max_bytes=max_chars * _WEB_FETCH_BODY_MULTIPLIER,
            enforce_validation=False,
            enforce_initial_ssrf=True,
            enforce_redirect_ssrf=True,
        )
        if execute_result.get("success") is False:
            return execute_result

        headers = execute_result.get("headers")
        content_type = _normalize_content_type(
            str(headers.get("Content-Type", "")) if isinstance(headers, dict) else ""
        )
        body_text = str(execute_result.get("body", ""))
        body_bytes = body_text.encode("utf-8", errors="replace")
        status_code = int(execute_result.get("status_code", 0))

        title: str | None = None
        if content_type == "text/html":
            extracted = trafilatura.extract(body_text, include_links=True, include_tables=True)
            metadata = trafilatura.extract_metadata(body_text)
            maybe_title_raw: Any = None
            if metadata is not None:
                try:
                    maybe_title_raw = metadata.title
                except AttributeError:
                    maybe_title_raw = None
            if isinstance(maybe_title_raw, str):
                maybe_title = maybe_title_raw.strip()
                if maybe_title:
                    title = maybe_title
            text = extracted if isinstance(extracted, str) and extracted.strip() else body_text
        elif content_type in {"text/plain", "application/json", "text/xml", "application/xml"}:
            text = body_text
        elif content_type == "application/pdf":
            text = (
                f"PDF document, {len(body_bytes)} bytes. "
                "Use the shell tool with pdftotext to extract content."
            )
        else:
            display_type = content_type or "application/octet-stream"
            text = f"Binary content ({display_type}), {len(body_bytes)} bytes. No text extracted."

        truncated_text, text_truncated = _truncate_text(text, max_chars)
        return {
            "success": True,
            "url": url,
            "status_code": status_code,
            "content_type": content_type,
            "title": title,
            "text": truncated_text,
            "truncated": bool(execute_result.get("truncated", False) or text_truncated),
        }

    def _handle_web_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        policy = self._policies.get("_web_search")
        endpoint = self._web_search_endpoint()
        if policy is None:
            reason = "web_search integration not configured in secrets.yaml"
            return self._deny("_web_search", "GET", endpoint, reason)

        query = str(payload.get("query", "")).strip()
        if not query:
            return {"success": False, "error": "invalid_request", "reason": "query is required"}

        max_results = _to_positive_int(payload.get("max_results"))
        if max_results is None:
            max_results = self._web_search_max_results()

        search_format = self._web_search_format()
        if search_format == "brave":
            query_params = {"q": query}
        else:
            query_params = {"q": query, "format": "json"}
        search_url = _append_query_params(endpoint, query_params)
        execute_result = self._execute_policy_request(
            policy=policy,
            integration_name="_web_search",
            method="GET",
            url=search_url,
            model_headers={},
            body=None,
        )
        if execute_result.get("success") is False:
            return execute_result

        body_raw = execute_result.get("body", "")
        try:
            payload_json = json.loads(str(body_raw))
        except json.JSONDecodeError as exc:
            return {"success": False, "error": "invalid_json", "detail": str(exc)}
        if not isinstance(payload_json, dict):
            return {"success": False, "error": "invalid_json", "detail": "expected object payload"}

        if search_format == "brave":
            results = _normalize_brave_results(payload_json, max_results)
        else:
            results = _normalize_searxng_results(payload_json, max_results)
        return {"success": True, "results": results}

    def _handle_list_integrations(self, payload: dict[str, Any]) -> dict[str, Any]:
        _ = payload
        return {"success": True, "integrations": list_integration_names(self._policies)}

    def _redact_value(self, value: Any) -> Any:
        if not self._redaction_tokens:
            return value
        if isinstance(value, str):
            return _redact_string(value, self._redaction_tokens)
        if isinstance(value, dict):
            redacted: dict[Any, Any] = {}
            for key, item in value.items():
                redacted_key = (
                    _redact_string(key, self._redaction_tokens) if isinstance(key, str) else key
                )
                redacted[redacted_key] = self._redact_value(item)
            return redacted
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_value(item) for item in value)
        return value

    def _consume_rate_limit(self, integration: str) -> bool:
        limiter = self._rate_limiters.get(integration)
        if limiter is None:
            return True
        return limiter.consume()

    def _public_policy_config(self) -> dict[str, Any]:
        broker_section = self._config.get("broker")
        if not isinstance(broker_section, dict):
            return {}
        public_policy = broker_section.get("public_policy")
        if not isinstance(public_policy, dict):
            return {}
        return public_policy

    def _build_public_policy(self, public_cfg: dict[str, Any]) -> Policy:
        allowed_methods_raw = public_cfg.get("allowed_methods", ["GET"])
        allowed_methods: tuple[str, ...] = ("GET",)
        if isinstance(allowed_methods_raw, list):
            normalized = [
                str(item).upper()
                for item in allowed_methods_raw
                if isinstance(item, str) and item.strip()
            ]
            if normalized:
                allowed_methods = tuple(normalized)
        max_response_bytes = _to_positive_int(public_cfg.get("max_response_bytes"))
        if max_response_bytes is None:
            max_response_bytes = _DEFAULT_PUBLIC_MAX_RESPONSE_BYTES
        return Policy(
            name="_public",
            auth_type="none",
            token="",
            header_name="Authorization",
            allowed_hosts=("*",),
            allowed_methods=allowed_methods,
            allowed_paths=("/*",),
            protected_headers=("Authorization",),
            default_headers={},
            max_response_bytes=max_response_bytes,
            rate_limit_requests=None,
            rate_limit_period_seconds=None,
        )

    def _web_fetch_max_chars(self) -> int:
        section = self._config.get("web_fetch")
        if not isinstance(section, dict):
            return _DEFAULT_WEB_FETCH_MAX_CHARS
        value = _to_positive_int(section.get("max_chars"))
        if value is None:
            return _DEFAULT_WEB_FETCH_MAX_CHARS
        return value

    def _web_search_endpoint(self) -> str:
        section = self._config.get("web_search")
        if not isinstance(section, dict):
            return ""
        endpoint = section.get("endpoint")
        return endpoint.strip() if isinstance(endpoint, str) else ""

    def _web_search_format(self) -> str:
        section = self._config.get("web_search")
        if not isinstance(section, dict):
            return "brave"
        raw = section.get("format", "brave")
        fmt = str(raw).strip().lower()
        return "brave" if fmt != "searxng" else "searxng"

    def _web_search_max_results(self) -> int:
        section = self._config.get("web_search")
        if not isinstance(section, dict):
            return 10
        value = _to_positive_int(section.get("max_results"))
        return value if value is not None else 10


def _collect_credential_tokens(credentials: dict[str, Any]) -> tuple[str, ...]:
    tokens: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "token" and isinstance(item, str) and item:
                    tokens.add(item)
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(credentials)
    return tuple(sorted(tokens, key=len, reverse=True))


def _normalize_policy(name: str, raw_policy: Any) -> Policy | None:
    if not isinstance(raw_policy, dict):
        return None

    auth_type_raw = raw_policy.get("auth_type")
    if not isinstance(auth_type_raw, str):
        return None
    auth_type = auth_type_raw.strip().lower()
    if auth_type not in {"bearer", "header"}:
        return None

    token_raw = raw_policy.get("token")
    if not isinstance(token_raw, str):
        return None

    allowed_hosts = _normalize_string_tuple(raw_policy.get("allowed_hosts"))
    allowed_methods = _normalize_string_tuple(raw_policy.get("allowed_methods"))
    allowed_paths = _normalize_string_tuple(raw_policy.get("allowed_paths"))
    protected_headers = _normalize_string_tuple(raw_policy.get("protected_headers"))
    if (
        allowed_hosts is None
        or allowed_methods is None
        or allowed_paths is None
        or protected_headers is None
    ):
        return None

    normalized_methods = tuple(item.upper() for item in allowed_methods)

    default_headers_raw = raw_policy.get("default_headers")
    if not isinstance(default_headers_raw, dict):
        return None
    default_headers: dict[str, str] = {}
    for key, value in default_headers_raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            return None
        default_headers[key] = value

    max_response_bytes = _to_positive_int(raw_policy.get("max_response_bytes"))
    if max_response_bytes is None:
        return None

    rate_limit = raw_policy.get("rate_limit")
    rate_limit_requests: int | None = None
    rate_limit_period_seconds: float | None = None
    if isinstance(rate_limit, dict):
        rate_limit_requests = _to_positive_int(rate_limit.get("requests"))
        rate_limit_period_seconds = _to_positive_float(rate_limit.get("per_seconds"))

    header_name = "Authorization"
    if auth_type == "header":
        header_name_raw = raw_policy.get("header_name", "Authorization")
        if not isinstance(header_name_raw, str) or not header_name_raw.strip():
            return None
        header_name = header_name_raw.strip()

    return Policy(
        name=name,
        auth_type=auth_type,
        token=token_raw,
        header_name=header_name,
        allowed_hosts=allowed_hosts,
        allowed_methods=normalized_methods,
        allowed_paths=allowed_paths,
        protected_headers=protected_headers,
        default_headers=default_headers,
        max_response_bytes=max_response_bytes,
        rate_limit_requests=rate_limit_requests,
        rate_limit_period_seconds=rate_limit_period_seconds,
    )


def _normalize_string_tuple(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value:
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        items.append(item.strip())
    return tuple(items)


def _redact_string(value: str, tokens: tuple[str, ...]) -> str:
    redacted = value
    for token in tokens:
        redacted = redacted.replace(token, "[REDACTED]")
    return redacted


def _to_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def _to_positive_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0.0:
        return None
    return parsed


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_http_request_payload(
    payload: dict[str, Any],
) -> tuple[str, str, dict[str, str], str | None]:
    method = str(payload.get("method", "GET")).upper()
    url = str(payload.get("url", ""))
    model_headers = _normalize_headers(payload.get("headers"))
    body_raw = payload.get("body")
    body = body_raw if isinstance(body_raw, str) or body_raw is None else str(body_raw)
    return method, url, model_headers, body


def _normalize_headers(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            normalized[key] = item
    return normalized


def _normalize_content_type(raw: str) -> str:
    lowered = raw.lower().strip()
    if ";" in lowered:
        lowered = lowered.split(";", 1)[0].strip()
    return lowered


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    notice = f"\n\n[... truncated, original {len(text)} chars ...]"
    return text[:limit] + notice, True


def _normalize_brave_results(payload: dict[str, Any], max_results: int) -> list[dict[str, str]]:
    web_obj = payload.get("web")
    if not isinstance(web_obj, dict):
        return []
    raw_results = web_obj.get("results")
    if not isinstance(raw_results, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("description", "")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        normalized.append({"title": title, "url": url, "snippet": str(snippet)})
        if len(normalized) >= max_results:
            break
    return normalized


def _normalize_searxng_results(payload: dict[str, Any], max_results: int) -> list[dict[str, str]]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("content", "")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        normalized.append({"title": title, "url": url, "snippet": str(snippet)})
        if len(normalized) >= max_results:
            break
    return normalized


def _append_query_params(url: str, params: dict[str, str]) -> str:
    split = urlsplit(url)
    existing = parse_qsl(split.query, keep_blank_values=True)
    for key, value in params.items():
        existing.append((key, value))
    query = urlencode(existing)
    return urlunsplit((split.scheme, split.netloc, split.path, query, split.fragment))


def _redirect_target_from_result(
    base_url: str,
    status_code: int | None,
    result: dict[str, Any],
) -> str | None:
    if status_code not in _REDIRECT_STATUS_CODES:
        return None
    headers = result.get("headers")
    if not isinstance(headers, dict):
        return None
    location_raw = headers.get("Location")
    if not isinstance(location_raw, str):
        return None
    location = location_raw.strip()
    if not location:
        return None
    return urljoin(base_url, location)
