"""Request broker policy validation primitives."""

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


@dataclass
class PolicyResult:
    """Result of validating a broker request against policy."""

    allowed: bool
    reason: str | None


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
    """Host-side request broker (validation slice for C5.3)."""

    def __init__(self, credentials: dict[str, Any], config: dict[str, Any]) -> None:
        self._credentials = credentials
        self._config = config
        self._redaction_tokens = _collect_credential_tokens(credentials)
        self._rate_limiters: dict[str, _TokenBucket] = {}
        for integration, policy in credentials.items():
            if not isinstance(policy, dict):
                continue
            rate_limit = policy.get("rate_limit")
            if not isinstance(rate_limit, dict):
                continue
            requests_count = _to_positive_int(rate_limit.get("requests"))
            per_seconds = _to_positive_float(rate_limit.get("per_seconds"))
            if requests_count is None or per_seconds is None:
                continue
            self._rate_limiters[integration] = _TokenBucket(requests_count, per_seconds)

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        if action == "http_request":
            result = self._handle_http_request(payload)
        elif action == "web_fetch":
            result = self._handle_web_fetch(payload)
        elif action == "web_search":
            result = self._handle_web_search(payload)
        elif action == "list_integrations":
            result = self._handle_list_integrations(payload)
        else:
            result = {"success": False, "error": f"unknown action: {action}"}

        redacted = self._redact_value(result)
        if isinstance(redacted, dict):
            return redacted
        return {"success": False, "error": "internal_error", "detail": "invalid broker response"}

    def _validate(
        self,
        policy: dict[str, Any],
        method: str,
        url: str,
        headers: dict[str, str],
    ) -> PolicyResult:
        integration = str(policy.get("name", "<unknown>"))
        method_upper = method.upper()
        allowed_methods = policy.get("allowed_methods", [])
        if method_upper not in allowed_methods:
            return PolicyResult(
                allowed=False,
                reason=(
                    f"method {method_upper} not in allowed_methods {allowed_methods} "
                    f"for integration '{integration}'"
                ),
            )

        parsed = urlparse(url)
        host = parsed.hostname or ""
        allowed_hosts = policy.get("allowed_hosts", [])
        if not any(fnmatch(host, str(pattern)) for pattern in allowed_hosts):
            return PolicyResult(
                allowed=False,
                reason=f"host {host} not in allowed_hosts for integration '{integration}'",
            )

        path = parsed.path or "/"
        allowed_paths = policy.get("allowed_paths", [])
        if not any(fnmatch(path, str(pattern)) for pattern in allowed_paths):
            return PolicyResult(
                allowed=False,
                reason=(
                    f"path {path} not matched by allowed_paths {allowed_paths} "
                    f"for integration '{integration}'"
                ),
            )

        protected_headers = policy.get("protected_headers", [])
        protected_by_lower = {str(name).lower(): str(name) for name in protected_headers}
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
        policy: dict[str, Any],
        headers: dict[str, str],
        url: str,
    ) -> tuple[dict[str, str], str]:
        final_headers: dict[str, str] = {}
        defaults = policy.get("default_headers", {})
        if isinstance(defaults, dict):
            for key, value in defaults.items():
                if isinstance(key, str) and isinstance(value, str):
                    final_headers[key] = value
        for key, value in headers.items():
            final_headers[key] = value

        auth_type = str(policy.get("auth_type", "")).lower()
        integration = str(policy.get("name", "<unknown>"))
        token = str(policy.get("token", ""))
        LOGGER.debug("injecting %s credential for integration %s", auth_type, integration)

        if auth_type == "bearer":
            final_headers["Authorization"] = f"Bearer {token}"
            return final_headers, url

        if auth_type == "header":
            header_name_raw = policy.get("header_name", "Authorization")
            header_name = (
                header_name_raw.strip()
                if isinstance(header_name_raw, str) and header_name_raw.strip()
                else "Authorization"
            )
            final_headers[header_name] = token
            return final_headers, url

        return final_headers, url

    def _ssrf_check(self, url: str) -> PolicyResult:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return PolicyResult(allowed=False, reason="DNS resolution failed for <unknown>")

        try:
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
            if self._is_reserved_ip(ip_obj):
                return PolicyResult(
                    allowed=False,
                    reason=f"SSRF: {hostname} resolves to reserved address {ip_text}",
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
    def _is_reserved_ip(ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        reserved_ranges = (
            ipaddress.ip_network("10.0.0.0/8"),
            ipaddress.ip_network("172.16.0.0/12"),
            ipaddress.ip_network("192.168.0.0/16"),
            ipaddress.ip_network("127.0.0.0/8"),
            ipaddress.ip_network("169.254.0.0/16"),
            ipaddress.ip_network("::1/128"),
            ipaddress.ip_network("fc00::/7"),
        )
        return any(ip_obj in net for net in reserved_ranges)

    def _handle_http_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        integration = payload.get("integration")
        method = str(payload.get("method", "GET")).upper()
        url = str(payload.get("url", ""))
        model_headers = _normalize_headers(payload.get("headers"))
        body_raw = payload.get("body")
        body = body_raw if isinstance(body_raw, str) or body_raw is None else str(body_raw)

        if isinstance(integration, str) and integration:
            policy = self._credentials.get(integration)
            if not isinstance(policy, dict):
                reason = f"integration '{integration}' not found in secrets.yaml"
                return self._deny(integration, method, url, reason)
            integration_name = integration
            use_public_policy = False
        else:
            public_cfg = self._public_policy_config()
            if not bool(public_cfg.get("enabled", True)):
                reason = "public requests disabled; specify an integration name"
                return self._deny("_public", method, url, reason)
            policy = self._build_public_policy(public_cfg)
            integration_name = "_public"
            use_public_policy = True

        policy = dict(policy)
        policy["name"] = integration_name

        validation = self._validate(policy, method, url, model_headers)
        if not validation.allowed:
            return self._deny(integration_name, method, url, validation.reason or "policy denied")

        if not self._consume_rate_limit(integration_name):
            return {
                "success": False,
                "error": "rate_limited",
                "reason": f"rate limit exceeded for integration '{integration_name}'",
            }

        if use_public_policy:
            ssrf = self._ssrf_check(url)
            if not ssrf.allowed:
                return self._deny(integration_name, method, url, ssrf.reason or "SSRF denied")

        final_headers, final_url = self._inject(policy, model_headers, url)
        max_bytes = _to_positive_int(policy.get("max_response_bytes"))
        if max_bytes is None:
            max_bytes = _DEFAULT_PUBLIC_MAX_RESPONSE_BYTES
        if use_public_policy:
            def public_redirect_guard(
                redirect_url: str,
                redirect_method: str,
            ) -> dict[str, Any] | None:
                ssrf = self._ssrf_check(redirect_url)
                if not ssrf.allowed:
                    return self._deny(
                        integration_name,
                        redirect_method,
                        redirect_url,
                        ssrf.reason or "SSRF denied",
                    )
                return None

            redirect_guard = public_redirect_guard
        else:
            def integration_redirect_guard(
                redirect_url: str,
                redirect_method: str,
            ) -> dict[str, Any] | None:
                validation = self._validate(policy, redirect_method, redirect_url, model_headers)
                if not validation.allowed:
                    return self._deny(
                        integration_name,
                        redirect_method,
                        redirect_url,
                        validation.reason or "policy denied",
                    )
                return None

            redirect_guard = integration_redirect_guard

        return self._execute_with_redirects(
            method=method,
            url=final_url,
            headers=final_headers,
            body=body,
            max_bytes=max_bytes,
            redirect_guard=redirect_guard,
        )

    def _handle_web_fetch(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = str(payload.get("url", ""))
        ssrf = self._ssrf_check(url)
        if not ssrf.allowed:
            return self._deny("_web_fetch", "GET", url, ssrf.reason or "SSRF denied")

        max_chars = self._web_fetch_max_chars()

        def web_fetch_redirect_guard(
            redirect_url: str,
            redirect_method: str,
        ) -> dict[str, Any] | None:
            ssrf = self._ssrf_check(redirect_url)
            if not ssrf.allowed:
                return self._deny(
                    "_web_fetch",
                    redirect_method,
                    redirect_url,
                    ssrf.reason or "SSRF denied",
                )
            return None

        execute_result = self._execute_with_redirects(
            method="GET",
            url=url,
            headers={},
            body=None,
            max_bytes=max_chars * _WEB_FETCH_BODY_MULTIPLIER,
            redirect_guard=web_fetch_redirect_guard,
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
        policy = self._credentials.get("_web_search")
        endpoint = self._web_search_endpoint()
        if not isinstance(policy, dict):
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

        policy = dict(policy)
        policy["name"] = "_web_search"
        validation = self._validate(policy, "GET", search_url, {})
        if not validation.allowed:
            return self._deny(
                "_web_search",
                "GET",
                search_url,
                validation.reason or "policy denied",
            )

        headers, final_url = self._inject(policy, {}, search_url)
        max_bytes = _to_positive_int(policy.get("max_response_bytes"))
        if max_bytes is None:
            max_bytes = _DEFAULT_PUBLIC_MAX_RESPONSE_BYTES

        def web_search_redirect_guard(
            redirect_url: str,
            redirect_method: str,
        ) -> dict[str, Any] | None:
            hop_validation = self._validate(policy, redirect_method, redirect_url, {})
            if not hop_validation.allowed:
                return self._deny(
                    "_web_search",
                    redirect_method,
                    redirect_url,
                    hop_validation.reason or "policy denied",
                )
            return None

        execute_result = self._execute_with_redirects(
            method="GET",
            url=final_url,
            headers=headers,
            body=None,
            max_bytes=max_bytes,
            redirect_guard=web_search_redirect_guard,
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
        return {"success": True, "integrations": list_integration_names(self._credentials)}

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

    def _build_public_policy(self, public_cfg: dict[str, Any]) -> dict[str, Any]:
        allowed_methods_raw = public_cfg.get("allowed_methods", ["GET"])
        allowed_methods = ["GET"]
        if isinstance(allowed_methods_raw, list):
            normalized = [
                str(item).upper()
                for item in allowed_methods_raw
                if isinstance(item, str) and item.strip()
            ]
            if normalized:
                allowed_methods = normalized
        max_response_bytes = _to_positive_int(public_cfg.get("max_response_bytes"))
        if max_response_bytes is None:
            max_response_bytes = _DEFAULT_PUBLIC_MAX_RESPONSE_BYTES
        return {
            "name": "_public",
            "auth_type": "none",
            "token": "",
            "allowed_hosts": ["*"],
            "allowed_methods": allowed_methods,
            "allowed_paths": ["/*"],
            "protected_headers": ["Authorization"],
            "default_headers": {},
            "max_response_bytes": max_response_bytes,
        }

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
