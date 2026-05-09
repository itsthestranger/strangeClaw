"""Host-side request broker for sandbox HTTP tools."""

from __future__ import annotations

import fnmatch
import ipaddress
import socket
import time
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
import trafilatura

_DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
_DEFAULT_READ_TIMEOUT_SECONDS = 30.0
_DEFAULT_WEB_FETCH_MAX_CHARS = 20000
_DEFAULT_WEB_SEARCH_MAX_RESULTS = 10
_DEFAULT_PUBLIC_MAX_RESPONSE_BYTES = 524288
_DEFAULT_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_WEB_SEARCH_FORMAT = "brave"
_DEFAULT_WEB_SEARCH_USER_AGENT = "strangeclaw/0.1 (+broker-search)"
_DEFAULT_WEB_FETCH_USER_AGENT = "strangeclaw/0.1 (+broker-fetch)"
_RESERVED_V4_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
]
_RESERVED_V6_NETWORKS = [
    ipaddress.ip_network("::1/128"),
]


@dataclass(frozen=True)
class PolicyResult:
    allowed: bool
    reason: str | None = None


class RequestBroker:
    """Policy-enforced host-side HTTP executor for tool requests."""

    def __init__(
        self,
        *,
        credentials: dict[str, dict[str, Any]],
        public_policy: dict[str, Any] | None,
        web_search_config: dict[str, Any] | None = None,
    ) -> None:
        self._credentials = {name: dict(value) for name, value in credentials.items()}
        self._public_policy = self._normalize_public_policy(public_policy)
        self._web_search_config = self._normalize_web_search_config(web_search_config)
        self._rate_limit_state: dict[str, dict[str, float]] = {}
        self._http = requests.Session()

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        if not isinstance(action, str) or not action.strip():
            return {
                "success": False,
                "error": "invalid_request",
                "reason": "missing or invalid action field",
            }

        if action == "http_request":
            return self._handle_http_request(payload)
        if action == "web_fetch":
            return self._handle_web_fetch(payload)
        if action == "web_search":
            return self._handle_web_search(payload)
        return {
            "success": False,
            "error": "invalid_request",
            "reason": f"unsupported broker action '{action}'",
        }

    def _handle_http_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        integration_name = payload.get("integration")
        method, method_error = self._normalize_method(payload.get("method"))
        if method_error is not None:
            return self._invalid_request(method_error)
        url, url_error = self._normalize_url(payload.get("url"))
        if url_error is not None:
            return self._invalid_request(url_error)
        headers, headers_error = self._normalize_headers(payload.get("headers", {}))
        if headers_error is not None:
            return self._invalid_request(headers_error)
        body_raw = payload.get("body")
        if body_raw is not None and not isinstance(body_raw, str):
            return self._invalid_request("http_request.body must be a string or null")

        if integration_name is None:
            if not self._public_policy["enabled"]:
                return self._policy_denied(
                    reason="public requests are disabled; specify an integration name",
                    integration="_public",
                    method=method,
                    url=url,
                )
            return self._execute_with_policy(
                policy=self._public_policy,
                integration_name="_public",
                method=method,
                url=url,
                headers=headers,
                body=body_raw,
                ssrf_protected=True,
            )

        if not isinstance(integration_name, str) or not integration_name.strip():
            return self._invalid_request("http_request.integration must be a string when provided")
        integration_key = integration_name.strip()
        policy = self._credentials.get(integration_key)
        if policy is None:
            return self._policy_denied(
                reason=f"integration '{integration_key}' is not configured",
                integration=integration_key,
                method=method,
                url=url,
            )

        return self._execute_with_policy(
            policy=policy,
            integration_name=integration_key,
            method=method,
            url=url,
            headers=headers,
            body=body_raw,
            ssrf_protected=False,
        )

    def _handle_web_fetch(self, payload: dict[str, Any]) -> dict[str, Any]:
        url, url_error = self._normalize_url(payload.get("url"))
        if url_error is not None:
            return self._invalid_request(url_error)

        ssrf_check = self._validate_public_url(url)
        if not ssrf_check.allowed:
            return self._policy_denied(
                reason=ssrf_check.reason or "public URL denied",
                integration="_public",
                method="GET",
                url=url,
            )

        execution = self._execute(
            method="GET",
            url=url,
            headers={"User-Agent": _DEFAULT_WEB_FETCH_USER_AGENT},
            body=None,
            max_bytes=int(self._public_policy["max_response_bytes"]),
        )
        if execution.get("success") is not True:
            return execution

        status_code = int(execution.get("status_code", 0))
        content_type = str(execution.get("content_type", ""))
        body_text = str(execution.get("body", ""))
        truncated_by_bytes = bool(execution.get("truncated", False))

        title: str | None = None
        extracted_text: str
        normalized_content_type = self._normalize_content_type(content_type)
        if normalized_content_type == "text/html":
            extracted = trafilatura.extract(body_text, include_links=True, include_tables=True)
            metadata = trafilatura.extract_metadata(body_text)
            maybe_title = getattr(metadata, "title", None) if metadata is not None else None
            if isinstance(maybe_title, str) and maybe_title.strip():
                title = maybe_title.strip()
            if isinstance(extracted, str) and extracted.strip():
                extracted_text = extracted
            else:
                extracted_text = body_text
        elif normalized_content_type in {
            "text/plain",
            "application/json",
            "text/xml",
            "application/xml",
        }:
            extracted_text = body_text
        elif normalized_content_type == "application/pdf":
            byte_len = int(execution.get("raw_bytes_len", 0))
            extracted_text = (
                f"PDF document, {byte_len} bytes. "
                "Use shell tool with pdftotext to extract content."
            )
        else:
            byte_len = int(execution.get("raw_bytes_len", 0))
            display_type = normalized_content_type or "application/octet-stream"
            extracted_text = (
                f"Binary content ({display_type}), {byte_len} bytes. "
                "No text extracted."
            )

        original_length = len(extracted_text)
        truncated_text, truncated_by_chars = _truncate_text(
            extracted_text,
            _DEFAULT_WEB_FETCH_MAX_CHARS,
        )
        truncated = truncated_by_bytes or truncated_by_chars
        if truncated_by_bytes:
            truncated_text = (
                f"{truncated_text}\n\n"
                "[... response body capped at "
                f"{int(self._public_policy['max_response_bytes'])} bytes ...]"
            )

        return {
            "success": True,
            "url": url,
            "status_code": status_code,
            "content_type": normalized_content_type,
            "title": title,
            "text": truncated_text,
            "truncated": truncated,
            "original_length": original_length,
        }

    def _handle_web_search(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query")
        if not isinstance(query, str) or not query.strip():
            return self._invalid_request("web_search.query must be a non-empty string")

        max_results_raw = payload.get("max_results", self._web_search_config["max_results"])
        if isinstance(max_results_raw, bool):
            return self._invalid_request("web_search.max_results must be a positive integer")
        try:
            max_results = int(max_results_raw)
        except (TypeError, ValueError):
            return self._invalid_request("web_search.max_results must be a positive integer")
        if max_results <= 0:
            return self._invalid_request("web_search.max_results must be a positive integer")

        integration = self._credentials.get("_web_search")
        if integration is None:
            return self._policy_denied(
                reason="integration '_web_search' is not configured",
                integration="_web_search",
                method="GET",
                url=self._web_search_config["endpoint"],
            )

        policy_check = self._validate(
            policy=integration,
            method="GET",
            url=self._web_search_config["endpoint"],
            headers={},
            ssrf_protected=False,
        )
        if not policy_check.allowed:
            return self._policy_denied(
                reason=policy_check.reason or "web_search policy denied",
                integration="_web_search",
                method="GET",
                url=self._web_search_config["endpoint"],
            )

        rate_check = self._check_rate_limit("_web_search", integration)
        if not rate_check.allowed:
            return {
                "success": False,
                "error": "rate_limited",
                "reason": rate_check.reason,
                "integration": "_web_search",
                "requested_method": "GET",
                "requested_url": self._web_search_config["endpoint"],
            }

        request_headers: dict[str, str] = {"User-Agent": _DEFAULT_WEB_SEARCH_USER_AGENT}
        endpoint = self._web_search_config["endpoint"]
        fmt = self._web_search_config["format"]
        if fmt == "brave":
            request_headers, endpoint, inject_error = self._inject(
                policy=integration,
                headers=request_headers,
                url=endpoint,
            )
            if inject_error is not None:
                return self._invalid_request(inject_error)
            params = {"q": query.strip()}
        else:
            params = {"q": query.strip(), "format": "json"}

        try:
            response = self._http.get(
                endpoint,
                headers=request_headers,
                params=params,
                timeout=(_DEFAULT_CONNECT_TIMEOUT_SECONDS, _DEFAULT_READ_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            payload_json = response.json()
        except requests.RequestException as exc:
            return {
                "success": False,
                "error": "request_failed",
                "detail": f"web_search request failed: {exc}",
            }
        except ValueError as exc:
            return {
                "success": False,
                "error": "invalid_response",
                "detail": f"web_search returned invalid JSON: {exc}",
            }

        if not isinstance(payload_json, dict):
            return {
                "success": False,
                "error": "invalid_response",
                "detail": "web_search response must be a JSON object",
            }

        if fmt == "brave":
            results = _normalize_brave_results(payload_json, max_results=max_results)
        else:
            results = _normalize_searxng_results(payload_json, max_results=max_results)

        return {
            "success": True,
            "query": query.strip(),
            "results": results,
        }

    def _execute_with_policy(
        self,
        *,
        policy: dict[str, Any],
        integration_name: str,
        method: str,
        url: str,
        headers: dict[str, str],
        body: str | None,
        ssrf_protected: bool,
    ) -> dict[str, Any]:
        policy_check = self._validate(
            policy=policy,
            method=method,
            url=url,
            headers=headers,
            ssrf_protected=ssrf_protected,
        )
        if not policy_check.allowed:
            return self._policy_denied(
                reason=policy_check.reason or "policy denied",
                integration=integration_name,
                method=method,
                url=url,
            )

        rate_check = self._check_rate_limit(integration_name, policy)
        if not rate_check.allowed:
            return {
                "success": False,
                "error": "rate_limited",
                "reason": rate_check.reason,
                "integration": integration_name,
                "requested_method": method,
                "requested_url": url,
            }

        effective_headers, effective_url, inject_error = self._inject(
            policy=policy,
            headers=headers,
            url=url,
        )
        if inject_error is not None:
            return self._invalid_request(inject_error)

        return self._execute(
            method=method,
            url=effective_url,
            headers=effective_headers,
            body=body,
            max_bytes=int(policy.get("max_response_bytes", _DEFAULT_PUBLIC_MAX_RESPONSE_BYTES)),
        )

    def _validate(
        self,
        *,
        policy: dict[str, Any],
        method: str,
        url: str,
        headers: dict[str, str],
        ssrf_protected: bool,
    ) -> PolicyResult:
        allowed_methods = {str(item).upper() for item in policy.get("allowed_methods", ["GET"])}
        if method.upper() not in allowed_methods:
            return PolicyResult(False, f"method {method} is not allowed")

        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return PolicyResult(False, "url must include a hostname")

        allowed_hosts = [str(item) for item in policy.get("allowed_hosts", ["*"])]
        if not any(fnmatch.fnmatch(host, pattern.lower()) for pattern in allowed_hosts):
            return PolicyResult(False, f"host {host} not matched by allowed_hosts {allowed_hosts}")

        path = parsed.path or "/"
        allowed_paths = [str(item) for item in policy.get("allowed_paths", ["/*"])]
        if not any(fnmatch.fnmatch(path, pattern) for pattern in allowed_paths):
            return PolicyResult(False, f"path {path} not matched by allowed_paths {allowed_paths}")

        protected_headers = {str(item).lower() for item in policy.get("protected_headers", [])}
        lower_headers = {key.lower(): key for key in headers}
        blocked = sorted(name for name in protected_headers if name in lower_headers)
        if blocked:
            return PolicyResult(
                False,
                f"request includes protected headers {blocked}; these are broker-managed",
            )

        if ssrf_protected:
            ssrf_check = self._validate_public_url(url)
            if not ssrf_check.allowed:
                return ssrf_check

        return PolicyResult(True)

    def _inject(
        self,
        *,
        policy: dict[str, Any],
        headers: dict[str, str],
        url: str,
    ) -> tuple[dict[str, str], str, str | None]:
        merged = dict(policy.get("default_headers", {}))
        merged.update(headers)

        auth_type = str(policy.get("auth_type", "")).lower()
        token = policy.get("token")
        if auth_type and auth_type != "none":
            if not isinstance(token, str) or not token:
                return headers, url, "integration token is not configured"

        if auth_type == "bearer":
            merged["Authorization"] = f"Bearer {token}"
        elif auth_type == "header":
            header_name = policy.get("header_name", "Authorization")
            if not isinstance(header_name, str) or not header_name.strip():
                return headers, url, "integration header_name is invalid"
            prefix = policy.get("header_prefix", "")
            if not isinstance(prefix, str):
                return headers, url, "integration header_prefix must be a string"
            merged[header_name.strip()] = f"{prefix}{token}"
        elif auth_type == "query":
            query_param = policy.get("query_param")
            if not isinstance(query_param, str) or not query_param.strip():
                return headers, url, "integration query_param is invalid"
            parsed = urlsplit(url)
            query_items = parse_qsl(parsed.query, keep_blank_values=True)
            query_items.append((query_param.strip(), cast(str, token)))
            url = urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    urlencode(query_items),
                    parsed.fragment,
                )
            )

        return merged, url, None

    def _execute(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: str | None,
        max_bytes: int,
    ) -> dict[str, Any]:
        try:
            response = self._http.request(
                method=method,
                url=url,
                headers=headers,
                data=body,
                timeout=(_DEFAULT_CONNECT_TIMEOUT_SECONDS, _DEFAULT_READ_TIMEOUT_SECONDS),
                stream=True,
            )
            raw_body, truncated = _read_limited_bytes(response, byte_limit=max_bytes)
            status_code = int(response.status_code)
            response_headers = _headers_to_dict(response.headers)
            content_type = response_headers.get("Content-Type", "")
            body_text = raw_body.decode("utf-8", errors="replace")
            return {
                "success": True,
                "status_code": status_code,
                "headers": response_headers,
                "content_type": content_type,
                "body": body_text,
                "truncated": truncated,
                "raw_bytes_len": len(raw_body),
            }
        except requests.RequestException as exc:
            return {
                "success": False,
                "error": "request_failed",
                "detail": str(exc),
            }

    def _check_rate_limit(self, integration_name: str, policy: dict[str, Any]) -> PolicyResult:
        rate_limit = policy.get("rate_limit")
        if not isinstance(rate_limit, dict):
            return PolicyResult(True)

        requests_per_window = rate_limit.get("requests")
        per_seconds = rate_limit.get("per_seconds")
        if (
            isinstance(requests_per_window, bool)
            or not isinstance(requests_per_window, int)
            or requests_per_window <= 0
            or isinstance(per_seconds, bool)
            or not isinstance(per_seconds, (int, float))
            or float(per_seconds) <= 0
        ):
            return PolicyResult(True)

        now = time.monotonic()
        state = self._rate_limit_state.setdefault(
            integration_name,
            {
                "tokens": float(requests_per_window),
                "updated_at": now,
            },
        )
        elapsed = max(0.0, now - state["updated_at"])
        refill_rate = float(requests_per_window) / float(per_seconds)
        state["tokens"] = min(float(requests_per_window), state["tokens"] + (elapsed * refill_rate))
        state["updated_at"] = now
        if state["tokens"] < 1.0:
            return PolicyResult(False, f"rate limit exceeded for integration '{integration_name}'")
        state["tokens"] -= 1.0
        return PolicyResult(True)

    def _validate_public_url(self, url: str) -> PolicyResult:
        parsed = urlsplit(url)
        host = parsed.hostname
        if not host:
            return PolicyResult(False, "url must include a hostname")
        try:
            infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            return PolicyResult(False, f"dns resolution failed for host {host}: {exc}")

        for info in infos:
            ip_raw = info[4][0]
            ip_obj = ipaddress.ip_address(ip_raw)
            if isinstance(ip_obj, ipaddress.IPv4Address):
                if any(ip_obj in network for network in _RESERVED_V4_NETWORKS):
                    return PolicyResult(False, f"ssrf protection blocked private address {ip_obj}")
            else:
                if any(ip_obj in network for network in _RESERVED_V6_NETWORKS):
                    return PolicyResult(False, f"ssrf protection blocked private address {ip_obj}")
        return PolicyResult(True)

    @staticmethod
    def _normalize_method(method_raw: Any) -> tuple[str, str | None]:
        if not isinstance(method_raw, str) or not method_raw.strip():
            return "", "http_request.method must be a non-empty string"
        return method_raw.strip().upper(), None

    @staticmethod
    def _normalize_url(url_raw: Any) -> tuple[str, str | None]:
        if not isinstance(url_raw, str) or not url_raw.strip():
            return "", "http_request.url must be a non-empty string"
        return url_raw.strip(), None

    @staticmethod
    def _normalize_headers(raw: Any) -> tuple[dict[str, str], str | None]:
        if raw is None:
            return {}, None
        if not isinstance(raw, dict):
            return {}, "http_request.headers must be an object when provided"
        normalized: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return {}, "http_request.headers must contain only string keys and values"
            normalized[key] = value
        return normalized, None

    @staticmethod
    def _normalize_public_policy(public_policy: dict[str, Any] | None) -> dict[str, Any]:
        raw = public_policy or {}
        enabled = bool(raw.get("enabled", True))
        allowed_methods_raw = raw.get("allowed_methods", ["GET"])
        if not isinstance(allowed_methods_raw, list):
            allowed_methods_raw = ["GET"]
        allowed_methods = [
            str(method).strip().upper()
            for method in allowed_methods_raw
            if str(method).strip()
        ]
        if not allowed_methods:
            allowed_methods = ["GET"]

        max_response_raw = raw.get("max_response_bytes", _DEFAULT_PUBLIC_MAX_RESPONSE_BYTES)
        if isinstance(max_response_raw, bool):
            max_response_bytes = _DEFAULT_PUBLIC_MAX_RESPONSE_BYTES
        else:
            try:
                max_response_bytes = int(max_response_raw)
            except (TypeError, ValueError):
                max_response_bytes = _DEFAULT_PUBLIC_MAX_RESPONSE_BYTES
        if max_response_bytes <= 0:
            max_response_bytes = _DEFAULT_PUBLIC_MAX_RESPONSE_BYTES

        return {
            "enabled": enabled,
            "auth_type": "none",
            "allowed_hosts": ["*"],
            "allowed_methods": allowed_methods,
            "allowed_paths": ["/*"],
            "protected_headers": ["authorization"],
            "default_headers": {},
            "max_response_bytes": max_response_bytes,
        }

    @staticmethod
    def _normalize_web_search_config(web_search_config: dict[str, Any] | None) -> dict[str, Any]:
        raw = web_search_config or {}
        endpoint = raw.get("endpoint", _DEFAULT_WEB_SEARCH_ENDPOINT)
        if not isinstance(endpoint, str) or not endpoint.strip():
            endpoint = _DEFAULT_WEB_SEARCH_ENDPOINT

        fmt = raw.get("format", _DEFAULT_WEB_SEARCH_FORMAT)
        if not isinstance(fmt, str):
            fmt = _DEFAULT_WEB_SEARCH_FORMAT
        fmt = fmt.strip().lower()
        if fmt not in {"brave", "searxng"}:
            fmt = _DEFAULT_WEB_SEARCH_FORMAT

        max_results_raw = raw.get("max_results", _DEFAULT_WEB_SEARCH_MAX_RESULTS)
        if isinstance(max_results_raw, bool):
            max_results = _DEFAULT_WEB_SEARCH_MAX_RESULTS
        else:
            try:
                max_results = int(max_results_raw)
            except (TypeError, ValueError):
                max_results = _DEFAULT_WEB_SEARCH_MAX_RESULTS
        if max_results <= 0:
            max_results = _DEFAULT_WEB_SEARCH_MAX_RESULTS

        return {
            "endpoint": endpoint.strip(),
            "format": fmt,
            "max_results": max_results,
        }

    @staticmethod
    def _normalize_content_type(raw: str) -> str:
        if not isinstance(raw, str):
            return ""
        return raw.split(";", 1)[0].strip().lower()

    @staticmethod
    def _invalid_request(reason: str) -> dict[str, Any]:
        return {
            "success": False,
            "error": "invalid_request",
            "reason": reason,
        }

    @staticmethod
    def _policy_denied(
        *,
        reason: str,
        integration: str,
        method: str,
        url: str,
    ) -> dict[str, Any]:
        return {
            "success": False,
            "error": "policy_denied",
            "reason": reason,
            "integration": integration,
            "requested_method": method,
            "requested_url": url,
        }


def _headers_to_dict(headers: Any) -> dict[str, str]:
    if isinstance(headers, dict):
        return {str(k): str(v) for k, v in headers.items()}
    items = getattr(headers, "items", None)
    if callable(items):
        return {str(k): str(v) for k, v in items()}
    return {}


def _read_limited_bytes(response: requests.Response, *, byte_limit: int) -> tuple[bytes, bool]:
    if byte_limit <= 0:
        return b"", True

    buffer = bytearray()
    remaining = byte_limit
    truncated = False
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue
        if len(chunk) <= remaining:
            buffer.extend(chunk)
            remaining -= len(chunk)
            if remaining == 0:
                truncated = True
                break
            continue

        buffer.extend(chunk[:remaining])
        truncated = True
        remaining = 0
        break

    return bytes(buffer), truncated


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    if limit <= 0:
        return "", bool(value)
    if len(value) <= limit:
        return value, False
    truncated = value[:limit]
    notice = f"\n\n[... truncated, original {len(value)} chars ...]"
    return f"{truncated}{notice}", True


def _normalize_brave_results(payload: dict[str, Any], *, max_results: int) -> list[dict[str, str]]:
    web = payload.get("web")
    if not isinstance(web, dict):
        return []
    rows = web.get("results")
    if not isinstance(rows, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("description")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        normalized.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet if isinstance(snippet, str) else "",
            }
        )
        if len(normalized) >= max_results:
            break
    return normalized


def _normalize_searxng_results(
    payload: dict[str, Any],
    *,
    max_results: int,
) -> list[dict[str, str]]:
    rows = payload.get("results")
    if not isinstance(rows, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("content")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        normalized.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet if isinstance(snippet, str) else "",
            }
        )
        if len(normalized) >= max_results:
            break
    return normalized
