"""Request broker policy validation primitives."""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class PolicyResult:
    """Result of validating a broker request against policy."""

    allowed: bool
    reason: str | None


class RequestBroker:
    """Host-side request broker (validation slice for C5.3)."""

    def __init__(self, credentials: dict[str, Any], config: dict[str, Any]) -> None:
        self._credentials = credentials
        self._config = config

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

        if auth_type == "query":
            query_param_raw = policy.get("query_param")
            query_param = (
                query_param_raw.strip()
                if isinstance(query_param_raw, str) and query_param_raw.strip()
                else "token"
            )
            split = urlsplit(url)
            params = parse_qsl(split.query, keep_blank_values=True)
            params.append((query_param, token))
            final_url = urlunsplit(
                (split.scheme, split.netloc, split.path, urlencode(params), split.fragment)
            )
            return final_headers, final_url

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
        session.max_redirects = 5
        try:
            response = session.request(
                method=method,
                url=url,
                headers=headers,
                data=body,
                timeout=(10, 30),
                allow_redirects=True,
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
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body_text,
                "truncated": truncated,
            }
        except requests.RequestException as exc:
            return {"success": False, "error": exc.__class__.__name__, "detail": str(exc)}
        finally:
            session.close()

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
