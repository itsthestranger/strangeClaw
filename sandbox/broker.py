"""Request broker policy validation primitives."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any
from urllib.parse import urlparse


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
