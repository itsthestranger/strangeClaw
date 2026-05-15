"""Host-only secrets loading and validation for request broker integrations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import yaml

LOGGER = logging.getLogger(__name__)
DEFAULT_SECRETS_PATH = Path.home() / ".strangeclaw" / "secrets.yaml"
_VALID_AUTH_TYPES = {"bearer", "header"}
_VALID_ALLOWED_SCHEMES = {"http", "https"}


def load_secrets(path: str | None = None) -> dict[str, dict[str, Any]]:
    """Load and validate broker credentials from secrets.yaml."""
    secrets_path = Path(path).expanduser() if path is not None else DEFAULT_SECRETS_PATH
    if not secrets_path.exists():
        LOGGER.info("secrets.yaml not found - no integrations configured")
        return {}

    try:
        with secrets_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        LOGGER.warning("Failed to load secrets file %s: %s", secrets_path, exc)
        return {}

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        LOGGER.warning(
            "Invalid secrets file format in %s: top-level mapping required",
            secrets_path,
        )
        return {}

    credentials_raw = loaded.get("credentials", {})
    if credentials_raw is None:
        return {}
    if not isinstance(credentials_raw, dict):
        LOGGER.warning(
            "Invalid secrets file format in %s: credentials must be a mapping",
            secrets_path,
        )
        return {}

    credentials: dict[str, dict[str, Any]] = {}
    for name, record in credentials_raw.items():
        if not isinstance(name, str):
            LOGGER.warning("skipping integration %r: integration name must be a string", name)
            continue
        normalized, error = _normalize_record(record)
        if error is not None:
            LOGGER.warning("skipping integration '%s': %s", name, error)
            continue
        credentials[name] = normalized
    return credentials


def list_integration_names(credentials: dict[str, Any]) -> list[str]:
    """List user-visible integrations, excluding built-in underscore-prefixed names."""
    return sorted(name for name in credentials if not name.startswith("_"))


def _normalize_record(record: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(record, dict):
        return {}, "record must be a mapping"

    auth_type = record.get("auth_type")
    if not isinstance(auth_type, str):
        return {}, "auth_type must be one of: bearer, header"
    normalized_auth_type = auth_type.strip().lower()
    if normalized_auth_type not in _VALID_AUTH_TYPES:
        return {}, "auth_type must be one of: bearer, header"

    token = record.get("token")
    if not isinstance(token, str) or not token.strip():
        return {}, "token must be a non-empty string"

    allowed_hosts = _validate_non_empty_string_list(record.get("allowed_hosts"), "allowed_hosts")
    if isinstance(allowed_hosts, str):
        return {}, allowed_hosts

    allowed_methods_any = record.get("allowed_methods", ["GET"])
    allowed_methods = _validate_non_empty_string_list(allowed_methods_any, "allowed_methods")
    if isinstance(allowed_methods, str):
        return {}, allowed_methods
    normalized_methods = [method.upper() for method in allowed_methods]

    allowed_paths_any = record.get("allowed_paths", ["/*"])
    allowed_paths = _validate_non_empty_string_list(allowed_paths_any, "allowed_paths")
    if isinstance(allowed_paths, str):
        return {}, allowed_paths

    allowed_schemes_any = record.get("allowed_schemes", ["https"])
    allowed_schemes = _validate_non_empty_string_list(allowed_schemes_any, "allowed_schemes")
    if isinstance(allowed_schemes, str):
        return {}, allowed_schemes
    normalized_schemes: list[str] = []
    for scheme in allowed_schemes:
        lowered = scheme.lower()
        if lowered not in _VALID_ALLOWED_SCHEMES:
            return {}, "allowed_schemes must contain only: http, https"
        if lowered not in normalized_schemes:
            normalized_schemes.append(lowered)

    protected_headers_any = record.get("protected_headers", ["Authorization"])
    protected_headers = _validate_non_empty_string_list(protected_headers_any, "protected_headers")
    if isinstance(protected_headers, str):
        return {}, protected_headers

    default_headers_any = record.get("default_headers", {})
    if not isinstance(default_headers_any, dict):
        return {}, "default_headers must be a mapping"
    default_headers: dict[str, str] = {}
    for key, value in default_headers_any.items():
        if not isinstance(key, str):
            return {}, "default_headers keys must be strings"
        if not isinstance(value, str):
            return {}, f"default_headers.{key} must be a string"
        default_headers[key] = value

    max_response_bytes_any = record.get("max_response_bytes", 524288)
    if isinstance(max_response_bytes_any, bool):
        return {}, "max_response_bytes must be an integer"
    try:
        max_response_bytes = int(max_response_bytes_any)
    except (TypeError, ValueError):
        return {}, "max_response_bytes must be an integer"
    if max_response_bytes <= 0:
        return {}, "max_response_bytes must be greater than zero"

    rate_limit = record.get("rate_limit")
    if rate_limit is not None and not isinstance(rate_limit, dict):
        return {}, "rate_limit must be a mapping when provided"

    normalized: dict[str, Any] = {
        "auth_type": normalized_auth_type,
        "token": token,
        "allowed_hosts": allowed_hosts,
        "allowed_methods": normalized_methods,
        "allowed_paths": allowed_paths,
        "allowed_schemes": normalized_schemes,
        "protected_headers": protected_headers,
        "default_headers": default_headers,
        "max_response_bytes": max_response_bytes,
        "rate_limit": rate_limit,
    }

    header_name = "Authorization"
    if "header_name" in record:
        header_name_any = record["header_name"]
        if not isinstance(header_name_any, str) or not header_name_any.strip():
            return {}, "header_name must be a non-empty string when provided"
        header_name = header_name_any.strip()
    if normalized_auth_type == "header":
        normalized["header_name"] = header_name
        protected_lower = {value.lower() for value in protected_headers}
        if header_name.lower() not in protected_lower:
            normalized["protected_headers"] = [*protected_headers, header_name]
    elif "header_name" in record:
        normalized["header_name"] = header_name

    if "query_param" in record:
        return {}, "query_param is not supported; use bearer/header auth with host policy"

    return normalized, None


def _validate_non_empty_string_list(value: Any, field_name: str) -> list[str] | str:
    if not isinstance(value, list) or not value:
        return f"{field_name} must be a non-empty list of strings"
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return f"{field_name} must be a non-empty list of strings"
        items.append(item.strip())
    return cast(list[str], items)
