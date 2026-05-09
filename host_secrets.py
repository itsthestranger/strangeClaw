"""Host-only secrets loading for broker integrations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import yaml

LOGGER = logging.getLogger(__name__)
_DEFAULT_SECRETS_PATH = Path("~/.strangeclaw/secrets.yaml").expanduser()
_VALID_AUTH_TYPES = {"bearer", "header", "query"}
_DEFAULT_ALLOWED_METHODS = ["GET"]
_DEFAULT_ALLOWED_PATHS = ["/*"]
_DEFAULT_PROTECTED_HEADERS = ["Authorization"]
_DEFAULT_MAX_RESPONSE_BYTES = 524288


class SecretsError(ValueError):
    """Raised when secrets.yaml cannot be parsed at all."""


def load_secrets(path: str | Path | None = None) -> dict[str, Any]:
    """Load and normalize secrets.yaml credentials.

    Invalid records are skipped with per-record error logs.
    """
    secrets_path = Path(path).expanduser() if path is not None else _DEFAULT_SECRETS_PATH
    if not secrets_path.exists():
        return {}

    try:
        with secrets_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SecretsError(f"Invalid YAML in {secrets_path}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SecretsError(f"Secrets file {secrets_path} must contain a top-level mapping.")

    credentials_raw = loaded.get("credentials", {})
    if credentials_raw is None:
        return {"credentials": {}}
    if not isinstance(credentials_raw, dict):
        raise SecretsError("Secrets field credentials must be a mapping when provided.")

    normalized: dict[str, Any] = {}
    for name, raw_record in credentials_raw.items():
        if not isinstance(name, str) or not name:
            LOGGER.error("Skipping invalid secrets credential name: %r", name)
            continue
        try:
            normalized[name] = _normalize_record(name=name, raw_record=raw_record)
        except ValueError as exc:
            LOGGER.error("Skipping secrets credential '%s': %s", name, exc)

    return {"credentials": normalized}


def _normalize_record(*, name: str, raw_record: Any) -> dict[str, Any]:
    if not isinstance(raw_record, dict):
        raise ValueError("record must be a mapping")

    auth_type = raw_record.get("auth_type")
    if not isinstance(auth_type, str):
        raise ValueError("auth_type must be a string")
    auth_type = auth_type.strip().lower()
    if auth_type not in _VALID_AUTH_TYPES:
        raise ValueError("auth_type must be one of bearer|header|query")

    token = raw_record.get("token")
    if not isinstance(token, str) or not token.strip():
        raise ValueError("token must be a non-empty string")

    allowed_hosts = _normalize_string_list(raw_record.get("allowed_hosts"), field="allowed_hosts")
    if not allowed_hosts:
        raise ValueError("allowed_hosts must be a non-empty list of strings")

    allowed_methods = _normalize_allowed_methods(raw_record.get("allowed_methods"))
    allowed_paths = _normalize_allowed_paths(raw_record.get("allowed_paths"))
    protected_headers = _normalize_string_list(
        raw_record.get("protected_headers", _DEFAULT_PROTECTED_HEADERS),
        field="protected_headers",
    )
    default_headers = _normalize_headers(
        raw_record.get("default_headers", {}),
        field="default_headers",
    )

    max_response_bytes_raw = raw_record.get("max_response_bytes", _DEFAULT_MAX_RESPONSE_BYTES)
    if isinstance(max_response_bytes_raw, bool):
        raise ValueError("max_response_bytes must be an integer")
    try:
        max_response_bytes = int(max_response_bytes_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_response_bytes must be an integer") from exc
    if max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be greater than zero")

    normalized: dict[str, Any] = {
        "auth_type": auth_type,
        "token": token,
        "allowed_hosts": allowed_hosts,
        "allowed_methods": allowed_methods,
        "allowed_paths": allowed_paths,
        "protected_headers": protected_headers,
        "default_headers": default_headers,
        "max_response_bytes": max_response_bytes,
    }

    if auth_type == "header":
        header_name = raw_record.get("header_name", "Authorization")
        if not isinstance(header_name, str) or not header_name.strip():
            raise ValueError("header_name must be a non-empty string when auth_type=header")
        normalized["header_name"] = header_name.strip()

    if auth_type == "query":
        query_param = raw_record.get("query_param")
        if not isinstance(query_param, str) or not query_param.strip():
            raise ValueError("query_param must be a non-empty string when auth_type=query")
        normalized["query_param"] = query_param.strip()

    rate_limit = raw_record.get("rate_limit")
    if rate_limit is not None:
        normalized["rate_limit"] = _normalize_rate_limit(rate_limit)

    # Preserve compatibility with previous config-backed header behavior.
    header_prefix = raw_record.get("header_prefix")
    if header_prefix is not None:
        if not isinstance(header_prefix, str):
            raise ValueError("header_prefix must be a string when provided")
        normalized["header_prefix"] = header_prefix

    return normalized


def _normalize_allowed_methods(raw: Any) -> list[str]:
    methods = _normalize_string_list(
        raw if raw is not None else _DEFAULT_ALLOWED_METHODS,
        field="allowed_methods",
    )
    if not methods:
        raise ValueError("allowed_methods must be a non-empty list of strings")
    return [method.strip().upper() for method in methods]


def _normalize_allowed_paths(raw: Any) -> list[str]:
    paths = _normalize_string_list(
        raw if raw is not None else _DEFAULT_ALLOWED_PATHS,
        field="allowed_paths",
    )
    if not paths:
        raise ValueError("allowed_paths must be a non-empty list of strings")
    return paths


def _normalize_rate_limit(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("rate_limit must be an object")

    requests_raw = raw.get("requests")
    per_seconds_raw = raw.get("per_seconds")
    if isinstance(requests_raw, bool) or not isinstance(requests_raw, int) or requests_raw <= 0:
        raise ValueError("rate_limit.requests must be a positive integer")
    if isinstance(per_seconds_raw, bool) or not isinstance(per_seconds_raw, (int, float)):
        raise ValueError("rate_limit.per_seconds must be a positive number")
    per_seconds = float(per_seconds_raw)
    if per_seconds <= 0:
        raise ValueError("rate_limit.per_seconds must be a positive number")

    return {
        "requests": requests_raw,
        "per_seconds": per_seconds,
    }


def _normalize_string_list(raw: Any, *, field: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a list")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field} must contain non-empty strings")
        values.append(item.strip())
    return values


def _normalize_headers(raw: Any, *, field: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be an object")
    normalized: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(value, str):
            raise ValueError(f"{field} must contain string keys and string values")
        normalized[key] = value
    return normalized


def credentials_from_loaded(loaded: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract normalized credentials mapping from `load_secrets()` output."""
    raw = loaded.get("credentials", {})
    if not isinstance(raw, dict):
        return {}
    return cast(dict[str, dict[str, Any]], raw)
