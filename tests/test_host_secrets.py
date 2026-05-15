"""Tests for host-only secrets loading and validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from host_secrets import list_integration_names, load_secrets


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)


def test_load_secrets_returns_empty_for_missing_file(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    secrets_path = tmp_path / "missing.yaml"

    with caplog.at_level(logging.INFO):
        loaded = load_secrets(str(secrets_path))

    assert loaded == {}
    assert "secrets.yaml not found - no integrations configured" in caplog.text


def test_load_secrets_valid_record_applies_defaults(tmp_path: Path) -> None:
    secrets_path = tmp_path / "secrets.yaml"
    _write_yaml(
        secrets_path,
        {
            "credentials": {
                "notion": {
                    "auth_type": "bearer",
                    "token": "notion-token",
                    "allowed_hosts": ["api.notion.com"],
                }
            }
        },
    )

    loaded = load_secrets(str(secrets_path))

    assert loaded == {
        "notion": {
            "auth_type": "bearer",
            "token": "notion-token",
            "allowed_hosts": ["api.notion.com"],
            "allowed_methods": ["GET"],
            "allowed_paths": ["/*"],
            "allowed_schemes": ["https"],
            "protected_headers": ["Authorization"],
            "default_headers": {},
            "max_response_bytes": 524288,
            "rate_limit": None,
        }
    }


def test_load_secrets_skips_invalid_records_without_logging_tokens(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    leaked_token = "super-secret-should-not-appear"
    secrets_path = tmp_path / "secrets.yaml"
    _write_yaml(
        secrets_path,
        {
            "credentials": {
                "valid": {
                    "auth_type": "header",
                    "token": "ok-token",
                    "allowed_hosts": ["api.example.com"],
                    "allowed_methods": ["post"],
                },
                "missing_token": {
                    "auth_type": "bearer",
                    "token": "",
                    "allowed_hosts": ["api.example.com"],
                },
                "bad_auth": {
                    "auth_type": "oauth",
                    "token": leaked_token,
                    "allowed_hosts": ["api.example.com"],
                },
                "bad_hosts": {
                    "auth_type": "header",
                    "token": leaked_token,
                    "allowed_hosts": [],
                },
            }
        },
    )

    with caplog.at_level(logging.WARNING):
        loaded = load_secrets(str(secrets_path))

    assert list(loaded) == ["valid"]
    assert loaded["valid"]["allowed_methods"] == ["POST"]
    assert "skipping integration 'missing_token'" in caplog.text
    assert "skipping integration 'bad_auth'" in caplog.text
    assert "skipping integration 'bad_hosts'" in caplog.text
    assert leaked_token not in caplog.text


def test_load_secrets_rejects_legacy_query_auth_type(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    secrets_path = tmp_path / "secrets.yaml"
    _write_yaml(
        secrets_path,
        {
            "credentials": {
                "legacy_query": {
                    "auth_type": "query",
                    "token": "legacy-token",
                    "allowed_hosts": ["api.example.com"],
                }
            }
        },
    )

    with caplog.at_level(logging.WARNING):
        loaded = load_secrets(str(secrets_path))

    assert loaded == {}
    assert "auth_type must be one of: bearer, header" in caplog.text


def test_list_integration_names_excludes_internal_and_sorts() -> None:
    credentials: dict[str, dict[str, str]] = {
        "github": {},
        "_web_search": {},
        "notion": {},
        "_internal": {},
    }

    names = list_integration_names(credentials)

    assert names == ["github", "notion"]


def test_load_secrets_skips_web_search_with_empty_token(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    secrets_path = tmp_path / "secrets.yaml"
    _write_yaml(
        secrets_path,
        {
            "credentials": {
                "_web_search": {
                    "auth_type": "header",
                    "header_name": "X-Subscription-Token",
                    "token": "",
                    "allowed_hosts": ["localhost"],
                    "allowed_methods": ["GET"],
                    "allowed_paths": ["/*"],
                }
            }
        },
    )

    with caplog.at_level(logging.WARNING):
        loaded = load_secrets(str(secrets_path))

    assert loaded == {}
    assert "skipping integration '_web_search': token must be a non-empty string" in caplog.text


def test_load_secrets_accepts_web_search_placeholder_token(tmp_path: Path) -> None:
    secrets_path = tmp_path / "secrets.yaml"
    _write_yaml(
        secrets_path,
        {
            "credentials": {
                "_web_search": {
                    "auth_type": "header",
                    "header_name": "X-Subscription-Token",
                    "token": "unused-local-searxng-token",
                    "allowed_hosts": ["localhost", "127.0.0.1"],
                    "allowed_methods": ["GET"],
                    "allowed_paths": ["/search"],
                }
            }
        },
    )

    loaded = load_secrets(str(secrets_path))

    assert loaded["_web_search"]["token"] == "unused-local-searxng-token"
    assert loaded["_web_search"]["allowed_hosts"] == ["localhost", "127.0.0.1"]


def test_load_secrets_defaults_allowed_schemes_to_https(tmp_path: Path) -> None:
    secrets_path = tmp_path / "secrets.yaml"
    _write_yaml(
        secrets_path,
        {
            "credentials": {
                "github": {
                    "auth_type": "bearer",
                    "token": "gh-secret",
                    "allowed_hosts": ["api.github.com"],
                }
            }
        },
    )

    loaded = load_secrets(str(secrets_path))

    assert loaded["github"]["allowed_schemes"] == ["https"]


def test_load_secrets_rejects_invalid_allowed_schemes(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    secrets_path = tmp_path / "secrets.yaml"
    _write_yaml(
        secrets_path,
        {
            "credentials": {
                "bad": {
                    "auth_type": "bearer",
                    "token": "token",
                    "allowed_hosts": ["api.example.com"],
                    "allowed_schemes": ["ftp"],
                }
            }
        },
    )

    with caplog.at_level(logging.WARNING):
        loaded = load_secrets(str(secrets_path))

    assert loaded == {}
    assert "allowed_schemes must contain only: http, https" in caplog.text


def test_load_secrets_header_auth_auto_protects_header_name(tmp_path: Path) -> None:
    secrets_path = tmp_path / "secrets.yaml"
    _write_yaml(
        secrets_path,
        {
            "credentials": {
                "custom": {
                    "auth_type": "header",
                    "token": "custom-secret",
                    "header_name": "X-Api-Key",
                    "allowed_hosts": ["api.example.com"],
                    "protected_headers": ["Authorization"],
                }
            }
        },
    )

    loaded = load_secrets(str(secrets_path))

    assert loaded["custom"]["header_name"] == "X-Api-Key"
    assert loaded["custom"]["protected_headers"] == ["Authorization", "X-Api-Key"]
