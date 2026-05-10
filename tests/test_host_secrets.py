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
                    "auth_type": "query",
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


def test_list_integration_names_excludes_internal_and_sorts() -> None:
    credentials: dict[str, dict[str, str]] = {
        "github": {},
        "_web_search": {},
        "notion": {},
        "_internal": {},
    }

    names = list_integration_names(credentials)

    assert names == ["github", "notion"]
