from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from host_secrets import SecretsError, credentials_from_loaded, load_secrets


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_load_secrets_absent_file_returns_empty_mapping(tmp_path: Path) -> None:
    loaded = load_secrets(tmp_path / "missing.yaml")

    assert loaded == {}
    assert credentials_from_loaded(loaded) == {}


def test_load_secrets_parses_valid_record_with_defaults(tmp_path: Path) -> None:
    path = tmp_path / "secrets.yaml"
    _write(
        path,
        {
            "credentials": {
                "github": {
                    "auth_type": "header",
                    "token": "ghp_test_secret",
                    "header_name": "Authorization",
                    "allowed_hosts": ["api.github.com"],
                }
            }
        },
    )

    loaded = load_secrets(path)
    credentials = credentials_from_loaded(loaded)

    assert set(credentials) == {"github"}
    assert credentials["github"]["allowed_methods"] == ["GET"]
    assert credentials["github"]["allowed_paths"] == ["/*"]
    assert credentials["github"]["protected_headers"] == ["Authorization"]
    assert credentials["github"]["max_response_bytes"] == 524288


def test_load_secrets_skips_invalid_record_without_logging_token(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "secrets.yaml"
    _write(
        path,
        {
            "credentials": {
                "ok": {
                    "auth_type": "bearer",
                    "token": "token-ok",
                    "allowed_hosts": ["api.example.com"],
                },
                "bad": {
                    "auth_type": "bearer",
                    "token": "",  # invalid
                    "allowed_hosts": ["api.example.com"],
                },
            }
        },
    )

    with caplog.at_level("ERROR"):
        loaded = load_secrets(path)

    credentials = credentials_from_loaded(loaded)
    assert set(credentials) == {"ok"}
    assert "bad" in caplog.text
    assert "token-ok" not in caplog.text


def test_load_secrets_rejects_non_mapping_file(tmp_path: Path) -> None:
    path = tmp_path / "secrets.yaml"
    path.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(SecretsError, match="top-level mapping"):
        load_secrets(path)
