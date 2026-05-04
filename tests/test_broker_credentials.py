"""Tests for host-only request broker credentials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from broker.credentials import (
    CredentialConfigError,
    default_credentials_path,
    load_host_credentials,
)


def _write_credentials(path: Path, data: dict[str, Any], *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    path.chmod(mode)


def _valid_credentials(token: str = "${NOTION_TOKEN}") -> dict[str, Any]:
    return {
        "credentials": {
            "notion": {
                "type": "bearer",
                "token": token,
                "allowed_hosts": ["api.notion.com", "API.NOTION.COM"],
                "allowed_methods": ["GET", "POST", "PATCH", "POST"],
                "allowed_paths": ["/v1/pages", "/v1/data_sources/*", "/v1/pages"],
                "default_headers": {"Notion-Version": "2026-03-11"},
            }
        }
    }


def test_load_host_credentials_allows_missing_file(tmp_path: Path) -> None:
    registry = load_host_credentials(tmp_path / "missing.yaml")

    assert registry.credentials == {}
    assert registry.names() == []
    assert registry.safe_metadata() == []


def test_load_host_credentials_rejects_missing_file_when_required(tmp_path: Path) -> None:
    with pytest.raises(CredentialConfigError, match="not found"):
        load_host_credentials(tmp_path / "missing.yaml", allow_missing=False)


def test_default_credentials_path_uses_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert default_credentials_path() == tmp_path / ".strangeclaw" / "secrets.yaml"


def test_load_host_credentials_resolves_env_and_normalizes_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTION_TOKEN", "notion-secret-token")
    path = tmp_path / "secrets.yaml"
    _write_credentials(path, _valid_credentials())

    registry = load_host_credentials(path)
    credential = registry.get("notion")

    assert credential is not None
    assert credential.name == "notion"
    assert credential.credential_type == "bearer"
    assert credential.token == "notion-secret-token"
    assert credential.allowed_hosts == ("api.notion.com",)
    assert credential.allowed_methods == ("GET", "POST", "PATCH")
    assert credential.allowed_paths == ("/v1/pages", "/v1/data_sources/*")
    assert credential.default_headers == {"Notion-Version": "2026-03-11"}


def test_safe_metadata_contains_no_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NOTION_TOKEN", "notion-secret-token")
    path = tmp_path / "secrets.yaml"
    _write_credentials(path, _valid_credentials())

    metadata = load_host_credentials(path).safe_metadata()

    assert metadata == [
        {
            "name": "notion",
            "type": "bearer",
            "allowed_hosts": ["api.notion.com"],
            "allowed_methods": ["GET", "POST", "PATCH"],
            "allowed_paths": ["/v1/pages", "/v1/data_sources/*"],
        }
    ]
    assert "notion-secret-token" not in repr(metadata)


def test_synthetic_integration_loads_without_provider_specific_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LINEAR_TOKEN", "linear-secret-token")
    path = tmp_path / "secrets.yaml"
    _write_credentials(
        path,
        {
            "credentials": {
                "linear": {
                    "type": "bearer",
                    "token": "${LINEAR_TOKEN}",
                    "allowed_hosts": ["api.linear.app"],
                    "allowed_methods": ["GET", "POST"],
                    "allowed_paths": ["/graphql"],
                    "default_headers": {"Content-Type": "application/json"},
                }
            }
        },
    )

    registry = load_host_credentials(path)

    assert registry.names() == ["linear"]
    assert registry.get("linear") is not None
    assert registry.safe_metadata()[0]["allowed_hosts"] == ["api.linear.app"]


def test_load_host_credentials_rejects_missing_env_var(tmp_path: Path) -> None:
    path = tmp_path / "secrets.yaml"
    _write_credentials(path, _valid_credentials(token="${MISSING_NOTION_TOKEN}"))

    with pytest.raises(CredentialConfigError, match="MISSING_NOTION_TOKEN"):
        load_host_credentials(path)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda data: data["credentials"].__setitem__("bad provider", {}), "keys"),
        (lambda data: data["credentials"]["notion"].__setitem__("type", "basic"), "type"),
        (
            lambda data: data["credentials"]["notion"].__setitem__(
                "allowed_hosts",
                ["https://api.notion.com"],
            ),
            "allowed_hosts",
        ),
        (
            lambda data: data["credentials"]["notion"].__setitem__(
                "allowed_hosts",
                ["api.notion.com:443"],
            ),
            "allowed_hosts",
        ),
        (
            lambda data: data["credentials"]["notion"].__setitem__(
                "allowed_methods",
                ["TRACE"],
            ),
            "allowed_methods",
        ),
        (
            lambda data: data["credentials"]["notion"].__setitem__(
                "allowed_paths",
                ["v1/pages"],
            ),
            "allowed_paths",
        ),
        (
            lambda data: data["credentials"]["notion"].__setitem__(
                "allowed_paths",
                ["/v1/(pages|blocks)"],
            ),
            "allowed_paths",
        ),
        (
            lambda data: data["credentials"]["notion"].__setitem__(
                "default_headers",
                {"Authorization": "Bearer nope"},
            ),
            "default_headers",
        ),
    ],
)
def test_load_host_credentials_rejects_invalid_policy_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Any,
    match: str,
) -> None:
    monkeypatch.setenv("NOTION_TOKEN", "notion-secret-token")
    data = _valid_credentials()
    mutator(data)
    path = tmp_path / "secrets.yaml"
    _write_credentials(path, data)

    with pytest.raises(CredentialConfigError, match=match):
        load_host_credentials(path)


def test_load_host_credentials_rejects_non_mapping_credentials(tmp_path: Path) -> None:
    path = tmp_path / "secrets.yaml"
    _write_credentials(path, {"credentials": ["not", "mapping"]})

    with pytest.raises(CredentialConfigError, match="credentials must be a mapping"):
        load_host_credentials(path)


def test_load_host_credentials_rejects_insecure_permissions_on_posix(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("POSIX permission check only applies on POSIX")
    path = tmp_path / "secrets.yaml"
    _write_credentials(path, _valid_credentials(token="plain-token"), mode=0o644)

    with pytest.raises(CredentialConfigError, match="group/world-readable"):
        load_host_credentials(path)


def test_load_host_credentials_can_allow_insecure_permissions_for_tests(
    tmp_path: Path,
) -> None:
    path = tmp_path / "secrets.yaml"
    _write_credentials(path, _valid_credentials(token="plain-token"), mode=0o644)

    registry = load_host_credentials(path, allow_insecure_permissions=True)

    assert registry.names() == ["notion"]
