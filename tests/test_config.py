"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from config import ConfigError, load_config


def _base_config(api_key: str = "${ANTHROPIC_API_KEY}") -> dict[str, Any]:
    return {
        "mode": "yolo",
        "adapter": "cli",
        "approval_mode": "review",
        "llm": {
            "model": "anthropic/claude-sonnet-4-20250514",
            "api_key": api_key,
            "max_tokens": 4096,
            "temperature": 0.2,
        },
        "context": {
            "token_budget": 4000,
            "summary_threshold": 10,
            "max_output_chars": 8000,
        },
        "loop": {"max_iterations": 50},
        "skills": {"directory": "./skills"},
        "firecracker": {
            "binary": "/usr/local/bin/firecracker",
            "kernel": "./firecracker/kernel/vmlinux",
            "rootfs": "./firecracker/rootfs/agent.ext4",
            "vcpu": 1,
            "mem_mb": 512,
            "host_iface": None,
            "boot_timeout": 30,
            "tap_subnet_base": "172.16.0.0",
        },
        "telegram": {"token": ""},
    }


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)


def test_load_config_uses_user_path_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SC_TEST_KEY", "test-user-key")
    user_config = tmp_path / ".strangeclaw" / "config.yaml"
    _write_config(user_config, _base_config(api_key="${SC_TEST_KEY}"))

    loaded = load_config()

    assert loaded["llm"]["api_key"] == "test-user-key"


def test_load_config_falls_back_to_example_when_user_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fallback-key")

    loaded = load_config()

    assert loaded["llm"]["api_key"] == "fallback-key"
    assert loaded["mode"] == "yolo"


def test_load_config_reports_missing_required_fields(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    del config["llm"]["model"]
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"llm\.model"):
        load_config(config_path)


def test_load_config_reports_missing_environment_variable(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _base_config(api_key="${THIS_KEY_DOES_NOT_EXIST}"))

    with pytest.raises(ConfigError, match="THIS_KEY_DOES_NOT_EXIST"):
        load_config(config_path)


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(tmp_path / "does-not-exist.yaml")
