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
        "adapters": {"enabled": ["cli"]},
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

    assert isinstance(loaded["llm"]["api_key"], str)
    assert loaded["llm"]["api_key"] != ""
    assert isinstance(loaded["mode"], str)


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


def test_load_config_sets_optional_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _base_config(api_key="plain-key"))

    loaded = load_config(config_path)

    assert loaded["llm"]["api_base"] is None
    assert loaded["tools"] == {
        "shell": True,
        "web_search": True,
        "web_fetch": True,
        "http_request": True,
    }
    assert loaded["web_search"] == {
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "format": "brave",
        "max_results": 10,
    }
    assert loaded["web_fetch"] == {"max_response_bytes": 524288}
    assert loaded["skills"] == {"directory": "./skills", "max_file_chars": 20000}
    assert loaded["host_services"] == {
        "llm_timeout_seconds": 120,
        "llm_max_request_bytes": 2 * 1024 * 1024,
    }
    assert loaded["firecracker"]["log_export"] == {"enabled": False, "max_bytes": 32 * 1024}
    assert loaded["firecracker"]["lifecycle_status_messages"] is True
    assert loaded["firecracker"]["session_idle_timeout_seconds"] == 1800
    assert loaded["coordinator"] == {"max_active_sessions": 8}
    assert loaded["session_journal"] == {"enabled": False, "max_bytes": 1 * 1024 * 1024}


def test_load_config_coordinator_override_max_active_sessions(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["coordinator"] = {"max_active_sessions": 3}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    loaded = load_config(config_path)

    assert loaded["coordinator"] == {"max_active_sessions": 3}


def test_load_config_rejects_invalid_coordinator_max_active_sessions(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["coordinator"] = {"max_active_sessions": 0}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"coordinator\.max_active_sessions"):
        load_config(config_path)


def test_load_config_defaults_skills_section_when_missing(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    del config["skills"]
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    loaded = load_config(config_path)

    assert loaded["skills"] == {"directory": "./skills", "max_file_chars": 20000}


def test_load_config_defaults_skills_directory_when_missing(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["skills"] = {"max_file_chars": 12345}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    loaded = load_config(config_path)

    assert loaded["skills"] == {"directory": "./skills", "max_file_chars": 12345}


def test_load_config_rejects_invalid_skills_directory(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["skills"]["directory"] = "   "
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"skills\.directory"):
        load_config(config_path)


def test_load_config_rejects_invalid_llm_api_base_type(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["llm"]["api_base"] = 123
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"llm\.api_base"):
        load_config(config_path)


@pytest.mark.parametrize(
    "host_services",
    [
        {"llm_timeout_seconds": 0, "llm_max_request_bytes": 1024},
        {"llm_timeout_seconds": 30, "llm_max_request_bytes": 0},
        {"llm_timeout_seconds": True, "llm_max_request_bytes": 1024},
        {"llm_timeout_seconds": 30, "llm_max_request_bytes": False},
    ],
)
def test_load_config_rejects_invalid_host_services(
    tmp_path: Path,
    host_services: dict[str, Any],
) -> None:
    config = _base_config(api_key="plain-key")
    config["host_services"] = host_services
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"host_services"):
        load_config(config_path)


def test_load_config_rejects_invalid_fire_log_export_fields(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["firecracker"]["log_export"] = {"enabled": "yes", "max_bytes": "huge"}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"log_export\.enabled"):
        load_config(config_path)

    config["firecracker"]["log_export"] = {"enabled": True, "max_bytes": 0}
    _write_config(config_path, config)
    with pytest.raises(ConfigError, match=r"log_export\.max_bytes"):
        load_config(config_path)


def test_load_config_rejects_invalid_fire_lifecycle_status_messages(
    tmp_path: Path,
) -> None:
    config = _base_config(api_key="plain-key")
    config["firecracker"]["lifecycle_status_messages"] = "yes"
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"lifecycle_status_messages"):
        load_config(config_path)


def test_load_config_rejects_invalid_fire_session_idle_timeout_seconds(
    tmp_path: Path,
) -> None:
    config = _base_config(api_key="plain-key")
    config["firecracker"]["session_idle_timeout_seconds"] = -1
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"session_idle_timeout_seconds"):
        load_config(config_path)


def test_load_config_warns_on_unknown_tool_names(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _base_config(api_key="plain-key")
    config["tools"] = {"shell": True, "not_a_real_tool": False}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with caplog.at_level("WARNING"):
        loaded = load_config(config_path)

    assert loaded["tools"]["shell"] is True
    assert "not_a_real_tool" not in loaded["tools"]
    assert "Unknown tool name in config.tools" in caplog.text


def test_load_config_rejects_invalid_web_search_format(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["web_search"] = {"endpoint": "http://localhost:8080/search", "format": "duck"}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"web_search\.format"):
        load_config(config_path)


def test_load_config_allows_brave_without_web_search_api_key(
    tmp_path: Path,
) -> None:
    config = _base_config(api_key="plain-key")
    config["web_search"] = {
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "format": "brave",
    }
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    loaded = load_config(config_path)

    assert loaded["web_search"]["format"] == "brave"
    assert "api_key" not in loaded["web_search"]


def test_load_config_rejects_legacy_web_search_api_key(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["web_search"] = {
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "format": "brave",
        "api_key": "secret-key",
    }
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(
        ConfigError,
        match=(
            r"web_search\.api_key is no longer supported\. Move search credentials to "
            r"~/.strangeclaw/secrets\.yaml under credentials\._web_search\.token\."
        ),
    ):
        load_config(config_path)


def test_load_config_accepts_empty_legacy_web_search_api_key_and_strips_it(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["web_search"] = {
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "format": "brave",
        "api_key": "",
    }
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    loaded = load_config(config_path)

    assert loaded["web_search"] == {
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "format": "brave",
        "max_results": 10,
    }


def test_load_config_rejects_legacy_web_fetch_max_chars(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["web_fetch"] = {"max_chars": 20000}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(
        ConfigError,
        match=r"web_fetch\.max_chars has been removed.*max_response_bytes",
    ):
        load_config(config_path)


def test_load_config_rejects_invalid_web_fetch_max_response_bytes(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["web_fetch"] = {"max_response_bytes": 0}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"web_fetch\.max_response_bytes"):
        load_config(config_path)


def test_load_config_rejects_legacy_integrations_field(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["integrations"] = {}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"integrations is no longer supported"):
        load_config(config_path)


def test_load_config_rejects_invalid_skills_max_file_chars(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["skills"]["max_file_chars"] = 0
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"skills\.max_file_chars"):
        load_config(config_path)


def test_fire_sanitized_skills_match_loaded_config_defaults(tmp_path: Path) -> None:
    from sandbox.fire import _sanitize_agent_config_for_mmds

    config = _base_config(api_key="plain-key")
    del config["skills"]
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    loaded = load_config(config_path)
    fire_payload = _sanitize_agent_config_for_mmds(loaded)

    assert fire_payload["skills"] == loaded["skills"]
    assert fire_payload["web_search"] == loaded["web_search"]
    assert fire_payload["web_search"]["max_results"] == 10
    assert "api_key" not in fire_payload["web_search"]
    assert "integrations" not in fire_payload


def test_load_config_rejects_invalid_session_journal_fields(tmp_path: Path) -> None:
    config = _base_config(api_key="plain-key")
    config["session_journal"] = {"enabled": "yes", "max_bytes": "huge"}
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, config)

    with pytest.raises(ConfigError, match=r"session_journal\.enabled"):
        load_config(config_path)

    config["session_journal"] = {"enabled": True, "max_bytes": 0}
    _write_config(config_path, config)
    with pytest.raises(ConfigError, match=r"session_journal\.max_bytes"):
        load_config(config_path)
