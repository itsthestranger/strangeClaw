"""Configuration loading for strangeclaw."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, cast

import yaml

DEFAULT_FALLBACK_CONFIG = Path(__file__).resolve().parent / "config.example.yaml"
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
LOGGER = logging.getLogger(__name__)

REQUIRED_FIELDS: tuple[tuple[str, ...], ...] = (
    ("mode",),
    ("adapters", "enabled"),
    ("approval_mode",),
    ("llm", "model"),
    ("llm", "api_key"),
    ("llm", "max_tokens"),
    ("llm", "temperature"),
    ("context", "token_budget"),
    ("context", "summary_threshold"),
    ("context", "max_output_chars"),
    ("loop", "max_iterations"),
    ("skills", "directory"),
    ("firecracker", "binary"),
    ("firecracker", "kernel"),
    ("firecracker", "rootfs"),
    ("firecracker", "vcpu"),
    ("firecracker", "mem_mb"),
    ("firecracker", "host_iface"),
    ("firecracker", "boot_timeout"),
    ("firecracker", "tap_subnet_base"),
    ("telegram", "token"),
)


class ConfigError(ValueError):
    """Raised when configuration cannot be loaded or validated."""


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and validate strangeclaw configuration."""
    source_path = _resolve_config_path(config_path)
    raw = _load_yaml(source_path)
    resolved_any = _resolve_env_vars(raw, path=())
    if not isinstance(resolved_any, dict):
        raise ConfigError(f"Config file {source_path} must contain a top-level mapping.")
    resolved = cast(dict[str, Any], resolved_any)
    _validate_required_fields(resolved, source_path)
    _validate_optional_fields(resolved)
    return resolved


def _resolve_config_path(config_path: Path | None) -> Path:
    default_user_config = Path.home() / ".strangeclaw" / "config.yaml"
    if config_path is not None:
        path = config_path.expanduser()
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        return path

    if default_user_config.exists():
        return default_user_config
    if DEFAULT_FALLBACK_CONFIG.exists():
        return DEFAULT_FALLBACK_CONFIG
    raise ConfigError(
        "No configuration found. Expected either "
        f"{default_user_config} or {DEFAULT_FALLBACK_CONFIG}."
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(loaded, dict):
        raise ConfigError(f"Config file {path} must contain a top-level mapping.")
    return cast(dict[str, Any], loaded)


def _resolve_env_vars(value: Any, path: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_vars(item, path + (str(key),)) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item, path + (str(index),)) for index, item in enumerate(value)]
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: _lookup_env(match.group(1), path), value)
    return value


def _lookup_env(var_name: str, path: tuple[str, ...]) -> str:
    value = os.getenv(var_name)
    if value is None:
        dotted_path = ".".join(path) if path else "<root>"
        raise ConfigError(
            f"Missing environment variable '{var_name}' required by config field '{dotted_path}'."
        )
    return value


def _validate_required_fields(config: dict[str, Any], source_path: Path) -> None:
    missing: list[str] = []
    for field_path in REQUIRED_FIELDS:
        current: Any = config
        valid = True
        for key in field_path:
            if not isinstance(current, dict) or key not in current:
                valid = False
                break
            current = current[key]
        if not valid:
            missing.append(".".join(field_path))

    if missing:
        missing_text = ", ".join(missing)
        raise ConfigError(
            f"Missing required config fields in {source_path}: {missing_text}"
        )


def _validate_optional_fields(config: dict[str, Any]) -> None:
    _validate_llm_optional_fields(config)
    _validate_firecracker_optional_fields(config)
    _validate_session_journal_optional_fields(config)


def _validate_llm_optional_fields(config: dict[str, Any]) -> None:
    llm_section = config.get("llm")
    if not isinstance(llm_section, dict):
        raise ConfigError("Config field llm must be a mapping.")

    if "api_base" not in llm_section:
        llm_section["api_base"] = None
        return

    api_base = llm_section["api_base"]
    if api_base is None:
        return
    if not isinstance(api_base, str) or not api_base.strip():
        raise ConfigError("Config field llm.api_base must be a non-empty string or null.")
    llm_section["api_base"] = api_base.strip()


def _validate_firecracker_optional_fields(config: dict[str, Any]) -> None:
    firecracker_section = config.get("firecracker")
    if not isinstance(firecracker_section, dict):
        raise ConfigError("Config field firecracker must be a mapping.")

    host_expose = firecracker_section.get("host_expose")
    if host_expose is None:
        firecracker_section["host_expose"] = {"enabled": False, "ports": []}
        host_expose = firecracker_section["host_expose"]
    if not isinstance(host_expose, dict):
        raise ConfigError("Config field firecracker.host_expose must be a mapping.")

    enabled = host_expose.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("Config field firecracker.host_expose.enabled must be a boolean.")

    raw_ports = host_expose.get("ports", [])
    if not isinstance(raw_ports, list):
        raise ConfigError("Config field firecracker.host_expose.ports must be a list.")

    seen_ports: set[int] = set()
    ports: list[int] = []
    for index, item in enumerate(raw_ports):
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigError(
                "Config field firecracker.host_expose.ports must contain integers "
                f"in range 1..65535 (invalid at index {index})."
            )
        if item < 1 or item > 65535:
            raise ConfigError(
                "Config field firecracker.host_expose.ports must contain integers "
                f"in range 1..65535 (invalid value {item} at index {index})."
            )
        if item in seen_ports:
            continue
        seen_ports.add(item)
        ports.append(item)

    host_expose["enabled"] = enabled
    host_expose["ports"] = ports

    if enabled and not ports:
        LOGGER.warning(
            "firecracker.host_expose is enabled but no ports were configured; "
            "this setting currently has no effect."
        )

    log_export = firecracker_section.get("log_export")
    if log_export is None:
        firecracker_section["log_export"] = {"enabled": False, "max_bytes": 32 * 1024}
        log_export = firecracker_section["log_export"]
    if not isinstance(log_export, dict):
        raise ConfigError("Config field firecracker.log_export must be a mapping.")

    log_export_enabled = log_export.get("enabled", False)
    if not isinstance(log_export_enabled, bool):
        raise ConfigError("Config field firecracker.log_export.enabled must be a boolean.")

    max_bytes_raw = log_export.get("max_bytes", 32 * 1024)
    if isinstance(max_bytes_raw, bool):
        raise ConfigError("Config field firecracker.log_export.max_bytes must be an integer.")
    try:
        max_bytes = int(max_bytes_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Config field firecracker.log_export.max_bytes must be an integer."
        ) from exc
    if max_bytes <= 0:
        raise ConfigError(
            "Config field firecracker.log_export.max_bytes must be greater than zero."
        )

    log_export["enabled"] = log_export_enabled
    log_export["max_bytes"] = max_bytes

    lifecycle_status_messages = firecracker_section.get("lifecycle_status_messages", True)
    if not isinstance(lifecycle_status_messages, bool):
        raise ConfigError(
            "Config field firecracker.lifecycle_status_messages must be a boolean."
        )
    firecracker_section["lifecycle_status_messages"] = lifecycle_status_messages


def _validate_session_journal_optional_fields(config: dict[str, Any]) -> None:
    journal_section = config.get("session_journal")
    if journal_section is None:
        config["session_journal"] = {"enabled": False, "max_bytes": 1 * 1024 * 1024}
        return
    if not isinstance(journal_section, dict):
        raise ConfigError("Config field session_journal must be a mapping.")

    enabled = journal_section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("Config field session_journal.enabled must be a boolean.")

    max_bytes_raw = journal_section.get("max_bytes", 1 * 1024 * 1024)
    if isinstance(max_bytes_raw, bool):
        raise ConfigError("Config field session_journal.max_bytes must be an integer.")
    try:
        max_bytes = int(max_bytes_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Config field session_journal.max_bytes must be an integer.") from exc
    if max_bytes <= 0:
        raise ConfigError("Config field session_journal.max_bytes must be greater than zero.")

    journal_section["enabled"] = enabled
    journal_section["max_bytes"] = max_bytes
