"""Configuration loading for strangeclaw."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, cast

import yaml

DEFAULT_FALLBACK_CONFIG = Path(__file__).resolve().parent / "config.example.yaml"
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

REQUIRED_FIELDS: tuple[tuple[str, ...], ...] = (
    ("mode",),
    ("adapter",),
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
