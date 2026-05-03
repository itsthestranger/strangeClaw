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
_KNOWN_TOOL_NAMES = {"shell", "web_search", "web_fetch", "http_request"}

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
    _validate_tools_optional_fields(config)
    _validate_web_search_optional_fields(config)
    _validate_web_fetch_optional_fields(config)
    _validate_skills_optional_fields(config)
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


def _validate_tools_optional_fields(config: dict[str, Any]) -> None:
    tools_section = config.get("tools")
    default_tools = {
        "shell": True,
        "web_search": True,
        "web_fetch": True,
        "http_request": True,
    }
    if tools_section is None:
        config["tools"] = default_tools
        return
    if not isinstance(tools_section, dict):
        raise ConfigError("Config field tools must be a mapping.")

    normalized = dict(default_tools)
    for key, value in tools_section.items():
        if not isinstance(key, str):
            raise ConfigError("Config field tools keys must be strings.")
        if not isinstance(value, bool):
            raise ConfigError(f"Config field tools.{key} must be a boolean.")
        if key not in _KNOWN_TOOL_NAMES:
            LOGGER.warning(
                "Unknown tool name in config.tools: %s. Supported tools: %s",
                key,
                ", ".join(sorted(_KNOWN_TOOL_NAMES)),
            )
            continue
        normalized[key] = value
    config["tools"] = normalized


def _validate_web_search_optional_fields(config: dict[str, Any]) -> None:
    web_search = config.get("web_search")
    if web_search is None:
        config["web_search"] = {
            "endpoint": "https://api.search.brave.com/res/v1/web/search",
            "format": "brave",
            "api_key": "",
            "max_results": 10,
        }
        web_search = config["web_search"]
    if not isinstance(web_search, dict):
        raise ConfigError("Config field web_search must be a mapping.")

    endpoint = web_search.get("endpoint", "https://api.search.brave.com/res/v1/web/search")
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ConfigError("Config field web_search.endpoint must be a non-empty string.")
    web_search["endpoint"] = endpoint.strip()

    fmt = web_search.get("format", "brave")
    if not isinstance(fmt, str):
        raise ConfigError("Config field web_search.format must be a string.")
    normalized_format = fmt.strip().lower()
    if normalized_format not in {"brave", "searxng"}:
        raise ConfigError("Config field web_search.format must be either 'brave' or 'searxng'.")
    web_search["format"] = normalized_format

    api_key = web_search.get("api_key", "")
    if not isinstance(api_key, str):
        raise ConfigError("Config field web_search.api_key must be a string.")
    web_search["api_key"] = api_key
    if normalized_format == "brave" and not api_key.strip():
        LOGGER.warning(
            "web_search.format is brave but web_search.api_key is empty. "
            "Brave requests will fail until an API key is set."
        )

    max_results_raw = web_search.get("max_results", 10)
    if isinstance(max_results_raw, bool):
        raise ConfigError("Config field web_search.max_results must be an integer.")
    try:
        max_results = int(max_results_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Config field web_search.max_results must be an integer.") from exc
    if max_results <= 0:
        raise ConfigError("Config field web_search.max_results must be greater than zero.")
    web_search["max_results"] = max_results


def _validate_web_fetch_optional_fields(config: dict[str, Any]) -> None:
    web_fetch = config.get("web_fetch")
    if web_fetch is None:
        config["web_fetch"] = {"max_chars": 20000}
        return
    if not isinstance(web_fetch, dict):
        raise ConfigError("Config field web_fetch must be a mapping.")

    max_chars_raw = web_fetch.get("max_chars", 20000)
    if isinstance(max_chars_raw, bool):
        raise ConfigError("Config field web_fetch.max_chars must be an integer.")
    try:
        max_chars = int(max_chars_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Config field web_fetch.max_chars must be an integer.") from exc
    if max_chars <= 0:
        raise ConfigError("Config field web_fetch.max_chars must be greater than zero.")
    web_fetch["max_chars"] = max_chars


def _validate_skills_optional_fields(config: dict[str, Any]) -> None:
    skills = config.get("skills")
    if skills is None:
        config["skills"] = {"directory": "./skills", "max_file_chars": 20000}
        skills = config["skills"]
    if not isinstance(skills, dict):
        raise ConfigError("Config field skills must be a mapping.")

    directory_raw = skills.get("directory", "./skills")
    if not isinstance(directory_raw, str) or not directory_raw.strip():
        raise ConfigError("Config field skills.directory must be a non-empty string.")
    skills["directory"] = directory_raw.strip()

    max_file_chars_raw = skills.get("max_file_chars", 20000)
    if isinstance(max_file_chars_raw, bool):
        raise ConfigError("Config field skills.max_file_chars must be an integer.")
    try:
        max_file_chars = int(max_file_chars_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Config field skills.max_file_chars must be an integer.") from exc
    if max_file_chars <= 0:
        raise ConfigError("Config field skills.max_file_chars must be greater than zero.")
    skills["max_file_chars"] = max_file_chars


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
