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
_KNOWN_TOOL_NAMES = {"shell", "web_search", "web_fetch", "http_request", "spawn_subagent"}

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
    _validate_legacy_integrations_field(config)
    _validate_llm_optional_fields(config)
    _validate_coordinator_optional_fields(config)
    _validate_subagents_optional_fields(config)
    _validate_tools_optional_fields(config)
    _validate_web_search_optional_fields(config)
    _validate_web_fetch_optional_fields(config)
    _validate_skills_optional_fields(config)
    _validate_host_services_optional_fields(config)
    _validate_firecracker_optional_fields(config)
    _validate_session_journal_optional_fields(config)


def _validate_legacy_integrations_field(config: dict[str, Any]) -> None:
    if "integrations" not in config:
        return
    raise ConfigError(
        "Config field integrations is no longer supported. "
        "Move external API credentials to ~/.strangeclaw/secrets.yaml."
    )


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


def _validate_coordinator_optional_fields(config: dict[str, Any]) -> None:
    coordinator_section = config.get("coordinator")
    if coordinator_section is None:
        config["coordinator"] = {"max_active_sessions": 8}
        return
    if not isinstance(coordinator_section, dict):
        raise ConfigError("Config field coordinator must be a mapping.")

    max_active_sessions_raw = coordinator_section.get("max_active_sessions", 8)
    if isinstance(max_active_sessions_raw, bool):
        raise ConfigError("Config field coordinator.max_active_sessions must be an integer.")
    try:
        max_active_sessions = int(max_active_sessions_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Config field coordinator.max_active_sessions must be an integer."
        ) from exc
    if max_active_sessions <= 0:
        raise ConfigError(
            "Config field coordinator.max_active_sessions must be greater than zero."
        )
    coordinator_section["max_active_sessions"] = max_active_sessions


def _validate_subagents_optional_fields(config: dict[str, Any]) -> None:
    defaults: dict[str, Any] = {
        "enabled": False,
        "max_children_per_task": 3,
        "max_iterations": 20,
        "timeout_seconds": 600,
        "max_context_chars": 20000,
        "max_result_chars": 20000,
        "max_files_bytes": 10 * 1024 * 1024,
        "journal_events": "summary",
    }
    section = config.get("subagents")
    if section is None:
        config["subagents"] = dict(defaults)
        return
    if not isinstance(section, dict):
        raise ConfigError("Config field subagents must be a mapping.")

    enabled = section.get("enabled", defaults["enabled"])
    if not isinstance(enabled, bool):
        raise ConfigError("Config field subagents.enabled must be a boolean.")

    normalized: dict[str, Any] = {"enabled": enabled}
    for key in (
        "max_children_per_task",
        "max_iterations",
        "timeout_seconds",
        "max_context_chars",
        "max_result_chars",
        "max_files_bytes",
    ):
        normalized[key] = _require_positive_int(
            section.get(key, defaults[key]), f"subagents.{key}"
        )

    journal_events_raw = section.get("journal_events", defaults["journal_events"])
    if not isinstance(journal_events_raw, str):
        raise ConfigError("Config field subagents.journal_events must be a string.")
    journal_events = journal_events_raw.strip().lower()
    if journal_events not in {"none", "summary", "full"}:
        raise ConfigError(
            "Config field subagents.journal_events must be one of: none, summary, full."
        )
    normalized["journal_events"] = journal_events

    config["subagents"] = normalized


def _require_positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"Config field {field_name} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Config field {field_name} must be an integer.") from exc
    if result <= 0:
        raise ConfigError(f"Config field {field_name} must be greater than zero.")
    return result


def _validate_tools_optional_fields(config: dict[str, Any]) -> None:
    tools_section = config.get("tools")
    default_tools = {
        "shell": True,
        "web_search": True,
        "web_fetch": True,
        "http_request": True,
        "spawn_subagent": False,
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

    if "api_key" in web_search:
        api_key = web_search.get("api_key")
        if not isinstance(api_key, str):
            raise ConfigError("Config field web_search.api_key must be a string when provided.")
        if api_key.strip():
            raise ConfigError(
                "web_search.api_key is no longer supported. Move search credentials to "
                "~/.strangeclaw/secrets.yaml under credentials._web_search.token."
            )
        web_search.pop("api_key", None)

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
        config["web_fetch"] = {"max_response_bytes": 524288}
        return
    if not isinstance(web_fetch, dict):
        raise ConfigError("Config field web_fetch must be a mapping.")

    if "max_chars" in web_fetch:
        raise ConfigError(
            "Config field web_fetch.max_chars has been removed. Use "
            "web_fetch.max_response_bytes instead."
        )

    max_response_bytes_raw = web_fetch.get("max_response_bytes", 524288)
    if isinstance(max_response_bytes_raw, bool):
        raise ConfigError("Config field web_fetch.max_response_bytes must be an integer.")
    try:
        max_response_bytes = int(max_response_bytes_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError("Config field web_fetch.max_response_bytes must be an integer.") from exc
    if max_response_bytes <= 0:
        raise ConfigError(
            "Config field web_fetch.max_response_bytes must be greater than zero."
        )
    web_fetch["max_response_bytes"] = max_response_bytes


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


def _validate_host_services_optional_fields(config: dict[str, Any]) -> None:
    host_services = config.get("host_services")
    if host_services is None:
        config["host_services"] = {
            "llm_timeout_seconds": 120,
            "llm_max_request_bytes": 2 * 1024 * 1024,
        }
        return
    if not isinstance(host_services, dict):
        raise ConfigError("Config field host_services must be a mapping.")

    llm_timeout_raw = host_services.get("llm_timeout_seconds", 120)
    if isinstance(llm_timeout_raw, bool):
        raise ConfigError("Config field host_services.llm_timeout_seconds must be an integer.")
    if not isinstance(llm_timeout_raw, int):
        raise ConfigError(
            "Config field host_services.llm_timeout_seconds must be an integer."
        )
    llm_timeout_seconds = llm_timeout_raw
    if llm_timeout_seconds <= 0:
        raise ConfigError(
            "Config field host_services.llm_timeout_seconds must be greater than zero."
        )
    host_services["llm_timeout_seconds"] = llm_timeout_seconds

    llm_max_request_bytes_raw = host_services.get("llm_max_request_bytes", 2 * 1024 * 1024)
    if isinstance(llm_max_request_bytes_raw, bool):
        raise ConfigError(
            "Config field host_services.llm_max_request_bytes must be an integer."
        )
    if not isinstance(llm_max_request_bytes_raw, int):
        raise ConfigError(
            "Config field host_services.llm_max_request_bytes must be an integer."
        )
    llm_max_request_bytes = llm_max_request_bytes_raw
    if llm_max_request_bytes <= 0:
        raise ConfigError(
            "Config field host_services.llm_max_request_bytes must be greater than zero."
        )
    host_services["llm_max_request_bytes"] = llm_max_request_bytes


def _validate_firecracker_optional_fields(config: dict[str, Any]) -> None:
    firecracker_section = config.get("firecracker")
    if not isinstance(firecracker_section, dict):
        raise ConfigError("Config field firecracker must be a mapping.")

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

    timeout_raw = firecracker_section.get("session_idle_timeout_seconds", 1800)
    if isinstance(timeout_raw, bool):
        raise ConfigError(
            "Config field firecracker.session_idle_timeout_seconds must be an integer."
        )
    try:
        timeout = int(timeout_raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "Config field firecracker.session_idle_timeout_seconds must be an integer."
        ) from exc
    if timeout < 0:
        raise ConfigError(
            "Config field firecracker.session_idle_timeout_seconds "
            "must be greater than or equal to zero."
        )
    firecracker_section["session_idle_timeout_seconds"] = timeout


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
