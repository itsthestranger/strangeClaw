"""Top-level strangeclaw entrypoint module."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from adapters.cli import CLIAdapter
from adapters.telegram import TelegramAdapter, TelegramLimits
from config import load_config
from coordinator import Coordinator
from sandbox.fire import FireSandbox, load_firecracker_config
from sandbox.yolo import YoloSandbox
from session import load as load_session


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="strangeclaw")
    parser.add_argument("--resume", type=str, default=None, help="Resume from session id")
    return parser.parse_args(argv)


def _build_yolo_sandbox(config: dict[str, Any]) -> YoloSandbox:
    return YoloSandbox(
        agent_config=config,
    )


def _build_sandbox_factory(config: dict[str, Any]) -> Callable[[], Any]:
    mode = config.get("mode")
    if mode == "yolo":
        return lambda: _build_yolo_sandbox(config)
    if mode == "fire":
        fire_cfg = load_firecracker_config(config)
        return lambda: FireSandbox(
            firecracker_config=fire_cfg,
            agent_config=dict(config),
        )
    raise ValueError(f"Unsupported mode: {mode}")


def _enabled_adapters(config: dict[str, Any]) -> list[str]:
    adapters_cfg = config.get("adapters")
    if not isinstance(adapters_cfg, dict):
        raise ValueError("Config field adapters must be a mapping.")
    enabled = adapters_cfg.get("enabled")
    if not isinstance(enabled, list):
        raise ValueError("Config field adapters.enabled must be a list.")
    names = [str(item).strip() for item in enabled if str(item).strip()]
    if not names:
        raise ValueError("Config field adapters.enabled cannot be empty.")
    return names


def _telegram_adapter_config(
    config: dict[str, Any],
    *,
    multi_adapter_mode: bool,
) -> dict[str, Any]:
    telegram_cfg = config.get("telegram", {})
    if not isinstance(telegram_cfg, dict):
        raise ValueError("Config field telegram must be a mapping.")

    raw_chat_ids = telegram_cfg.get("allowed_chat_ids", [])
    if raw_chat_ids is not None and not isinstance(raw_chat_ids, list):
        raise ValueError("Config field telegram.allowed_chat_ids must be a list when provided.")
    allowed_chat_ids = [int(chat_id) for chat_id in raw_chat_ids] if raw_chat_ids else []

    local_mode = bool(telegram_cfg.get("local_mode", True))
    if not local_mode and not allowed_chat_ids:
        raise ValueError(
            "telegram.allowed_chat_ids must be configured when telegram.local_mode is false."
        )

    limits = TelegramLimits(
        max_active_sessions=int(telegram_cfg.get("max_active_sessions", 8)),
        max_output_total_bytes=int(telegram_cfg.get("max_output_total_bytes", 50 * 1024 * 1024)),
        max_output_file_bytes=int(telegram_cfg.get("max_output_file_bytes", 10 * 1024 * 1024)),
    )
    return {
        "token": str(telegram_cfg.get("token", "")),
        "allowed_chat_ids": allowed_chat_ids,
        "limits": limits,
        # Use namespaced session ids to avoid cross-adapter collisions in persistence paths.
        "session_id_prefix": "telegram-" if multi_adapter_mode else "",
    }


def _coordinator_max_active_sessions(
    *,
    config: dict[str, Any],
    enabled_adapters: list[str],
) -> int | None:
    if "telegram" not in enabled_adapters:
        return None
    telegram_cfg = config.get("telegram", {})
    if not isinstance(telegram_cfg, dict):
        raise ValueError("Config field telegram must be a mapping.")
    value = int(telegram_cfg.get("max_active_sessions", 8))
    if value <= 0:
        raise ValueError("telegram.max_active_sessions must be greater than zero.")
    return value


def _session_journal_config(config: dict[str, Any]) -> dict[str, Any]:
    journal_cfg = config.get("session_journal", {})
    if not isinstance(journal_cfg, dict):
        raise ValueError("Config field session_journal must be a mapping.")
    return {
        "enabled": bool(journal_cfg.get("enabled", False)),
        "max_bytes": int(journal_cfg.get("max_bytes", 1 * 1024 * 1024)),
    }


def _fire_lifecycle_status_messages_enabled(config: dict[str, Any]) -> bool:
    fire_cfg = config.get("firecracker", {})
    if not isinstance(fire_cfg, dict):
        raise ValueError("Config field firecracker must be a mapping.")
    value = fire_cfg.get("lifecycle_status_messages", True)
    if not isinstance(value, bool):
        raise ValueError("Config field firecracker.lifecycle_status_messages must be a boolean.")
    return value


def _session_idle_timeout_seconds(config: dict[str, Any]) -> int:
    fire_cfg = config.get("firecracker", {})
    if not isinstance(fire_cfg, dict):
        raise ValueError("Config field firecracker must be a mapping.")
    raw_value = fire_cfg.get("session_idle_timeout_seconds", 1800)
    if isinstance(raw_value, bool):
        raise ValueError(
            "Config field firecracker.session_idle_timeout_seconds must be an integer."
        )
    value = int(raw_value)
    if value < 0:
        raise ValueError(
            "Config field firecracker.session_idle_timeout_seconds "
            "must be greater than or equal to zero."
        )
    return value


def _build_adapter(
    *,
    adapter_name: str,
    config: dict[str, Any],
    args: argparse.Namespace,
    multi_adapter_mode: bool,
    coordinator: Coordinator,
    sandbox_factory: Callable[[], Any],
) -> CLIAdapter | TelegramAdapter:
    if adapter_name == "cli":
        resume_state = None
        if args.resume:
            session_dir = Path.home() / ".strangeclaw" / "sessions" / args.resume
            resume_state = load_session(session_dir)
            if resume_state is None:
                raise ValueError(f"Cannot resume: session state not found for {args.resume}")

        cli_adapter = CLIAdapter(
            coordinator=coordinator,
            approval_mode=str(config["approval_mode"]),
            llm_config=dict(config["llm"]),
            resume_session_id=args.resume,
            resume_state=resume_state,
        )
        return cli_adapter

    if adapter_name == "telegram":
        if args.resume:
            raise ValueError("Resume is only supported with the cli adapter.")
        telegram_params = _telegram_adapter_config(
            config,
            multi_adapter_mode=multi_adapter_mode,
        )
        telegram_adapter = TelegramAdapter(
            sandbox_factory=sandbox_factory,
            coordinator=coordinator,
            approval_mode=str(config["approval_mode"]),
            llm_config=dict(config["llm"]),
            token=telegram_params["token"],
            allowed_chat_ids=telegram_params["allowed_chat_ids"],
            limits=telegram_params["limits"],
            session_id_prefix=telegram_params["session_id_prefix"],
        )
        return telegram_adapter

    raise ValueError(f"Unsupported adapter: {adapter_name}")


def _stop_adapter(adapter: Any) -> None:
    stop = getattr(adapter, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception:
            return


def main(argv: list[str] | None = None) -> None:
    """Run the strangeclaw application."""
    args = _parse_args(argv)
    config = load_config()
    if config["mode"] == "fire" and args.resume:
        raise ValueError(
            "Cannot resume Fire mode sessions — VM filesystem is ephemeral. "
            "Start a new session."
        )
    sandbox_factory = _build_sandbox_factory(config)

    enabled_adapters = _enabled_adapters(config)
    unsupported = [name for name in enabled_adapters if name not in {"cli", "telegram"}]
    if unsupported:
        raise ValueError(f"Unsupported adapter: {unsupported[0]}")
    if args.resume and len(enabled_adapters) > 1:
        raise ValueError("Resume cannot be used when multiple adapters are enabled.")

    multi_adapter_mode = len(enabled_adapters) > 1
    coordinator = Coordinator(
        sandbox_factory=sandbox_factory,
        approval_mode=str(config["approval_mode"]),
        llm_config=dict(config["llm"]),
        max_active_sessions=_coordinator_max_active_sessions(
            config=config,
            enabled_adapters=enabled_adapters,
        ),
        session_journal=_session_journal_config(config),
        fire_lifecycle_status_messages=_fire_lifecycle_status_messages_enabled(config),
        session_idle_timeout_seconds=_session_idle_timeout_seconds(config),
    )
    created: list[tuple[str, CLIAdapter | TelegramAdapter]] = []
    for adapter_name in enabled_adapters:
        adapter = _build_adapter(
            adapter_name=adapter_name,
            config=config,
            args=args,
            multi_adapter_mode=multi_adapter_mode,
            coordinator=coordinator,
            sandbox_factory=sandbox_factory,
        )
        created.append((adapter_name, adapter))

    foreground_index = 0
    for index, (name, _) in enumerate(created):
        if name == "cli":
            foreground_index = index
            break

    background_threads: list[threading.Thread] = []
    background_adapters: list[CLIAdapter | TelegramAdapter] = []

    try:
        for index, (_, adapter) in enumerate(created):
            if index == foreground_index:
                continue
            thread = threading.Thread(target=adapter.run, daemon=True)
            thread.start()
            background_threads.append(thread)
            background_adapters.append(adapter)

        created[foreground_index][1].run()
    except KeyboardInterrupt:
        pass
    finally:
        for adapter in background_adapters:
            _stop_adapter(adapter)
        coordinator.stop_all()
        for _, adapter in created:
            _stop_adapter(adapter)
        for thread in background_threads:
            thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
