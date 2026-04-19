"""Tests for main entrypoint wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import main


class FakeSandbox:
    """Sandbox test double."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.stop_called = False

    def stop(self) -> None:
        self.stop_called = True


class FakeAdapter:
    """Adapter test double."""

    def __init__(
        self,
        *,
        coordinator: Any,
        approval_mode: str,
        llm_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.coordinator = coordinator
        self.approval_mode = approval_mode
        self.llm_config = llm_config
        self.extra = kwargs
        self.run_called = False
        self.raise_interrupt = False
        self.stop_called = False

    def run(self) -> None:
        self.run_called = True
        if self.raise_interrupt:
            raise KeyboardInterrupt()

    def stop(self) -> None:
        self.stop_called = True


class FakeTelegramAdapter:
    """Telegram adapter test double."""

    def __init__(
        self,
        *,
        sandbox_factory: Any,
        coordinator: Any,
        approval_mode: str,
        llm_config: dict[str, Any],
        token: str,
        allowed_chat_ids: list[int] | None = None,
        limits: Any | None = None,
        session_id_prefix: str = "",
    ) -> None:
        self.sandbox_factory = sandbox_factory
        self.coordinator = coordinator
        self.approval_mode = approval_mode
        self.llm_config = llm_config
        self.token = token
        self.allowed_chat_ids = allowed_chat_ids or []
        self.limits = limits
        self.session_id_prefix = session_id_prefix
        self.run_called = False
        self.stop_called = False

    def run(self) -> None:
        self.run_called = True

    def stop(self) -> None:
        self.stop_called = True


class FakeCoordinator:
    """Coordinator test double."""

    def __init__(
        self,
        *,
        sandbox_factory: Any,
        approval_mode: str,
        llm_config: dict[str, Any] | None = None,
        max_active_sessions: int | None = None,
        session_journal: dict[str, Any] | None = None,
        fire_lifecycle_status_messages: bool = True,
    ) -> None:
        self.sandbox_factory = sandbox_factory
        self.approval_mode = approval_mode
        self.llm_config = llm_config
        self.max_active_sessions = max_active_sessions
        self.session_journal = session_journal
        self.fire_lifecycle_status_messages = fire_lifecycle_status_messages
        self.seed_calls: list[tuple[str, dict[str, Any]]] = []
        self.stop_session_calls: list[str] = []
        self.stop_all_called = False

    def seed_state(self, *, session_id: str, state: dict[str, Any]) -> None:
        self.seed_calls.append((session_id, state))

    def start_task(self, *, session_id: str, text: str, sink: Any) -> str:
        del session_id
        del text
        del sink
        return "started"

    def pending_role(self, *, session_id: str) -> Any:
        del session_id
        return None

    def submit_reply(self, *, session_id: str, approved: bool, text: str) -> bool:
        del session_id
        del approved
        del text
        return True

    def stop_session(self, *, session_id: str) -> None:
        self.stop_session_calls.append(session_id)

    def stop_all(self) -> None:
        self.stop_all_called = True


def _config() -> dict[str, Any]:
    return {
        "mode": "yolo",
        "adapters": {"enabled": ["cli"]},
        "approval_mode": "review",
        "llm": {"model": "x", "api_key": "k"},
        "skills": {"directory": "./skills"},
        "loop": {"max_iterations": 7},
        "context": {"token_budget": 1234, "summary_threshold": 9},
        "telegram": {
            "token": "bot-token",
            "local_mode": True,
            "allowed_chat_ids": [123],
            "max_active_sessions": 8,
            "max_output_total_bytes": 50 * 1024 * 1024,
            "max_output_file_bytes": 10 * 1024 * 1024,
        },
    }


def test_main_wires_config_to_sandbox_and_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    def fake_load_config() -> dict[str, Any]:
        return _config()

    def fake_sandbox_factory(**kwargs: Any) -> FakeSandbox:
        sandbox = FakeSandbox(**kwargs)
        created["sandbox"] = sandbox
        return sandbox

    def fake_coordinator_factory(**kwargs: Any) -> FakeCoordinator:
        coordinator = FakeCoordinator(**kwargs)
        created["coordinator"] = coordinator
        return coordinator

    def fake_adapter_factory(
        *, coordinator: Any, approval_mode: str, llm_config: dict[str, Any], **kwargs: Any
    ) -> FakeAdapter:
        adapter = FakeAdapter(
            coordinator=coordinator,
            approval_mode=approval_mode,
            llm_config=llm_config,
            **kwargs,
        )
        created["adapter"] = adapter
        return adapter

    monkeypatch.setattr(main, "load_config", fake_load_config)
    monkeypatch.setattr(main, "YoloSandbox", fake_sandbox_factory)
    monkeypatch.setattr(main, "Coordinator", fake_coordinator_factory)
    monkeypatch.setattr(main, "CLIAdapter", fake_adapter_factory)

    main.main([])

    coordinator = created["coordinator"]
    sandbox = coordinator.sandbox_factory()
    adapter = created["adapter"]
    assert sandbox.kwargs["skills_dir"] == "./skills"
    assert sandbox.kwargs["max_iterations"] == 7
    assert sandbox.kwargs["token_budget"] == 1234
    assert sandbox.kwargs["summary_threshold"] == 9
    assert adapter.approval_mode == "review"
    assert adapter.llm_config == {"model": "x", "api_key": "k"}
    assert adapter.coordinator is coordinator
    assert coordinator.fire_lifecycle_status_messages is True
    assert adapter.run_called is True


def test_main_stops_sandbox_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    monkeypatch.setattr(main, "load_config", _config)

    def fake_coordinator_factory(**kwargs: Any) -> FakeCoordinator:
        coordinator = FakeCoordinator(**kwargs)
        created["coordinator"] = coordinator
        return coordinator

    def fake_adapter_factory(
        *, coordinator: Any, approval_mode: str, llm_config: dict[str, Any], **kwargs: Any
    ) -> FakeAdapter:
        del approval_mode
        del llm_config
        adapter = FakeAdapter(
            coordinator=coordinator,
            approval_mode="review",
            llm_config={},
            **kwargs,
        )
        adapter.raise_interrupt = True
        created["adapter"] = adapter
        return adapter

    monkeypatch.setattr(main, "Coordinator", fake_coordinator_factory)
    monkeypatch.setattr(main, "CLIAdapter", fake_adapter_factory)

    main.main([])

    assert created["coordinator"].stop_all_called is True
    assert created["adapter"].stop_called is True


def test_main_rejects_unsupported_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _config()
    bad["mode"] = "nimbus"
    monkeypatch.setattr(main, "load_config", lambda: bad)
    with pytest.raises(ValueError, match="Unsupported mode"):
        main.main([])


def test_main_rejects_resume_for_fire_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _config()
    bad["mode"] = "fire"
    monkeypatch.setattr(main, "load_config", lambda: bad)
    with pytest.raises(
        ValueError,
        match=(
            "Cannot resume Fire mode sessions — VM filesystem is ephemeral. "
            "Start a new session."
        ),
    ):
        main.main(["--resume", "abc-1"])


def test_main_wires_fire_mode_sandbox_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    cfg["mode"] = "fire"
    created: dict[str, Any] = {}

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "load_firecracker_config", lambda _: {"fire": "cfg"})

    def fake_fire_sandbox_factory(**kwargs: Any) -> FakeSandbox:
        sandbox = FakeSandbox(**kwargs)
        created["sandbox"] = sandbox
        return sandbox

    def fake_coordinator_factory(**kwargs: Any) -> FakeCoordinator:
        coordinator = FakeCoordinator(**kwargs)
        created["coordinator"] = coordinator
        return coordinator

    def fake_adapter_factory(
        *, coordinator: Any, approval_mode: str, llm_config: dict[str, Any], **kwargs: Any
    ) -> FakeAdapter:
        adapter = FakeAdapter(
            coordinator=coordinator,
            approval_mode=approval_mode,
            llm_config=llm_config,
            **kwargs,
        )
        created["adapter"] = adapter
        return adapter

    monkeypatch.setattr(main, "FireSandbox", fake_fire_sandbox_factory)
    monkeypatch.setattr(main, "Coordinator", fake_coordinator_factory)
    monkeypatch.setattr(main, "CLIAdapter", fake_adapter_factory)

    main.main([])

    coordinator = created["coordinator"]
    adapter = created["adapter"]
    sandbox = coordinator.sandbox_factory()
    assert sandbox.kwargs["firecracker_config"] == {"fire": "cfg"}
    assert sandbox.kwargs["llm_config"] == {"model": "x", "api_key": "k"}
    assert adapter.coordinator is coordinator
    assert adapter.run_called is True


def test_main_passes_fire_lifecycle_status_messages_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    cfg["firecracker"] = {"lifecycle_status_messages": False}
    created: dict[str, Any] = {}

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "Coordinator", lambda **kwargs: FakeCoordinator(**kwargs))

    def fake_adapter_factory(
        *, coordinator: Any, approval_mode: str, llm_config: dict[str, Any], **kwargs: Any
    ) -> FakeAdapter:
        del approval_mode
        del llm_config
        del kwargs
        created["coordinator"] = coordinator
        return FakeAdapter(coordinator=coordinator, approval_mode="review", llm_config={})

    monkeypatch.setattr(main, "CLIAdapter", fake_adapter_factory)

    main.main([])

    assert created["coordinator"].fire_lifecycle_status_messages is False


def test_main_rejects_unsupported_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _config()
    bad["adapters"] = {"enabled": ["discord"]}
    monkeypatch.setattr(main, "load_config", lambda: bad)
    with pytest.raises(ValueError, match="Unsupported adapter"):
        main.main([])


def test_main_loads_resume_state_and_passes_to_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: dict[str, Any] = {}
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(main, "load_config", _config)

    session_dir = tmp_path / ".strangeclaw" / "sessions" / "abc-1"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "state.json").write_text('{"goal":"resume goal"}', encoding="utf-8")

    def fake_coordinator_factory(**kwargs: Any) -> FakeCoordinator:
        return FakeCoordinator(**kwargs)

    def fake_adapter_factory(
        *, coordinator: Any, approval_mode: str, llm_config: dict[str, Any], **kwargs: Any
    ) -> FakeAdapter:
        del coordinator
        del approval_mode
        del llm_config
        created.update(kwargs)
        return FakeAdapter(coordinator=object(), approval_mode="review", llm_config={})

    monkeypatch.setattr(main, "Coordinator", fake_coordinator_factory)
    monkeypatch.setattr(main, "CLIAdapter", fake_adapter_factory)

    main.main(["--resume", "abc-1"])

    assert created["resume_session_id"] == "abc-1"
    assert created["resume_state"] == {"goal": "resume goal"}


def test_main_wires_telegram_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    cfg["adapters"] = {"enabled": ["telegram"]}
    created: dict[str, Any] = {}

    monkeypatch.setattr(main, "load_config", lambda: cfg)
    monkeypatch.setattr(main, "Coordinator", lambda **kwargs: FakeCoordinator(**kwargs))

    def fake_telegram_factory(**kwargs: Any) -> FakeTelegramAdapter:
        adapter = FakeTelegramAdapter(**kwargs)
        created["adapter"] = adapter
        return adapter

    monkeypatch.setattr(main, "TelegramAdapter", fake_telegram_factory)
    main.main([])

    adapter = created["adapter"]
    assert adapter.run_called is True
    assert adapter.approval_mode == "review"
    assert adapter.llm_config == {"model": "x", "api_key": "k"}
    assert adapter.token == "bot-token"
    assert adapter.allowed_chat_ids == [123]
    assert adapter.limits.max_active_sessions == 8
    assert adapter.session_id_prefix == ""
    assert adapter.coordinator is not None
    sandbox = adapter.sandbox_factory()
    assert sandbox._skills_dir == "./skills"  # noqa: SLF001


def test_main_rejects_resume_with_telegram(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    cfg["adapters"] = {"enabled": ["telegram"]}
    monkeypatch.setattr(main, "load_config", lambda: cfg)
    with pytest.raises(ValueError, match="Resume is only supported"):
        main.main(["--resume", "abc-1"])


def test_main_rejects_non_local_telegram_without_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    cfg["adapters"] = {"enabled": ["telegram"]}
    cfg["telegram"]["local_mode"] = False
    cfg["telegram"]["allowed_chat_ids"] = []
    monkeypatch.setattr(main, "load_config", lambda: cfg)
    with pytest.raises(ValueError, match="telegram.allowed_chat_ids must be configured"):
        main.main([])


def test_main_runs_cli_and_telegram_when_multiple_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    cfg["adapters"] = {"enabled": ["cli", "telegram"]}
    created: dict[str, Any] = {}

    monkeypatch.setattr(main, "load_config", lambda: cfg)

    def fake_sandbox_factory(**kwargs: Any) -> FakeSandbox:
        sandbox = FakeSandbox(**kwargs)
        created["sandbox"] = sandbox
        return sandbox

    def fake_coordinator_factory(**kwargs: Any) -> FakeCoordinator:
        coordinator = FakeCoordinator(**kwargs)
        created["coordinator"] = coordinator
        return coordinator

    def fake_cli_factory(
        *, coordinator: Any, approval_mode: str, llm_config: dict[str, Any], **kwargs: Any
    ) -> FakeAdapter:
        adapter = FakeAdapter(
            coordinator=coordinator,
            approval_mode=approval_mode,
            llm_config=llm_config,
            **kwargs,
        )
        created["cli"] = adapter
        return adapter

    def fake_telegram_factory(**kwargs: Any) -> FakeTelegramAdapter:
        adapter = FakeTelegramAdapter(**kwargs)
        created["telegram"] = adapter
        return adapter

    monkeypatch.setattr(main, "YoloSandbox", fake_sandbox_factory)
    monkeypatch.setattr(main, "Coordinator", fake_coordinator_factory)
    monkeypatch.setattr(main, "CLIAdapter", fake_cli_factory)
    monkeypatch.setattr(main, "TelegramAdapter", fake_telegram_factory)

    main.main([])

    cli = created["cli"]
    telegram = created["telegram"]
    assert cli.run_called is True
    assert telegram.run_called is True
    assert telegram.session_id_prefix == "telegram-"
    assert cli.coordinator is created["coordinator"]
    assert telegram.coordinator is created["coordinator"]


def test_main_rejects_resume_when_multiple_adapters_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _config()
    cfg["adapters"] = {"enabled": ["cli", "telegram"]}
    monkeypatch.setattr(main, "load_config", lambda: cfg)
    with pytest.raises(ValueError, match="Resume cannot be used"):
        main.main(["--resume", "abc-1"])


def test_main_rejects_missing_adapters_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _config()
    del cfg["adapters"]
    monkeypatch.setattr(main, "load_config", lambda: cfg)
    with pytest.raises(ValueError, match="adapters must be a mapping"):
        main.main([])
