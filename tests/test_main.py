"""Tests for main entrypoint wiring."""

from __future__ import annotations

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
        sandbox: FakeSandbox,
        approval_mode: str,
        llm_config: dict[str, Any],
    ) -> None:
        self.sandbox = sandbox
        self.approval_mode = approval_mode
        self.llm_config = llm_config
        self.run_called = False
        self.raise_interrupt = False

    def run(self) -> None:
        self.run_called = True
        if self.raise_interrupt:
            raise KeyboardInterrupt()


def _config() -> dict[str, Any]:
    return {
        "mode": "yolo",
        "adapter": "cli",
        "approval_mode": "review",
        "llm": {"model": "x", "api_key": "k"},
        "skills": {"directory": "./skills"},
        "loop": {"max_iterations": 7},
        "context": {"token_budget": 1234, "summary_threshold": 9},
    }


def test_main_wires_config_to_sandbox_and_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    def fake_load_config() -> dict[str, Any]:
        return _config()

    def fake_sandbox_factory(**kwargs: Any) -> FakeSandbox:
        sandbox = FakeSandbox(**kwargs)
        created["sandbox"] = sandbox
        return sandbox

    def fake_adapter_factory(
        *, sandbox: FakeSandbox, approval_mode: str, llm_config: dict[str, Any]
    ) -> FakeAdapter:
        adapter = FakeAdapter(sandbox=sandbox, approval_mode=approval_mode, llm_config=llm_config)
        created["adapter"] = adapter
        return adapter

    monkeypatch.setattr(main, "load_config", fake_load_config)
    monkeypatch.setattr(main, "YoloSandbox", fake_sandbox_factory)
    monkeypatch.setattr(main, "CLIAdapter", fake_adapter_factory)

    main.main()

    sandbox = created["sandbox"]
    adapter = created["adapter"]
    assert sandbox.kwargs["skills_dir"] == "./skills"
    assert sandbox.kwargs["max_iterations"] == 7
    assert sandbox.kwargs["token_budget"] == 1234
    assert sandbox.kwargs["summary_threshold"] == 9
    assert adapter.approval_mode == "review"
    assert adapter.llm_config == {"model": "x", "api_key": "k"}
    assert adapter.run_called is True


def test_main_stops_sandbox_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    monkeypatch.setattr(main, "load_config", _config)

    def fake_sandbox_factory(**kwargs: Any) -> FakeSandbox:
        del kwargs
        sandbox = FakeSandbox()
        created["sandbox"] = sandbox
        return sandbox

    def fake_adapter_factory(
        *, sandbox: FakeSandbox, approval_mode: str, llm_config: dict[str, Any]
    ) -> FakeAdapter:
        del approval_mode
        del llm_config
        adapter = FakeAdapter(sandbox=sandbox, approval_mode="review", llm_config={})
        adapter.raise_interrupt = True
        return adapter

    monkeypatch.setattr(main, "YoloSandbox", fake_sandbox_factory)
    monkeypatch.setattr(main, "CLIAdapter", fake_adapter_factory)

    main.main()

    assert created["sandbox"].stop_called is True


def test_main_rejects_unsupported_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _config()
    bad["mode"] = "fire"
    monkeypatch.setattr(main, "load_config", lambda: bad)
    with pytest.raises(ValueError, match="Unsupported mode"):
        main.main()


def test_main_rejects_unsupported_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _config()
    bad["adapter"] = "telegram"
    monkeypatch.setattr(main, "load_config", lambda: bad)
    with pytest.raises(ValueError, match="Unsupported adapter"):
        main.main()
