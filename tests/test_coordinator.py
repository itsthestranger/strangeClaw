"""Tests for host-level Coordinator orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from coordinator import Coordinator


class FakeSandbox:
    """Sandbox stub for coordinator tests."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = list(events)
        self.started_task: dict[str, Any] | None = None
        self.sent: list[dict[str, Any]] = []
        self.stop_calls = 0

    def run(self, task: dict[str, Any]) -> None:
        self.started_task = task

    def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        del timeout_seconds
        if not self._events:
            return None
        return self._events.pop(0)

    def stop(self) -> None:
        self.stop_calls += 1


def _wait_until(predicate: Any, timeout_seconds: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_coordinator_routes_plan_reply_and_done(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
        events=[
            {"type": "message", "role": "plan", "content": {"steps": ["one"]}},
            {"type": "done", "success": True, "reply": "ok", "state": {"goal": "g"}, "files": []},
        ]
    )
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
    )
    seen_events: list[dict[str, Any]] = []

    status = coordinator.start_task(session_id="sess-1", text="task", sink=seen_events.append)
    assert status == "started"
    assert _wait_until(lambda: coordinator.pending_role(session_id="sess-1") == "plan")

    submitted = coordinator.submit_reply(session_id="sess-1", approved=True, text="")
    assert submitted is True
    assert _wait_until(lambda: any(event.get("type") == "done" for event in seen_events))
    assert sandbox.sent[-1] == {"type": "user_reply", "approved": True, "text": ""}


def test_coordinator_follow_up_uses_saved_state_without_plan(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first = FakeSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "first",
                "state": {
                    "goal": "g",
                    "plan": {"steps": ["old"]},
                    "history": [{"type": "action"}],
                },
                "files": [],
            }
        ]
    )
    second = FakeSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "second",
                "state": {"goal": "g2", "history": []},
                "files": [],
            }
        ]
    )
    sandboxes = [first, second]
    coordinator = Coordinator(
        sandbox_factory=lambda: sandboxes.pop(0),
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
    )

    status_one = coordinator.start_task(session_id="sess-1", text="first", sink=lambda _: None)
    assert status_one == "started"
    assert _wait_until(lambda: first.stop_calls > 0)

    status_two = coordinator.start_task(session_id="sess-1", text="second", sink=lambda _: None)
    assert status_two == "started"
    assert _wait_until(lambda: second.stop_calls > 0)

    assert first.started_task is not None
    assert second.started_task is not None
    assert second.started_task["state"] == {"goal": "g", "history": [{"type": "action"}]}
    assert "plan" not in second.started_task["state"]


def test_coordinator_reports_busy_for_running_session() -> None:
    sandbox = FakeSandbox(events=[{"type": "message", "role": "plan", "content": {"steps": []}}])
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
    )
    status_one = coordinator.start_task(session_id="sess-1", text="first", sink=lambda _: None)
    status_two = coordinator.start_task(session_id="sess-1", text="second", sink=lambda _: None)

    assert status_one == "started"
    assert status_two == "busy"
    coordinator.stop_all()
