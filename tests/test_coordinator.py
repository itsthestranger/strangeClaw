"""Tests for host-level Coordinator orchestration."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from coordinator import Coordinator
from sandbox.fire import tap_name_for_session


class FakeSandbox:
    """Sandbox stub for coordinator tests."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = list(events)
        self.started_tasks: list[dict[str, Any]] = []
        self.start_session_ids: list[str | None] = []
        self.sent: list[dict[str, Any]] = []
        self.stop_calls = 0
        self.start_calls = 0
        self.send_task_calls = 0
        self._running = False

    def start(self, session_id: str | None = None) -> None:
        self.start_calls += 1
        self.start_session_ids.append(session_id)
        self._running = True

    def is_running(self) -> bool:
        return self._running

    def send_task(self, task: dict[str, Any]) -> None:
        self.send_task_calls += 1
        self.started_tasks.append(task)

    def send(self, event: dict[str, Any]) -> None:
        self.sent.append(event)

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        del timeout_seconds
        if not self._events:
            return None
        return self._events.pop(0)

    def stop(self) -> None:
        self.stop_calls += 1
        self._running = False


class FireSandbox(FakeSandbox):
    """Fire sandbox stub used to trigger Fire-specific coordinator behavior."""


class FailingSandbox(FakeSandbox):
    """Sandbox stub that fails on send_task()."""

    def send_task(self, task: dict[str, Any]) -> None:
        del task
        raise RuntimeError("boom")


class BlockingStopFireSandbox(FireSandbox):
    """Fire sandbox stub whose stop() blocks until explicitly released."""

    def __init__(
        self,
        events: list[dict[str, Any]],
        *,
        stop_entered: threading.Event,
        stop_release: threading.Event,
        stop_finished: threading.Event,
    ) -> None:
        super().__init__(events)
        self._stop_entered = stop_entered
        self._stop_release = stop_release
        self._stop_finished = stop_finished

    def stop(self) -> None:
        self.stop_calls += 1
        self._stop_entered.set()
        self._stop_release.wait(timeout=2.0)
        self._running = False
        self._stop_finished.set()


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
    assert sandbox.start_session_ids == ["sess-1"]
    coordinator.stop_all()


def test_coordinator_passes_session_id_to_sandbox_start(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
        events=[
            {"type": "done", "success": True, "reply": "ok", "state": {"goal": "g"}, "files": []}
        ]
    )
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
    )

    status = coordinator.start_task(session_id="session-xyz", text="task", sink=lambda _: None)
    assert status == "started"
    assert _wait_until(lambda: sandbox.start_calls == 1)
    assert sandbox.start_session_ids == ["session-xyz"]
    coordinator.stop_all()


def test_coordinator_concurrent_sessions_use_distinct_fire_identity() -> None:
    first = FireSandbox(events=[{"type": "message", "role": "status", "content": "running"}])
    second = FireSandbox(events=[{"type": "message", "role": "status", "content": "running"}])
    sandboxes = [first, second]
    coordinator = Coordinator(
        sandbox_factory=lambda: sandboxes.pop(0),
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
    )

    status_one = coordinator.start_task(session_id="fire-a", text="first", sink=lambda _: None)
    status_two = coordinator.start_task(session_id="fire-b", text="second", sink=lambda _: None)

    assert status_one == "started"
    assert status_two == "started"
    assert _wait_until(lambda: first.start_calls == 1 and second.start_calls == 1)
    assert first.start_session_ids == ["fire-a"]
    assert second.start_session_ids == ["fire-b"]
    assert tap_name_for_session(first.start_session_ids[0] or "") != tap_name_for_session(
        second.start_session_ids[0] or ""
    )
    coordinator.stop_all()


def test_coordinator_follow_up_uses_saved_state_without_plan(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
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
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
    )
    first_seen: list[dict[str, Any]] = []

    status_one = coordinator.start_task(session_id="sess-1", text="first", sink=first_seen.append)
    assert status_one == "started"
    assert _wait_until(lambda: any(event.get("type") == "done" for event in first_seen))

    sandbox._events.append(
        {
            "type": "done",
            "success": True,
            "reply": "second",
            "state": {"goal": "g2", "history": []},
            "files": [],
        }
    )
    status_two = coordinator.start_task(session_id="sess-1", text="second", sink=lambda _: None)
    assert status_two == "started"
    assert _wait_until(lambda: len(sandbox.started_tasks) == 2)

    assert sandbox.start_calls == 1
    second_task = sandbox.started_tasks[1]
    assert second_task["state"] == {"goal": "g", "history": [{"type": "action"}]}
    assert "plan" not in second_task["state"]


def test_coordinator_worker_exit_does_not_stop_session_sandbox(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
        events=[
            {"type": "done", "success": True, "reply": "ok", "state": {"goal": "g"}, "files": []}
        ]
    )
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
    )

    status = coordinator.start_task(session_id="sess-1", text="task", sink=lambda _: None)
    assert status == "started"
    assert _wait_until(lambda: sandbox.send_task_calls == 1)
    assert _wait_until(lambda: sandbox.stop_calls == 0)
    assert sandbox.is_running() is True
    coordinator.stop_all()


def test_coordinator_stop_session_stops_session_sandbox(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(events=[])
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
    )

    status = coordinator.start_task(session_id="sess-1", text="task", sink=lambda _: None)
    assert status == "started"
    assert sandbox.is_running() is True

    coordinator.stop_session(session_id="sess-1")
    assert sandbox.stop_calls == 1
    assert sandbox.is_running() is False
    coordinator.stop_all()


def test_coordinator_starts_fresh_sandbox_after_stop_session(
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
                "state": {"goal": "g", "history": []},
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

    assert (
        coordinator.start_task(session_id="sess-1", text="first", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(lambda: first.send_task_calls == 1)
    coordinator.stop_session(session_id="sess-1")
    assert first.stop_calls == 1

    assert (
        coordinator.start_task(session_id="sess-1", text="second", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(lambda: second.send_task_calls == 1)
    assert second.start_calls == 1


def test_coordinator_restarts_sandbox_when_not_running_between_tasks(
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
                "state": {"goal": "g", "history": []},
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

    assert (
        coordinator.start_task(session_id="sess-1", text="first", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(lambda: first.send_task_calls == 1)
    first._running = False

    status = "busy"
    deadline = time.monotonic() + 2.0
    while status == "busy" and time.monotonic() < deadline:
        status = coordinator.start_task(session_id="sess-1", text="second", sink=lambda _: None)
        if status == "busy":
            time.sleep(0.01)
    assert status == "started"
    assert _wait_until(lambda: second.send_task_calls == 1)
    assert _wait_until(
        lambda: coordinator._sessions["sess-1"].worker is None  # type: ignore[attr-defined]
    )
    coordinator.stop_all()


def test_coordinator_replacing_dead_sandbox_does_not_stop_old_instance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first = FireSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "first",
                "state": {"goal": "g", "history": []},
                "files": [],
            }
        ]
    )
    second = FireSandbox(
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

    assert (
        coordinator.start_task(session_id="sess-1", text="first", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(
        lambda: coordinator._sessions["sess-1"].worker is None  # type: ignore[attr-defined]
    )

    # Simulate dead VM between tasks; coordinator should replace it.
    first._running = False  # noqa: SLF001
    assert (
        coordinator.start_task(session_id="sess-1", text="second", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(lambda: second.send_task_calls == 1)
    first_stop_calls_after_replacement = first.stop_calls
    coordinator.stop_all()

    # Expected safe behavior: old sandbox should have been stopped before replacement.
    # Current behavior leaves this at 0, proving the leak gap.
    assert first_stop_calls_after_replacement == 1


def test_coordinator_can_start_new_sandbox_before_old_stop_finishes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    stop_entered = threading.Event()
    stop_release = threading.Event()
    stop_finished = threading.Event()
    first = BlockingStopFireSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "first",
                "state": {"goal": "g", "history": []},
                "files": [],
            }
        ],
        stop_entered=stop_entered,
        stop_release=stop_release,
        stop_finished=stop_finished,
    )
    second = FireSandbox(
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

    assert (
        coordinator.start_task(session_id="sess-1", text="first", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(
        lambda: coordinator._sessions["sess-1"].worker is None  # type: ignore[attr-defined]
    )

    stop_thread = threading.Thread(
        target=coordinator.stop_session,
        kwargs={"session_id": "sess-1"},
        daemon=True,
    )
    stop_thread.start()
    assert stop_entered.wait(timeout=1.0) is True
    assert first.is_running() is True

    # While stop is in progress, session startup is deferred.
    status_during_stop = coordinator.start_task(
        session_id="sess-1",
        text="second",
        sink=lambda _: None,
    )
    assert status_during_stop == "busy"
    first_running_during_replacement = first.is_running()
    second_start_calls_during_replacement = second.start_calls

    stop_release.set()
    stop_thread.join(timeout=1.0)
    assert stop_finished.is_set()

    # Once stop completes, task start succeeds on a fresh sandbox.
    status_after_stop = coordinator.start_task(
        session_id="sess-1",
        text="second",
        sink=lambda _: None,
    )
    assert status_after_stop == "started"
    assert _wait_until(lambda: second.start_calls == 1)
    coordinator.stop_all()

    assert first_running_during_replacement is True
    assert second_start_calls_during_replacement == 0


def test_coordinator_reaps_idle_session_sandbox_after_timeout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
        events=[
            {"type": "done", "success": True, "reply": "ok", "state": {"goal": "g"}, "files": []}
        ]
    )
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        session_idle_timeout_seconds=1,
        _idle_reaper_interval_seconds=0.02,
    )

    assert (
        coordinator.start_task(session_id="sess-1", text="task", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(lambda: sandbox.send_task_calls == 1)
    assert _wait_until(
        lambda: coordinator._sessions["sess-1"].worker is None  # type: ignore[attr-defined]
    )
    with coordinator._lock:  # type: ignore[attr-defined]
        coordinator._sessions["sess-1"].last_task_completed_at = time.monotonic() - 5.0  # type: ignore[attr-defined]

    assert _wait_until(lambda: sandbox.stop_calls == 1)
    assert sandbox.is_running() is False
    coordinator.stop_all()


def test_coordinator_timeout_zero_disables_idle_reaper(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
        events=[
            {"type": "done", "success": True, "reply": "ok", "state": {"goal": "g"}, "files": []}
        ]
    )
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        session_idle_timeout_seconds=0,
        _idle_reaper_interval_seconds=0.02,
    )

    assert (
        coordinator.start_task(session_id="sess-1", text="task", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(lambda: sandbox.send_task_calls == 1)
    assert _wait_until(
        lambda: coordinator._sessions["sess-1"].worker is None  # type: ignore[attr-defined]
    )
    with coordinator._lock:  # type: ignore[attr-defined]
        coordinator._sessions["sess-1"].last_task_completed_at = time.monotonic() - 5.0  # type: ignore[attr-defined]

    time.sleep(0.1)
    assert sandbox.stop_calls == 0
    assert sandbox.is_running() is True
    coordinator.stop_all()


def test_coordinator_reaper_preserves_latest_state_across_vm_restart(
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
        session_idle_timeout_seconds=1,
        _idle_reaper_interval_seconds=0.02,
    )

    assert (
        coordinator.start_task(session_id="sess-1", text="first", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(lambda: first.send_task_calls == 1)
    assert _wait_until(
        lambda: coordinator._sessions["sess-1"].worker is None  # type: ignore[attr-defined]
    )
    with coordinator._lock:  # type: ignore[attr-defined]
        coordinator._sessions["sess-1"].last_task_completed_at = time.monotonic() - 5.0  # type: ignore[attr-defined]
    assert _wait_until(lambda: first.stop_calls == 1)

    status = "busy"
    deadline = time.monotonic() + 2.0
    while status == "busy" and time.monotonic() < deadline:
        status = coordinator.start_task(session_id="sess-1", text="second", sink=lambda _: None)
        if status == "busy":
            time.sleep(0.01)
    assert status == "started"
    assert _wait_until(lambda: second.send_task_calls == 1)
    second_task = second.started_tasks[0]
    assert second_task["state"] == {"goal": "g", "history": [{"type": "action"}]}
    assert "plan" not in second_task["state"]
    coordinator.stop_all()


def test_coordinator_reaper_does_not_stop_sandbox_with_active_worker() -> None:
    sandbox = FakeSandbox(events=[{"type": "message", "role": "status", "content": "working"}])
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        session_idle_timeout_seconds=1,
        _idle_reaper_interval_seconds=0.02,
    )

    assert (
        coordinator.start_task(session_id="sess-1", text="task", sink=lambda _: None)
        == "started"
    )
    with coordinator._lock:  # type: ignore[attr-defined]
        coordinator._sessions["sess-1"].last_task_completed_at = time.monotonic() - 5.0  # type: ignore[attr-defined]

    time.sleep(0.1)
    assert sandbox.stop_calls == 0
    coordinator.stop_all()


def test_coordinator_stop_all_joins_idle_reaper_thread() -> None:
    coordinator = Coordinator(
        sandbox_factory=lambda: FakeSandbox(events=[]),
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        session_idle_timeout_seconds=1,
        _idle_reaper_interval_seconds=0.02,
    )
    assert coordinator._idle_reaper_thread.is_alive() is True  # type: ignore[attr-defined]
    coordinator.stop_all()
    assert coordinator._idle_reaper_thread.is_alive() is False  # type: ignore[attr-defined]


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


def test_coordinator_capacity_counts_idle_fire_sandboxes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    first = FireSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "first",
                "state": {"goal": "g"},
                "files": [],
            }
        ]
    )
    second = FireSandbox(events=[])
    sandboxes = [first, second]
    coordinator = Coordinator(
        sandbox_factory=lambda: sandboxes.pop(0),
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        max_active_sessions=1,
    )

    assert (
        coordinator.start_task(session_id="fire-1", text="first", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(
        lambda: coordinator._sessions["fire-1"].worker is None  # type: ignore[attr-defined]
    )
    assert first.is_running() is True

    assert (
        coordinator.start_task(session_id="fire-2", text="second", sink=lambda _: None)
        == "capacity"
    )
    assert second.start_calls == 0
    coordinator.stop_all()


def test_coordinator_capacity_allows_existing_fire_session_continuation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FireSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "first",
                "state": {"goal": "g"},
                "files": [],
            }
        ]
    )
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        max_active_sessions=1,
    )

    assert (
        coordinator.start_task(session_id="fire-1", text="first", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(
        lambda: coordinator._sessions["fire-1"].worker is None  # type: ignore[attr-defined]
    )
    assert sandbox.is_running() is True
    sandbox._events.append(  # noqa: SLF001
        {
            "type": "done",
            "success": True,
            "reply": "second",
            "state": {"goal": "g"},
            "files": [],
        }
    )

    assert (
        coordinator.start_task(session_id="fire-1", text="follow-up", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(lambda: sandbox.send_task_calls == 2)
    coordinator.stop_all()


def test_coordinator_capacity_keeps_yolo_worker_based_semantics(
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
                "state": {"goal": "g"},
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
                "state": {"goal": "g"},
                "files": [],
            }
        ]
    )
    sandboxes = [first, second]
    coordinator = Coordinator(
        sandbox_factory=lambda: sandboxes.pop(0),
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        max_active_sessions=1,
    )

    assert (
        coordinator.start_task(session_id="yolo-1", text="first", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(
        lambda: coordinator._sessions["yolo-1"].worker is None  # type: ignore[attr-defined]
    )
    assert first.is_running() is True

    assert (
        coordinator.start_task(session_id="yolo-2", text="second", sink=lambda _: None)
        == "started"
    )
    assert _wait_until(lambda: second.send_task_calls == 1)
    coordinator.stop_all()


def test_coordinator_writes_redacted_bounded_session_journal(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    secret = "sk-secret-123"
    sandbox = FakeSandbox(
        events=[
            {
                "type": "message",
                "role": "status",
                "content": {"api_key": secret, "Authorization": "Bearer x"},
            },
            {
                "type": "done",
                "success": True,
                "reply": "ok",
                "state": {"goal": "g", "llm": {"api_key": secret}},
                "files": [],
            },
        ]
    )
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        session_journal={"enabled": True, "max_bytes": 300},
    )
    seen_events: list[dict[str, Any]] = []

    status = coordinator.start_task(session_id="sess-1", text="task", sink=seen_events.append)
    assert status == "started"
    assert _wait_until(lambda: any(event.get("type") == "done" for event in seen_events))

    journal_path = tmp_path / ".strangeclaw" / "sessions" / "sess-1" / "events.jsonl"
    assert journal_path.exists()
    assert journal_path.stat().st_size <= 300

    content = journal_path.read_text(encoding="utf-8")
    assert secret not in content
    assert "[REDACTED]" in content
    lines = [line for line in content.splitlines() if line.strip()]
    assert lines
    parsed_last = json.loads(lines[-1])
    assert parsed_last["event"]["type"] == "done"


def test_coordinator_emits_done_on_sandbox_runtime_error(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FailingSandbox(events=[])
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        session_journal={"enabled": True, "max_bytes": 4096},
    )
    seen_events: list[dict[str, Any]] = []

    status = coordinator.start_task(session_id="sess-err", text="task", sink=seen_events.append)
    assert status == "started"
    assert _wait_until(lambda: any(event.get("type") == "done" for event in seen_events))

    done = [event for event in seen_events if event.get("type") == "done"][-1]
    assert done["success"] is False
    assert "Sandbox runtime error" in str(done["reply"])

    journal_path = tmp_path / ".strangeclaw" / "sessions" / "sess-err" / "events.jsonl"
    assert journal_path.exists()
    journal_text = journal_path.read_text(encoding="utf-8")
    assert "Sandbox runtime error" in journal_text


def test_coordinator_emits_fire_lifecycle_status_messages(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FireSandbox(
        events=[
            {"type": "done", "success": True, "reply": "ok", "state": {"goal": "g"}, "files": []}
        ]
    )
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
    )
    seen_events: list[dict[str, Any]] = []

    status = coordinator.start_task(session_id="sess-fire", text="task", sink=seen_events.append)
    assert status == "started"
    assert _wait_until(lambda: any(event.get("type") == "done" for event in seen_events))

    statuses = [
        event
        for event in seen_events
        if event.get("type") == "message" and event.get("role") == "status"
    ]
    assert [event.get("content") for event in statuses] == [
        "Firecracker sandbox started successfully."
    ]
    assert seen_events[-1].get("type") == "done"
    assert sandbox.stop_calls == 0


def test_coordinator_can_disable_fire_lifecycle_status_messages(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FireSandbox(
        events=[
            {"type": "done", "success": True, "reply": "ok", "state": {"goal": "g"}, "files": []}
        ]
    )
    coordinator = Coordinator(
        sandbox_factory=lambda: sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        fire_lifecycle_status_messages=False,
    )
    seen_events: list[dict[str, Any]] = []

    status = coordinator.start_task(
        session_id="sess-fire-off",
        text="task",
        sink=seen_events.append,
    )
    assert status == "started"
    assert _wait_until(lambda: any(event.get("type") == "done" for event in seen_events))

    statuses = [
        event
        for event in seen_events
        if event.get("type") == "message" and event.get("role") == "status"
    ]
    assert statuses == []
    assert seen_events[-1].get("type") == "done"
