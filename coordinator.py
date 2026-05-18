"""Host-level coordination for multi-adapter session workers."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from adapters.session_persistence import (
    append_session_event_journal,
    persist_done_event,
    state_for_follow_up,
)

ReplyRole = Literal["plan", "clarification"]
StartStatus = Literal["started", "busy", "capacity"]
EventSink = Callable[[dict[str, Any]], None]
SandboxFactory = Callable[[], Any]


@dataclass
class _SessionRecord:
    worker: _SessionWorker | None = None
    pending_role: ReplyRole | None = None
    latest_state: dict[str, Any] | None = None
    sink: EventSink | None = None
    sandbox: Any | None = None
    last_task_completed_at: float | None = None


class Coordinator:
    """Single control-plane that owns all active session workers."""

    def __init__(
        self,
        *,
        sandbox_factory: SandboxFactory,
        approval_mode: str,
        llm_config: dict[str, Any] | None = None,
        max_active_sessions: int | None = None,
        session_journal: dict[str, Any] | None = None,
        fire_lifecycle_status_messages: bool = True,
        session_idle_timeout_seconds: int = 1800,
        _idle_reaper_interval_seconds: float = 60.0,
    ) -> None:
        if max_active_sessions is not None and max_active_sessions <= 0:
            raise ValueError("max_active_sessions must be positive when set.")
        if isinstance(session_idle_timeout_seconds, bool):
            raise ValueError("session_idle_timeout_seconds must be a non-negative integer.")
        if session_idle_timeout_seconds < 0:
            raise ValueError("session_idle_timeout_seconds must be a non-negative integer.")
        if _idle_reaper_interval_seconds <= 0:
            raise ValueError("_idle_reaper_interval_seconds must be greater than zero.")
        self._sandbox_factory = sandbox_factory
        self._approval_mode = approval_mode
        self._llm_config = dict(llm_config) if llm_config is not None else None
        self._max_active_sessions = max_active_sessions
        journal = dict(session_journal) if session_journal is not None else {}
        self._journal_enabled = bool(journal.get("enabled", False))
        self._journal_max_bytes = int(journal.get("max_bytes", 1 * 1024 * 1024))
        self._fire_lifecycle_status_messages = bool(fire_lifecycle_status_messages)
        self._session_idle_timeout_seconds = session_idle_timeout_seconds
        self._idle_reaper_interval_seconds = _idle_reaper_interval_seconds
        self._lock = threading.Lock()
        self._sessions: dict[str, _SessionRecord] = {}
        self._idle_reaper_stop = threading.Event()
        self._idle_reaper_thread = threading.Thread(
            target=self._run_idle_reaper,
            name="coordinator-idle-reaper",
            daemon=True,
        )
        self._idle_reaper_thread.start()

    def seed_state(self, *, session_id: str, state: dict[str, Any]) -> None:
        """Seed stored state for a session (used by resume flows)."""
        with self._lock:
            record = self._sessions.setdefault(session_id, _SessionRecord())
            record.latest_state = dict(state)

    def start_task(
        self,
        *,
        session_id: str,
        text: str,
        sink: EventSink,
    ) -> StartStatus:
        """Start a session worker task or report why it could not be started."""
        if not text.strip():
            raise ValueError("Task text cannot be empty.")

        start_error: Exception | None = None
        with self._lock:
            record = self._sessions.setdefault(session_id, _SessionRecord())
            if record.worker is not None and record.worker.is_running():
                return "busy"
            if (
                self._max_active_sessions is not None
                and self._active_session_count_locked() >= self._max_active_sessions
            ):
                return "capacity"

            task: dict[str, Any] = {
                "type": "task",
                "text": text.strip(),
                "session_id": session_id,
                "approval_mode": self._approval_mode,
            }
            if record.latest_state is not None:
                task["state"] = state_for_follow_up(record.latest_state)

            def on_event(event: dict[str, Any], *, sid: str = session_id) -> None:
                self._handle_worker_event(session_id=sid, event=event)

            def on_exit(*, sid: str = session_id) -> None:
                self._handle_worker_exit(session_id=sid)

            def on_error(error: Exception, *, sid: str = session_id) -> None:
                self._handle_worker_error(session_id=sid, error=error)

            record.sink = sink
            sandbox_started = False
            sandbox = record.sandbox
            try:
                if sandbox is None or not sandbox.is_running():
                    sandbox = self._sandbox_factory()
                    sandbox.start(session_id=session_id)
                    sandbox_started = True
                    record.sandbox = sandbox

                sandbox.send_task(task)

                worker = _SessionWorker(
                    sandbox=sandbox,
                    task_event=task,
                    on_event=on_event,
                    on_exit=on_exit,
                    on_error=on_error,
                    fire_lifecycle_status_messages=self._fire_lifecycle_status_messages,
                    sandbox_started_for_task=sandbox_started,
                )
                record.worker = worker
                record.pending_role = None
                worker.start()
                return "started"
            except Exception as exc:
                start_error = exc
                if sandbox_started and sandbox is not None:
                    try:
                        sandbox.stop()
                    except Exception:
                        pass
                    if record.sandbox is sandbox:
                        record.sandbox = None
                record.worker = None
                record.pending_role = None

        if start_error is not None:
            self._handle_worker_error(session_id=session_id, error=start_error)
            return "started"
        return "started"

    def pending_role(self, *, session_id: str) -> ReplyRole | None:
        """Return current pending reply role for a session, if any."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            return record.pending_role

    def submit_reply(
        self,
        *,
        session_id: str,
        approved: bool,
        text: str,
    ) -> bool:
        """Submit user reply for a waiting worker. Returns False if not waiting."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None or record.worker is None:
                return False
            if record.pending_role is None:
                return False
            worker = record.worker
            record.pending_role = None
        worker.submit_reply({"approved": approved, "text": text})
        return True

    def stop_session(self, *, session_id: str) -> None:
        """Stop a single session worker if active."""
        worker: _SessionWorker | None = None
        sandbox: Any | None = None
        with self._lock:
            record = self._sessions.get(session_id)
            if record is not None:
                worker = record.worker
                sandbox = record.sandbox
                record.pending_role = None
                record.worker = None
                record.sandbox = None
                record.last_task_completed_at = None
        if worker is not None:
            worker.stop()
        if sandbox is not None:
            try:
                sandbox.stop()
            except Exception:
                pass

    def stop_all(self) -> None:
        """Stop all session workers."""
        self._idle_reaper_stop.set()
        workers: list[_SessionWorker] = []
        sandboxes: list[Any] = []
        with self._lock:
            for record in self._sessions.values():
                if record.worker is not None:
                    workers.append(record.worker)
                    record.worker = None
                if record.sandbox is not None:
                    sandboxes.append(record.sandbox)
                    record.sandbox = None
                record.pending_role = None
                record.last_task_completed_at = None
        for worker in workers:
            worker.stop()
        for sandbox in sandboxes:
            try:
                sandbox.stop()
            except Exception:
                pass
        self._idle_reaper_thread.join(timeout=2.0)

    def _active_session_count_locked(self) -> int:
        count = 0
        for record in self._sessions.values():
            if record.worker is not None and record.worker.is_running():
                count += 1
        return count

    def _handle_worker_event(self, *, session_id: str, event: dict[str, Any]) -> None:
        sink: EventSink | None = None
        append_session_event_journal(
            session_id=session_id,
            event=event,
            enabled=self._journal_enabled,
            max_bytes=self._journal_max_bytes,
        )
        with self._lock:
            record = self._sessions.setdefault(session_id, _SessionRecord())
            if event.get("type") == "message":
                role = event.get("role")
                if role == "plan" and self._approval_mode == "review":
                    record.pending_role = "plan"
                elif role == "clarification":
                    record.pending_role = "clarification"
            if event.get("type") == "done":
                record.pending_role = None
                state = persist_done_event(session_id=session_id, done_event=event)
                if isinstance(state, dict):
                    record.latest_state = state
                record.last_task_completed_at = time.monotonic()
            sink = record.sink

        if sink is None:
            return
        try:
            sink(event)
        except Exception:
            return

    def _handle_worker_error(self, *, session_id: str, error: Exception) -> None:
        error_text = f"Sandbox runtime error: {error}"
        self._handle_worker_event(
            session_id=session_id,
            event={
                "type": "message",
                "role": "status",
                "content": error_text,
            },
        )
        self._handle_worker_event(
            session_id=session_id,
            event={
                "type": "done",
                "success": False,
                "reply": error_text,
                "state": {},
                "files": [],
            },
        )

    def _handle_worker_exit(self, *, session_id: str) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is not None:
                record.worker = None
                record.pending_role = None

    def _run_idle_reaper(self) -> None:
        while not self._idle_reaper_stop.wait(self._idle_reaper_interval_seconds):
            timeout_seconds = self._session_idle_timeout_seconds
            if timeout_seconds == 0:
                continue

            now = time.monotonic()
            sandboxes_to_stop: list[Any] = []
            with self._lock:
                for record in self._sessions.values():
                    sandbox = record.sandbox
                    if sandbox is None:
                        continue

                    worker = record.worker
                    if worker is not None and worker.is_running():
                        continue

                    completed_at = record.last_task_completed_at
                    if completed_at is None:
                        continue
                    if now - completed_at <= timeout_seconds:
                        continue

                    if not sandbox.is_running():
                        record.sandbox = None
                        record.last_task_completed_at = None
                        continue

                    record.sandbox = None
                    record.last_task_completed_at = None
                    sandboxes_to_stop.append(sandbox)

            for sandbox in sandboxes_to_stop:
                try:
                    sandbox.stop()
                except Exception:
                    pass


class _SessionWorker:
    """One active agent run bound to one sandbox and one session."""

    def __init__(
        self,
        *,
        sandbox: Any,
        task_event: dict[str, Any],
        on_event: Callable[[dict[str, Any]], None],
        on_exit: Callable[[], None],
        on_error: Callable[[Exception], None],
        fire_lifecycle_status_messages: bool = True,
        sandbox_started_for_task: bool = False,
    ) -> None:
        self._sandbox = sandbox
        self._task_event = task_event
        self._on_event = on_event
        self._on_exit = on_exit
        self._on_error = on_error
        self._fire_lifecycle_status_messages = fire_lifecycle_status_messages
        self._sandbox_started_for_task = sandbox_started_for_task
        self._reply_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread.is_alive()

    def submit_reply(self, reply: dict[str, Any]) -> None:
        self._reply_queue.put(reply)

    def stop(self) -> None:
        self._stop_event.set()
        self._reply_queue.put(None)
        try:
            self._sandbox.send({"type": "stop"})
        except Exception:
            pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            if self._sandbox_started_for_task:
                self._emit_fire_status_message("Firecracker sandbox started successfully.")
            while not self._stop_event.is_set():
                event = self._sandbox.receive(timeout_seconds=0.2)
                if event is None:
                    continue

                if event.get("type") == "done":
                    self._on_event(event)
                    return

                self._on_event(event)

                if event.get("type") != "message":
                    continue

                role = event.get("role")
                if role == "plan" and self._task_event.get("approval_mode") == "review":
                    reply = self._wait_for_reply()
                    if reply is None:
                        return
                    self._sandbox.send({"type": "user_reply", **reply})
                    continue

                if role == "clarification":
                    reply = self._wait_for_reply()
                    if reply is None:
                        return
                    self._sandbox.send({"type": "user_reply", **reply})
                    continue
        except Exception as exc:
            self._on_error(exc)
        finally:
            self._on_exit()

    def _wait_for_reply(self) -> dict[str, Any] | None:
        while not self._stop_event.is_set():
            try:
                reply = self._reply_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            return reply
        return None

    def _is_fire_sandbox(self) -> bool:
        return bool(self._sandbox.__class__.__name__ == "FireSandbox")

    def _emit_fire_status_message(self, content: str) -> None:
        if not self._fire_lifecycle_status_messages:
            return
        if not self._is_fire_sandbox():
            return
        self._on_event(
            {
                "type": "message",
                "role": "status",
                "content": content,
            }
        )
