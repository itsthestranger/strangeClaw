"""Tests for CLIAdapter."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from adapters.cli import CLIAdapter


class FakeSandbox:
    """Sandbox stub for CLI adapter tests."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.sent: list[dict[str, Any]] = []
        self.started_tasks: list[dict[str, Any]] = []
        self.stop_calls = 0

    def run(self, task: dict[str, Any]) -> None:
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


def test_get_task_includes_type_session() -> None:
    sandbox = FakeSandbox(events=[])
    adapter = CLIAdapter(
        sandbox=sandbox,
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        input_func=lambda _: "Do the thing",
    )

    task = adapter.get_task()
    assert task["type"] == "task"
    assert task["text"] == "Do the thing"
    assert task["approval_mode"] == "review"
    assert isinstance(task["session_id"], str)
    assert "llm" not in task


def test_get_task_rejects_empty_text() -> None:
    adapter = CLIAdapter(sandbox=FakeSandbox(events=[]), input_func=lambda _: "   ")
    with pytest.raises(ValueError, match="Task cannot be empty"):
        adapter.get_task()


def test_run_handles_plan_approval_and_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
        events=[
            {"type": "message", "role": "plan", "content": {"steps": ["one", "two"]}},
            {
                "type": "action",
                "tool": "shell",
                "args": {"command": "echo ok"},
                "result": {"exit_code": 0, "stdout": "", "stderr": ""},
            },
            {"type": "done", "success": True, "reply": "All good.", "state": {}, "files": []},
        ]
    )
    answers = iter(["Build it", "y", "/quit"])
    adapter = CLIAdapter(
        sandbox=sandbox,
        approval_mode="review",
        input_func=lambda _: next(answers),
    )

    adapter.run()

    assert len(sandbox.started_tasks) == 1
    assert sandbox.started_tasks[0]["approval_mode"] == "review"
    assert sandbox.sent == [{"type": "user_reply", "approved": True, "text": ""}]
    assert sandbox.stop_calls == 1

    output = capsys.readouterr().out
    assert "Plan:" in output
    assert "Action: shell (exit=0)" in output
    assert "Success: All good." in output


def test_run_handles_clarification_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
        events=[
            {"type": "message", "role": "clarification", "content": "Which port?"},
            {"type": "done", "success": True, "reply": "Done.", "state": {}, "files": []},
        ]
    )
    answers = iter(["Start task", "Use 8080", "/quit"])
    adapter = CLIAdapter(
        sandbox=sandbox,
        approval_mode="auto",
        input_func=lambda _: next(answers),
    )

    adapter.run()
    assert sandbox.sent == [{"type": "user_reply", "approved": True, "text": "Use 8080"}]
    assert sandbox.stop_calls == 1


def test_run_persists_done_state_and_output_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    file_content = b"artifact-content"
    sandbox = FakeSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "Finished.",
                "state": {"goal": "x", "history": [1, 2]},
                "files": [
                    {
                        "path": "nested/out.txt",
                        "content_b64": base64.b64encode(file_content).decode("ascii"),
                        "size_bytes": len(file_content),
                    }
                ],
            }
        ]
    )
    answers = iter(["Persist this", "/quit"])
    adapter = CLIAdapter(sandbox=sandbox, input_func=lambda _: next(answers))

    adapter.run()

    assert len(sandbox.started_tasks) == 1
    session_id = sandbox.started_tasks[0]["session_id"]
    state_path = tmp_path / ".strangeclaw" / "sessions" / session_id / "state.json"
    assert state_path.exists()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved == {"goal": "x", "history": [1, 2]}

    output_path = (
        tmp_path
        / ".strangeclaw"
        / "sessions"
        / session_id
        / "outputs"
        / "nested"
        / "out.txt"
    )
    assert output_path.read_bytes() == file_content
    assert sandbox.stop_calls == 1


def test_run_rejects_traversal_output_file_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    file_content = b"escape"
    sandbox = FakeSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "Finished.",
                "state": {"goal": "x", "history": []},
                "files": [
                    {
                        "path": "../escape.txt",
                        "content_b64": base64.b64encode(file_content).decode("ascii"),
                        "size_bytes": len(file_content),
                    }
                ],
            }
        ]
    )
    answers = iter(["Persist this"])
    adapter = CLIAdapter(sandbox=sandbox, input_func=lambda _: next(answers))

    with pytest.raises(ValueError, match="Invalid output file path"):
        adapter.run()
    assert sandbox.stop_calls == 1
    assert not (tmp_path / ".strangeclaw" / "sessions" / "escape.txt").exists()


def test_get_task_uses_resume_state_with_new_prompted_task() -> None:
    sandbox = FakeSandbox(events=[])
    adapter = CLIAdapter(
        sandbox=sandbox,
        resume_session_id="resume-1",
        resume_state={"goal": "resume-goal", "history": []},
        input_func=lambda _: "follow-up task",
    )
    task = adapter.get_task()
    assert task["session_id"] == "resume-1"
    assert task["text"] == "follow-up task"
    assert task["state"] == {"goal": "resume-goal", "history": []}


def test_run_keeps_same_session_id_across_follow_up_tasks(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "one",
                "state": {"history": []},
                "files": [],
            },
            {
                "type": "done",
                "success": True,
                "reply": "two",
                "state": {"history": []},
                "files": [],
            },
        ]
    )
    answers = iter(["first task", "second task", "/exit"])
    adapter = CLIAdapter(sandbox=sandbox, input_func=lambda _: next(answers))

    adapter.run()

    assert len(sandbox.started_tasks) == 2
    first_session = sandbox.started_tasks[0]["session_id"]
    second_session = sandbox.started_tasks[1]["session_id"]
    assert first_session == second_session
    assert sandbox.stop_calls == 1


def test_follow_up_task_reuses_state_but_forces_replan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    sandbox = FakeSandbox(
        events=[
            {
                "type": "done",
                "success": True,
                "reply": "first",
                "state": {
                    "goal": "g1",
                    "plan": {"steps": ["old"]},
                    "history": [{"type": "action"}],
                },
                "files": [],
            },
            {
                "type": "done",
                "success": True,
                "reply": "second",
                "state": {"goal": "g2", "history": [{"type": "action"}, {"type": "action"}]},
                "files": [],
            },
        ]
    )
    answers = iter(["task one", "task two", "/quit"])
    adapter = CLIAdapter(sandbox=sandbox, input_func=lambda _: next(answers))

    adapter.run()

    assert len(sandbox.started_tasks) == 2
    second_task = sandbox.started_tasks[1]
    assert "state" in second_task
    assert second_task["state"]["history"] == [{"type": "action"}]
    assert "plan" not in second_task["state"]
