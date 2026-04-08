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
        self.started_task: dict[str, Any] | None = None
        self.stopped = False

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
        self.stopped = True


def test_get_task_includes_type_session_and_llm() -> None:
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
    assert task["llm"] == {"model": "x", "api_key": "k"}


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
                "skill": "shell",
                "action": "run",
                "result": {"exit_code": 0, "stdout": "", "stderr": ""},
            },
            {"type": "done", "success": True, "reply": "All good.", "state": {}, "files": []},
        ]
    )
    answers = iter(["Build it", "y"])
    adapter = CLIAdapter(
        sandbox=sandbox,
        approval_mode="review",
        input_func=lambda _: next(answers),
    )

    adapter.run()

    assert sandbox.started_task is not None
    assert sandbox.started_task["approval_mode"] == "review"
    assert sandbox.sent == [{"type": "user_reply", "approved": True, "text": ""}]
    assert sandbox.stopped is True

    output = capsys.readouterr().out
    assert "Plan:" in output
    assert "Action: shell.run (exit=0)" in output
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
    answers = iter(["Start task", "Use 8080"])
    adapter = CLIAdapter(
        sandbox=sandbox,
        approval_mode="auto",
        input_func=lambda _: next(answers),
    )

    adapter.run()
    assert sandbox.sent == [{"type": "user_reply", "approved": True, "text": "Use 8080"}]


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
    adapter = CLIAdapter(sandbox=sandbox, input_func=lambda _: "Persist this")

    adapter.run()

    assert sandbox.started_task is not None
    session_id = sandbox.started_task["session_id"]
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


def test_get_task_uses_resume_state_without_prompt() -> None:
    sandbox = FakeSandbox(events=[])
    adapter = CLIAdapter(
        sandbox=sandbox,
        resume_session_id="resume-1",
        resume_state={"goal": "resume-goal", "history": []},
    )
    task = adapter.get_task()
    assert task["session_id"] == "resume-1"
    assert task["text"] == "resume-goal"
    assert task["state"] == {"goal": "resume-goal", "history": []}
