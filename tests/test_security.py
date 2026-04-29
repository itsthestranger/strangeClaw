"""Security-focused tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import session
from adapters.cli import CLIAdapter
from agent.skills import Skills, SkillsError


class FakeSandbox:
    """Minimal sandbox stub for persistence tests."""

    def __init__(self, done_event: dict[str, Any]) -> None:
        self._events = [done_event]

    def run(self, task: dict[str, Any]) -> None:
        del task

    def send(self, event: dict[str, Any]) -> None:
        del event

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        del timeout_seconds
        if not self._events:
            return None
        return self._events.pop(0)

    def stop(self) -> None:
        return


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def test_session_id_sanitization_rejects_invalid_characters() -> None:
    with pytest.raises(session.SessionError, match="Invalid session_id"):
        session.create("abc/def")
    with pytest.raises(session.SessionError, match="Invalid session_id"):
        session.create("abc def")
    with pytest.raises(session.SessionError, match="Invalid session_id"):
        session.create("abc_def")


def test_read_file_rejects_path_traversal() -> None:
    skills = Skills(str(_skills_root()))

    with pytest.raises(SkillsError, match="path traversal"):
        skills.read_file("shell", "../outside.txt")


def test_api_key_not_persisted_to_session_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    secret = "sk-test-secret-123"
    done_event = {
        "type": "done",
        "success": True,
        "reply": "done",
        "state": {
            "goal": "x",
            "llm": {"api_key": secret, "model": "fake"},
            "history": [{"step": 1, "api_key": secret}],
        },
        "files": [],
    }
    answers = iter(["save it", "/quit"])
    adapter = CLIAdapter(sandbox=FakeSandbox(done_event), input_func=lambda _: next(answers))

    adapter.run()

    sessions_dir = tmp_path / ".strangeclaw" / "sessions"
    state_files = list(sessions_dir.rglob("state.json"))
    assert len(state_files) == 1
    content = state_files[0].read_text(encoding="utf-8")
    assert secret not in content
    loaded = json.loads(content)
    assert loaded["llm"]["api_key"] == "[REDACTED]"
