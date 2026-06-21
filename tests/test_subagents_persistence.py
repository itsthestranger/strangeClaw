"""Persistence, journaling, and redaction parity for subagent observations (C10.7)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import session
from adapters.session_persistence import (
    append_session_event_journal,
    persist_done_event,
    redact_sensitive,
)


def _spawn_subagent_action_event() -> dict[str, Any]:
    # The parent's spawn_subagent observation: the child summary rides in the
    # wrapped result, and a (contrived) secret-keyed arg exercises redaction parity.
    return {
        "type": "action",
        "tool": "spawn_subagent",
        "args": {"task": "research", "authorization": "Bearer super-secret"},
        "result": {
            "exit_code": 0,
            "stdout": (
                "--- BEGIN DATA ---\n"
                '{"success":true,"status":"completed","child_id":"abc123",'
                '"duration_seconds":1.2,"reply":"child report"}\n'
                "--- END DATA ---"
            ),
            "stderr": "",
        },
    }


def test_redact_sensitive_redacts_spawn_subagent_event_keys() -> None:
    redacted = redact_sensitive(_spawn_subagent_action_event())
    # The event is not special-cased out of redaction: secret-named keys are masked.
    assert redacted["args"]["authorization"] == "[REDACTED]"
    assert redacted["tool"] == "spawn_subagent"


def test_journal_entry_for_spawn_subagent_is_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    append_session_event_journal(
        session_id="sess-1",
        event=_spawn_subagent_action_event(),
        enabled=True,
        max_bytes=1_000_000,
    )

    journal = (tmp_path / ".strangeclaw" / "sessions" / "sess-1" / "events.jsonl").read_text(
        encoding="utf-8"
    )
    assert "super-secret" not in journal
    assert "[REDACTED]" in journal
    # The child summary itself (status/reply) is still journaled under the parent.
    assert "child report" in journal


def test_journal_summary_entry_stays_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    journal_path = tmp_path / ".strangeclaw" / "sessions" / "sess-1" / "events.jsonl"
    for index in range(50):
        append_session_event_journal(
            session_id="sess-1",
            event={
                "type": "action",
                "tool": "spawn_subagent",
                "args": {"task": f"child-{index}"},
                "result": {"exit_code": 0, "stdout": "x" * 500, "stderr": ""},
            },
            enabled=True,
            max_bytes=4096,
        )

    assert journal_path.stat().st_size <= 4096


def test_persist_done_redacts_state_containing_subagent_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    done_event = {
        "type": "done",
        "success": True,
        "reply": "parent done",
        "state": {
            "goal": "g",
            "plan": {"steps": []},
            "history": [_spawn_subagent_action_event()],
            "summary": "",
        },
        "files": [],
    }

    redacted_state = persist_done_event(session_id="sess-2", done_event=done_event)
    assert redacted_state is not None
    assert redacted_state["history"][0]["args"]["authorization"] == "[REDACTED]"

    on_disk = session.load(tmp_path / ".strangeclaw" / "sessions" / "sess-2")
    assert on_disk is not None
    serialized = json.dumps(on_disk)
    assert "super-secret" not in serialized
