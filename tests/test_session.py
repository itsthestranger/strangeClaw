"""Tests for session helpers."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

import session


def test_create_makes_session_directory_in_user_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    session_dir = session.create("abc-123")

    assert session_dir == tmp_path / ".strangeclaw" / "sessions" / "abc-123"
    assert session_dir.exists()


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "one"
    state = {"goal": "demo", "steps": [1, 2, 3], "ok": True}

    session.save(session_dir, state)

    assert session.load(session_dir) == state


def test_load_returns_none_when_state_file_missing(tmp_path: Path) -> None:
    assert session.load(tmp_path / "missing") is None


def test_concurrent_saves_do_not_corrupt_state_file(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "parallel"
    errors: list[Exception] = []
    start = threading.Barrier(4)

    def writer(writer_id: int) -> None:
        try:
            start.wait()
            for idx in range(40):
                session.save(session_dir, {"writer": writer_id, "idx": idx})
        except Exception as exc:  # pragma: no cover - assertion path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    final_state = session.load(session_dir)
    assert isinstance(final_state, dict)
    assert final_state["writer"] in {0, 1, 2, 3}
    assert isinstance(final_state["idx"], int)


def test_create_rejects_invalid_session_id() -> None:
    with pytest.raises(session.SessionError, match="Invalid session_id"):
        session.create("../bad")
