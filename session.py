"""Session persistence helpers for strangeclaw."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, cast

SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")


class SessionError(ValueError):
    """Raised when session operations fail validation."""


def create(session_id: str) -> Path:
    """Create and return a session directory path."""
    _validate_session_id(session_id)
    session_root = Path.home() / ".strangeclaw" / "sessions"
    session_dir = session_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save(session_dir: Path, state: dict[str, Any]) -> None:
    """Persist session state atomically."""
    session_dir.mkdir(parents=True, exist_ok=True)
    state_path = session_dir / "state.json"
    payload = json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=session_dir,
        delete=False,
        prefix="state.",
        suffix=".tmp",
    ) as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, state_path)


def load(session_dir: Path) -> dict[str, Any] | None:
    """Load session state if present."""
    state_path = session_dir / "state.json"
    if not state_path.exists():
        return None

    with state_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, dict):
        raise SessionError(f"Session state in {state_path} must be a JSON object.")
    return cast(dict[str, Any], loaded)


def _validate_session_id(session_id: str) -> None:
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise SessionError(
            "Invalid session_id. Allowed characters: letters, numbers, hyphen."
        )
