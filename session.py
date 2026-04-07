"""Session persistence helpers for strangeclaw."""

from pathlib import Path
from typing import Any


def create(session_id: str) -> Path:
    """Create and return a session directory path."""
    raise NotImplementedError("Backlog task A2.2 implements session storage.")


def save(session_dir: Path, state: dict[str, Any]) -> None:
    """Persist session state atomically."""
    raise NotImplementedError("Backlog task A2.2 implements session storage.")


def load(session_dir: Path) -> dict[str, Any] | None:
    """Load session state if present."""
    raise NotImplementedError("Backlog task A2.2 implements session storage.")
