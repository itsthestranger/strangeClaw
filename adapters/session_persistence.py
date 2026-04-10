"""Shared adapter helpers for session persistence and follow-up state."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

import session


def persist_done_event(*, session_id: str, done_event: dict[str, Any]) -> dict[str, Any] | None:
    """Persist done event state and output files for a session."""
    state = done_event.get("state")
    if not isinstance(state, dict):
        return None

    redacted_state = _redact_sensitive(state)
    session_dir = session.create(session_id)
    session.save(session_dir, redacted_state)
    outputs_dir = session_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    files = done_event.get("files")
    if isinstance(files, list):
        for item in files:
            if not isinstance(item, dict):
                continue
            rel_path = item.get("path")
            content_b64 = item.get("content_b64")
            if not isinstance(rel_path, str) or not rel_path:
                continue
            if not isinstance(content_b64, str):
                continue

            output_path = _safe_output_path(outputs_dir, rel_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                decoded = base64.b64decode(content_b64, validate=True)
            except binascii.Error as exc:
                raise ValueError(f"Invalid base64 content for output file: {rel_path}") from exc
            output_path.write_bytes(decoded)

    return state


def state_for_follow_up(state: dict[str, Any]) -> dict[str, Any]:
    """Return follow-up state that preserves context while forcing re-planning."""
    next_state = dict(state)
    next_state.pop("plan", None)
    return next_state


def _safe_output_path(outputs_dir: Path, rel_path: str) -> Path:
    candidate = (outputs_dir / rel_path).resolve()
    root = outputs_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Invalid output file path: {rel_path}")
    return candidate


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key.lower() == "api_key":
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value

