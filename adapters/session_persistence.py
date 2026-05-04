"""Shared adapter helpers for session persistence and follow-up state."""

from __future__ import annotations

import base64
import binascii
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import session
from broker.redaction import redact_sensitive


def persist_done_event(*, session_id: str, done_event: dict[str, Any]) -> dict[str, Any] | None:
    """Persist done event state and output files for a session."""
    state = done_event.get("state")
    if not isinstance(state, dict):
        return None

    redacted_state = redact_sensitive(state)
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


def append_session_event_journal(
    *,
    session_id: str,
    event: dict[str, Any],
    enabled: bool,
    max_bytes: int,
    file_name: str = "events.jsonl",
) -> None:
    """Append a redacted journal event while enforcing a max-size bound.

    This function is intentionally fail-open: write failures must not affect runtime flow.
    """
    if not enabled or max_bytes <= 0:
        return

    try:
        session_dir = session.create(session_id)
        journal_path = session_dir / file_name
        entry = {
            "ts": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "event": redact_sensitive(event),
        }
        line = json.dumps(entry, ensure_ascii=True, sort_keys=True).encode("utf-8") + b"\n"
        if len(line) > max_bytes:
            return

        existing = b""
        if journal_path.exists():
            existing = journal_path.read_bytes()

        budget = max_bytes - len(line)
        if len(existing) > budget:
            existing = existing[-budget:]
            newline_index = existing.find(b"\n")
            if newline_index >= 0:
                existing = existing[newline_index + 1 :]
            else:
                existing = b""

        payload = existing + line
        session_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=session_dir,
            delete=False,
            prefix="events.",
            suffix=".tmp",
        ) as handle:
            handle.write(payload)
            temp_path = Path(handle.name)
        temp_path.replace(journal_path)
    except Exception:
        return


def _safe_output_path(outputs_dir: Path, rel_path: str) -> Path:
    candidate = (outputs_dir / rel_path).resolve()
    root = outputs_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Invalid output file path: {rel_path}")
    return candidate

