"""Event protocol helpers."""

from __future__ import annotations

import json
from typing import Any, cast

EVENT_TYPES = {
    "task",
    "user_reply",
    "stop",
    "agent_ready",
    "message",
    "action",
    "done",
}
MESSAGE_ROLES = {"plan", "clarification", "status", "reply"}


class ProtocolError(ValueError):
    """Raised when an event is malformed."""


def encode_event(event: dict[str, Any]) -> str:
    """Encode an event dictionary to a protocol line."""
    validate_event(event)
    return f"{json.dumps(event, separators=(',', ':'))}\n"


def decode_event(line: str) -> dict[str, Any]:
    """Decode and validate an event line."""
    text = line.strip()
    if not text:
        raise ProtocolError("Cannot decode empty event line.")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Invalid JSON event: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProtocolError("Decoded event must be a JSON object.")
    event = cast(dict[str, Any], parsed)
    validate_event(event)
    return event


def validate_event(event: dict[str, Any]) -> None:
    """Validate an event against strangeclaw protocol requirements."""
    event_type = event.get("type")
    if not isinstance(event_type, str):
        raise ProtocolError("Event field 'type' is required and must be a string.")
    if event_type not in EVENT_TYPES:
        raise ProtocolError(f"Unsupported event type: {event_type}")

    if event_type == "task":
        _require_str(event, "text")
        _require_str(event, "session_id")
        _require_str(event, "approval_mode")
        llm = event.get("llm")
        if llm is not None and not isinstance(llm, dict):
            raise ProtocolError("Event field 'llm' must be an object when provided.")
        return

    if event_type == "user_reply":
        _require_str(event, "text")
        _require_bool(event, "approved")
        return

    if event_type == "stop":
        return

    if event_type == "agent_ready":
        return

    if event_type == "message":
        role = _require_str(event, "role")
        if role not in MESSAGE_ROLES:
            raise ProtocolError(f"Unsupported message role: {role}")
        if "content" not in event:
            raise ProtocolError("Event field 'content' is required for message events.")
        return

    if event_type == "action":
        _require_str(event, "tool")
        _require_dict(event, "args")
        _require_dict(event, "result")
        return

    if event_type == "done":
        _require_bool(event, "success")
        _require_str(event, "reply")
        _require_dict(event, "state")
        files = event.get("files")
        if files is not None and not isinstance(files, list):
            raise ProtocolError("Event field 'files' must be an array when provided.")
        return


def _require_str(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str):
        raise ProtocolError(f"Event field '{key}' is required and must be a string.")
    return value


def _require_bool(event: dict[str, Any], key: str) -> bool:
    value = event.get(key)
    if not isinstance(value, bool):
        raise ProtocolError(f"Event field '{key}' is required and must be a boolean.")
    return value


def _require_dict(event: dict[str, Any], key: str) -> dict[str, Any]:
    value = event.get(key)
    if not isinstance(value, dict):
        raise ProtocolError(f"Event field '{key}' is required and must be an object.")
    return value
