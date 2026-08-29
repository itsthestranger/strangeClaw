"""Event protocol helpers."""

from __future__ import annotations

import json
from typing import Any, cast

# Closed key schema: every event type declares the full set of top-level keys it
# may carry (required and optional alike). Any key outside its set is rejected,
# so a field cannot be smuggled across the guest/host boundary by riding along on
# an otherwise valid event.
EVENT_KEYS: dict[str, frozenset[str]] = {
    "task": frozenset({"type", "text", "session_id", "approval_mode", "state"}),
    "user_reply": frozenset({"type", "text", "approved"}),
    "stop": frozenset({"type"}),
    "agent_ready": frozenset({"type"}),
    "message": frozenset({"type", "role", "content"}),
    "action": frozenset({"type", "tool", "args", "result"}),
    "done": frozenset({"type", "success", "reply", "state", "files"}),
    "broker_request": frozenset({"type", "request_id", "service", "payload"}),
    "broker_response": frozenset({"type", "request_id", "success", "payload", "error"}),
}
EVENT_TYPES = frozenset(EVENT_KEYS)
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

    _reject_unknown_keys(event, event_type)

    if event_type == "task":
        _require_str(event, "text")
        _require_str(event, "session_id")
        _require_str(event, "approval_mode")
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

    if event_type == "broker_request":
        _require_str(event, "request_id")
        _require_str(event, "service")
        _require_dict(event, "payload")
        return

    if event_type == "broker_response":
        _require_str(event, "request_id")
        success = _require_bool(event, "success")
        if success:
            _require_dict(event, "payload")
            return
        _require_str(event, "error")
        return


def _reject_unknown_keys(event: dict[str, Any], event_type: str) -> None:
    allowed = EVENT_KEYS[event_type]
    unknown = sorted(key for key in event if key not in allowed)
    if not unknown:
        return
    named = ", ".join(f"'{key}'" for key in unknown)
    raise ProtocolError(
        f"Event field {named} is not allowed on {event_type} events. "
        f"Allowed fields: {', '.join(sorted(allowed))}."
    )


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
