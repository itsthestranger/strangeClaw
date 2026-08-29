"""Tests for agent protocol encoding/decoding."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.protocol import EVENT_KEYS, EVENT_TYPES, ProtocolError, decode_event, encode_event

ROUNDTRIP_EVENTS: list[dict[str, Any]] = [
    {
        "type": "task",
        "text": "do thing",
        "session_id": "abc123",
        "approval_mode": "review",
    },
    {"type": "user_reply", "text": "yes", "approved": True},
    {"type": "stop"},
    {"type": "agent_ready"},
    {"type": "message", "role": "status", "content": "working"},
    {
        "type": "action",
        "tool": "shell",
        "args": {"command": "python3 --version"},
        "result": {"exit_code": 0, "stdout": "Python 3.13.0", "stderr": ""},
    },
    {
        "type": "broker_request",
        "request_id": "abc123",
        "service": "broker",
        "payload": {"action": "list_integrations"},
    },
    {
        "type": "broker_response",
        "request_id": "abc123",
        "success": True,
        "payload": {"success": True, "integrations": ["notion"]},
    },
    {
        "type": "broker_response",
        "request_id": "def456",
        "success": False,
        "error": "policy_denied",
    },
    {"type": "done", "success": True, "reply": "done", "state": {"goal": "x"}, "files": []},
    # Optional keys must survive the closed schema: ``state`` on task follow-ups
    # and an omitted ``files`` on done.
    {
        "type": "task",
        "text": "follow up",
        "session_id": "abc123",
        "approval_mode": "auto",
        "state": {"goal": "x", "history": []},
    },
    {"type": "done", "success": False, "reply": "failed", "state": {}},
]


@pytest.mark.parametrize("event", ROUNDTRIP_EVENTS)
def test_protocol_roundtrip(event: dict[str, object]) -> None:
    encoded = encode_event(event)
    decoded = decode_event(encoded)
    assert decoded == event


def test_decode_rejects_invalid_json() -> None:
    with pytest.raises(ProtocolError):
        decode_event("{oops}\n")


def test_encode_rejects_invalid_event_shape() -> None:
    with pytest.raises(ProtocolError, match="Unsupported event type"):
        encode_event({"type": "nope"})


def test_task_event_rejects_llm_field() -> None:
    event = {
        "type": "task",
        "text": "do thing",
        "session_id": "abc123",
        "approval_mode": "review",
        "llm": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
    }
    with pytest.raises(ProtocolError, match="Event field 'llm' is not allowed on task events"):
        encode_event(event)


def test_task_event_rejects_llm_field_on_decode() -> None:
    line = json.dumps(
        {
            "type": "task",
            "text": "do thing",
            "session_id": "abc123",
            "approval_mode": "review",
            "llm": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
        }
    )
    with pytest.raises(ProtocolError, match="Event field 'llm' is not allowed on task events"):
        decode_event(line)


@pytest.mark.parametrize("event", ROUNDTRIP_EVENTS)
def test_encode_rejects_undeclared_keys(event: dict[str, object]) -> None:
    smuggled = {**event, "smuggled": {"api_key": "sk-test"}}
    with pytest.raises(ProtocolError, match="Event field 'smuggled' is not allowed"):
        encode_event(smuggled)


@pytest.mark.parametrize("event", ROUNDTRIP_EVENTS)
def test_decode_rejects_undeclared_keys(event: dict[str, object]) -> None:
    line = json.dumps({**event, "smuggled": {"api_key": "sk-test"}})
    with pytest.raises(ProtocolError, match="Event field 'smuggled' is not allowed"):
        decode_event(line)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "message", "role": "status", "content": "hi", "llm": {"api_key": "sk-test"}},
        {
            "type": "action",
            "tool": "shell",
            "args": {},
            "result": {},
            "llm": {"api_key": "sk-test"},
        },
        {
            "type": "broker_request",
            "request_id": "abc123",
            "service": "llm",
            "payload": {},
            "llm": {"api_key": "sk-test"},
        },
    ],
)
def test_llm_field_is_rejected_on_every_event_type(event: dict[str, object]) -> None:
    """The general rule closes the types the bespoke task-only check never covered."""
    with pytest.raises(ProtocolError, match="Event field 'llm' is not allowed"):
        encode_event(event)


def test_unknown_key_error_names_every_offending_field() -> None:
    event = {"type": "stop", "beta": 2, "alpha": 1}
    with pytest.raises(ProtocolError) as excinfo:
        encode_event(event)
    message = str(excinfo.value)
    assert "'alpha'" in message
    assert "'beta'" in message
    assert "stop events" in message


def test_done_event_rejects_session_id() -> None:
    event = {
        "type": "done",
        "success": True,
        "reply": "done",
        "state": {},
        "files": [],
        "session_id": "abc123",
    }
    with pytest.raises(ProtocolError, match="Event field 'session_id' is not allowed"):
        encode_event(event)


def test_every_event_type_declares_a_key_set() -> None:
    assert set(EVENT_KEYS) == set(EVENT_TYPES)
    for event_type, keys in EVENT_KEYS.items():
        assert "type" in keys, event_type


def test_roundtrip_events_cover_every_event_type() -> None:
    assert {str(event["type"]) for event in ROUNDTRIP_EVENTS} == set(EVENT_TYPES)
