"""Tests for agent protocol encoding/decoding."""

from __future__ import annotations

import pytest

from agent.protocol import ProtocolError, decode_event, encode_event

ROUNDTRIP_EVENTS = [
    {
        "type": "task",
        "text": "do thing",
        "session_id": "abc123",
        "approval_mode": "review",
        "llm": {"model": "openai/gpt-4.1", "api_key": "sk-test"},
    },
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
