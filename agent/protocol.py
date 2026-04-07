"""Event protocol helpers."""

from typing import Any


def encode_event(event: dict[str, Any]) -> str:
    """Encode an event dictionary to a protocol line."""
    raise NotImplementedError("Backlog task A2.3 implements protocol encoding.")


def decode_event(line: str) -> dict[str, Any]:
    """Decode and validate an event line."""
    raise NotImplementedError("Backlog task A2.3 implements protocol decoding.")
