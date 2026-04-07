"""Transport abstractions for host-agent communication."""

from __future__ import annotations

import queue
from typing import Any

from agent.protocol import decode_event, encode_event


class InProcessTransport:
    """Queue-based in-process transport."""

    def __init__(
        self,
        incoming: queue.Queue[str] | None = None,
        outgoing: queue.Queue[str] | None = None,
    ) -> None:
        shared = incoming if incoming is not None else queue.Queue[str]()
        self._incoming: queue.Queue[str] = shared
        self._outgoing: queue.Queue[str] = outgoing if outgoing is not None else shared
        self._closed = False

    @classmethod
    def pair(cls) -> tuple[InProcessTransport, InProcessTransport]:
        """Create a connected host/agent transport pair."""
        a_to_b: queue.Queue[str] = queue.Queue()
        b_to_a: queue.Queue[str] = queue.Queue()
        return cls(incoming=b_to_a, outgoing=a_to_b), cls(incoming=a_to_b, outgoing=b_to_a)

    def send(self, event: dict[str, Any]) -> None:
        """Send an event."""
        if self._closed:
            raise RuntimeError("Transport is closed.")
        self._outgoing.put(encode_event(event))

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        """Receive an event or None on timeout."""
        if self._closed:
            raise RuntimeError("Transport is closed.")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be >= 0.")

        try:
            if timeout_seconds is None:
                line = self._incoming.get()
            else:
                line = self._incoming.get(timeout=timeout_seconds)
        except queue.Empty:
            return None
        return decode_event(line)

    def close(self) -> None:
        """Close transport resources."""
        self._closed = True
