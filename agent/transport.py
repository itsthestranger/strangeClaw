"""Transport abstractions for host-agent communication."""

from typing import Any


class InProcessTransport:
    """Queue-based in-process transport."""

    def send(self, event: dict[str, Any]) -> None:
        """Send an event."""
        raise NotImplementedError("Backlog task A2.3 implements transport.")

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        """Receive an event or None on timeout."""
        raise NotImplementedError("Backlog task A2.3 implements transport.")

    def close(self) -> None:
        """Close transport resources."""
        raise NotImplementedError("Backlog task A2.3 implements transport.")
