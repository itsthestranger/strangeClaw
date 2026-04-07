"""Firecracker sandbox placeholder."""

from typing import Any


class FireSandbox:
    """Firecracker sandbox interface."""

    def run(self, task: dict[str, Any]) -> None:
        """Start an agent run for a task."""
        raise NotImplementedError("Backlog task B2.4 implements FireSandbox.")

    def send(self, event: dict[str, Any]) -> None:
        """Send an event to the agent."""
        raise NotImplementedError("Backlog task B2.4 implements FireSandbox.")

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        """Receive an event from the agent."""
        raise NotImplementedError("Backlog task B2.4 implements FireSandbox.")

    def stop(self) -> None:
        """Stop the VM and clean up resources."""
        raise NotImplementedError("Backlog task B2.4 implements FireSandbox.")
