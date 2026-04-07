"""Yolo sandbox placeholder."""

from typing import Any


class YoloSandbox:
    """In-process sandbox interface."""

    def run(self, task: dict[str, Any]) -> None:
        """Start an agent run for a task."""
        raise NotImplementedError("Backlog task A6.1 implements YoloSandbox.")

    def send(self, event: dict[str, Any]) -> None:
        """Send an event to the agent."""
        raise NotImplementedError("Backlog task A6.1 implements YoloSandbox.")

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        """Receive an event from the agent."""
        raise NotImplementedError("Backlog task A6.1 implements YoloSandbox.")

    def stop(self) -> None:
        """Stop the agent run."""
        raise NotImplementedError("Backlog task A6.1 implements YoloSandbox.")
