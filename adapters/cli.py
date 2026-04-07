"""CLI adapter."""

from typing import Any


class CLIAdapter:
    """CLI interaction contract."""

    def get_task(self) -> dict[str, Any]:
        """Get initial task from the user."""
        raise NotImplementedError("Backlog task A6.2 implements CLIAdapter.")

    def show(self, event: dict[str, Any]) -> None:
        """Display an agent event."""
        raise NotImplementedError("Backlog task A6.2 implements CLIAdapter.")

    def get_reply(self, role: str) -> dict[str, Any]:
        """Get user feedback for plan review or clarification."""
        raise NotImplementedError("Backlog task A6.2 implements CLIAdapter.")

    def run(self) -> None:
        """Drive the adapter event loop."""
        raise NotImplementedError("Backlog task A6.2 implements CLIAdapter.")
