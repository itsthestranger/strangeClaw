"""Telegram adapter placeholder."""

from typing import Any


class TelegramAdapter:
    """Telegram interaction contract."""

    def get_task(self) -> dict[str, Any]:
        """Get initial task from the user."""
        raise NotImplementedError("Future task implements TelegramAdapter.")

    def show(self, event: dict[str, Any]) -> None:
        """Display an agent event."""
        raise NotImplementedError("Future task implements TelegramAdapter.")

    def get_reply(self, role: str) -> dict[str, Any]:
        """Get user feedback for plan review or clarification."""
        raise NotImplementedError("Future task implements TelegramAdapter.")

    def run(self) -> None:
        """Drive the adapter event loop."""
        raise NotImplementedError("Future task implements TelegramAdapter.")
