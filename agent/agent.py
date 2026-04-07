"""Core inspect-choose-act-observe loop."""

from typing import Any


class Agent:
    """Main agent runtime."""

    def run(self, task: dict[str, Any]) -> None:
        """Execute a task end-to-end."""
        raise NotImplementedError("Backlog task A5 implements Agent loop.")
