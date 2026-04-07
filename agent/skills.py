"""Skill discovery and execution."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ToolResult:
    """Result returned by a skill invocation."""

    exit_code: int
    stdout: str
    stderr: str


class Skills:
    """Skill loader/executor."""

    def __init__(self, skills_dir: str) -> None:
        """Initialize skill discovery."""
        raise NotImplementedError("Backlog task A4 implements skills.")

    def index(self) -> list[dict[str, str]]:
        """Return a one-line index for all skills."""
        raise NotImplementedError("Backlog task A4 implements skills.")

    def get_doc(self, skill_name: str) -> str:
        """Return full SKILL.md content for one skill."""
        raise NotImplementedError("Backlog task A4 implements skills.")

    def execute(self, tool_call: dict[str, Any]) -> ToolResult:
        """Validate and execute one tool call."""
        raise NotImplementedError("Backlog task A4 implements skills.")
