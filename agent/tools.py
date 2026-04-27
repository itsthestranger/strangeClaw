"""Built-in tool primitives for strangeclaw."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from agent.llm import ToolCall

_OUTPUT_CHUNK_SIZE = 4000
_DEFAULT_SHELL_TIMEOUT_SECONDS = 60.0
_KNOWN_TOOLS = ("shell", "web_search", "web_fetch", "http_request")


@dataclass(slots=True)
class ToolResult:
    """Result returned by a tool invocation."""

    exit_code: int
    stdout: str
    stderr: str


class Tools:
    """Built-in capability registry and dispatcher."""

    def __init__(self, config: dict[str, Any]) -> None:
        raw_tools = config.get("tools")
        if isinstance(raw_tools, dict):
            enabled: set[str] = {
                name
                for name in _KNOWN_TOOLS
                if bool(raw_tools.get(name, True))
            }
        else:
            enabled = set(_KNOWN_TOOLS)
        self._enabled = enabled

    def list_enabled(self) -> list[str]:
        """Return enabled tool names."""
        return sorted(self._enabled)

    def schema(self) -> list[dict[str, Any]]:
        """Return tool JSON Schemas for all enabled tools."""
        schemas: list[dict[str, Any]] = []
        if "shell" in self._enabled:
            schemas.append(self._shell_schema())
        return schemas

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute one tool call and return structured output."""
        tool_name = getattr(tool_call, "tool", None)
        args = getattr(tool_call, "args", None)
        if not isinstance(tool_name, str) or not tool_name:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="tool_call.tool must be a non-empty string.",
            )
        if not isinstance(args, dict):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="tool_call.args must be an object.",
            )
        if tool_name not in self._enabled:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr=f"tool {tool_name} is not enabled.",
            )
        if tool_name == "shell":
            return self._execute_shell(args)
        return ToolResult(
            exit_code=1,
            stdout="",
            stderr=f"tool {tool_name} is not implemented yet.",
        )

    def _execute_shell(self, args: dict[str, Any]) -> ToolResult:
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="shell.command must be a non-empty string.",
            )
        timeout_raw = args.get("timeout_seconds", _DEFAULT_SHELL_TIMEOUT_SECONDS)
        if isinstance(timeout_raw, bool) or not isinstance(timeout_raw, (int, float)):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="shell.timeout_seconds must be a positive number.",
            )
        timeout_seconds = float(timeout_raw)
        if timeout_seconds <= 0:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="shell.timeout_seconds must be a positive number.",
            )

        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            timeout_message = f"Command timed out after {timeout_seconds:.1f}s."
            stderr = f"{stderr}\n{timeout_message}" if stderr else timeout_message
            return ToolResult(
                exit_code=124,
                stdout=_truncate_output(stdout),
                stderr=_truncate_output(stderr),
            )
        except OSError as exc:
            return ToolResult(exit_code=1, stdout="", stderr=f"Failed to execute shell: {exc}")

        return ToolResult(
            exit_code=completed.returncode,
            stdout=_truncate_output(completed.stdout),
            stderr=_truncate_output(completed.stderr),
        )

    @staticmethod
    def _shell_schema() -> dict[str, Any]:
        return {
            "name": "shell",
            "description": "Run a shell command and return stdout/stderr/exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_seconds": {"type": "number", "default": 60.0},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        }


def _truncate_output(text: str, *, chunk_size: int = _OUTPUT_CHUNK_SIZE) -> str:
    if len(text) <= chunk_size * 2:
        return text
    omitted_chars = len(text) - (chunk_size * 2)
    return (
        f"{text[:chunk_size]}\n"
        f"...[truncated {omitted_chars} chars]...\n"
        f"{text[-chunk_size:]}"
    )
