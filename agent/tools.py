"""Built-in tool primitives for strangeclaw."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

from agent.broker_client import BrokerClient, HostServiceError
from agent.llm_types import ToolCall

_OUTPUT_CHUNK_SIZE = 4000
_DEFAULT_SHELL_TIMEOUT_SECONDS = 60.0
_HTTP_REQUEST_ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}

# Environment variables the shell tool hands to the command it runs. The child
# environment is built from this allowlist instead of being inherited, so a
# value exported only into the parent process (a one-shot token from a wrapper
# script, say) is not readable by a model-authored command. Operators widen the
# list with `shell.env_passthrough`. An allowlisted name that is missing from
# the parent is omitted rather than passed as an empty string: an empty `HOME`
# or `PATH` is a different, worse thing than an unset one.
SHELL_ENV_ALLOWLIST: tuple[str, ...] = ("PATH", "HOME", "LANG", "TERM", "TZ")

# Canonical registry of built-in capability tools. This is the single source of
# truth for which tools `Tools` implements; every other module that needs the
# capability tool set (subagent tool narrowing, config validation, Fire MMDS
# sanitization) imports from here so adding a tool means editing one place.
# `spawn_subagent` is an agent-dispatched orchestration capability, not a `Tools`
# method, so it is intentionally excluded here and layered on by callers that
# expose the config toggle (see `default_tool_toggles`).
CAPABILITY_TOOL_NAMES: tuple[str, ...] = ("shell", "web_search", "web_fetch", "http_request")

_SPAWN_SUBAGENT_TOGGLE = "spawn_subagent"


def default_tool_toggles() -> dict[str, bool]:
    """Default enable/disable state for every config-toggleable tool.

    Capability tools default on; the `spawn_subagent` orchestration capability
    defaults off (both it and `subagents.enabled` must be set to run children).
    """
    toggles: dict[str, bool] = {name: True for name in CAPABILITY_TOOL_NAMES}
    toggles[_SPAWN_SUBAGENT_TOGGLE] = False
    return toggles


@dataclass(slots=True)
class ToolResult:
    """Result returned by a tool invocation."""

    exit_code: int
    stdout: str
    stderr: str


class Tools:
    """Built-in capability registry and dispatcher."""

    def __init__(self, config: dict[str, Any], broker: BrokerClient | None = None) -> None:
        self._config = dict(config)
        self._broker = broker
        raw_tools = config.get("tools")
        if isinstance(raw_tools, dict):
            enabled: set[str] = {
                name
                for name in CAPABILITY_TOOL_NAMES
                if bool(raw_tools.get(name, True))
            }
        else:
            enabled = set(CAPABILITY_TOOL_NAMES)
        self._enabled = enabled

    def list_enabled(self) -> list[str]:
        """Return enabled tool names."""
        return sorted(self._enabled)

    def schema(self) -> list[dict[str, Any]]:
        """Return tool JSON Schemas for all enabled tools."""
        schemas: list[dict[str, Any]] = []
        if "shell" in self._enabled:
            schemas.append(self._shell_schema())
        if "web_search" in self._enabled:
            schemas.append(self._web_search_schema())
        if "web_fetch" in self._enabled:
            schemas.append(self._web_fetch_schema())
        if "http_request" in self._enabled:
            schemas.append(self._http_request_schema())
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
        if tool_name == "web_search":
            return self._execute_web_search(args)
        if tool_name == "web_fetch":
            return self._execute_web_fetch(args)
        if tool_name == "http_request":
            return self._execute_http_request(args)
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
                env=self._shell_env(),
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

    def _shell_env(self) -> dict[str, str]:
        """Build the shell child environment from the allowlist, not by inheritance."""
        env: dict[str, str] = {}
        for name in (*SHELL_ENV_ALLOWLIST, *self._configured_env_passthrough()):
            value = os.environ.get(name)
            if value is None:
                continue
            env[name] = value
        return env

    def _configured_env_passthrough(self) -> tuple[str, ...]:
        """Operator-added passthrough variable names from `shell.env_passthrough`."""
        shell_cfg = self._config.get("shell")
        if not isinstance(shell_cfg, dict):
            return ()
        raw_names = shell_cfg.get("env_passthrough")
        if not isinstance(raw_names, list):
            return ()
        return tuple(
            name.strip() for name in raw_names if isinstance(name, str) and name.strip()
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

    @staticmethod
    def _web_search_schema() -> dict[str, Any]:
        return {
            "name": "web_search",
            "description": "Search the web and return normalized result snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }

    @staticmethod
    def _web_fetch_schema() -> dict[str, Any]:
        return {
            "name": "web_fetch",
            "description": (
                "Fetch a URL and return the HTTP response "
                "(status, headers, decoded text body, truncation flag)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        }

    def _http_request_schema(self) -> dict[str, Any]:
        return {
            "name": "http_request",
            "description": "Make structured HTTP requests and return status, headers, and body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "url": {"type": "string"},
                    "integration": {
                        "type": ["string", "null"],
                        "description": (
                            "Named integration from secrets.yaml. Required for authenticated APIs. "
                            "Omit only for unauthenticated GET requests to public URLs."
                        ),
                    },
                    "headers": {"type": "object"},
                    "body": {"type": ["string", "null"]},
                },
                "required": ["method", "url"],
                "additionalProperties": False,
            },
        }

    def _execute_web_search(self, args: dict[str, Any]) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(exit_code=1, stdout="", stderr="web_search.query must be a string.")
        max_results = self._configured_max_results()
        if max_results is None:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="web_search.max_results must be a positive integer.",
            )
        return self._dispatch_broker_action(
            {
                "action": "web_search",
                "query": query.strip(),
                "max_results": max_results,
            },
            label="web_search",
        )

    def _execute_web_fetch(self, args: dict[str, Any]) -> ToolResult:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return ToolResult(exit_code=1, stdout="", stderr="web_fetch.url must be a string.")
        return self._dispatch_broker_action(
            {"action": "web_fetch", "url": url.strip()},
            label="web_fetch",
        )

    def _execute_http_request(self, args: dict[str, Any]) -> ToolResult:
        method_raw = args.get("method")
        if not isinstance(method_raw, str) or not method_raw.strip():
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="http_request.method must be a string.",
            )
        method = method_raw.strip().upper()
        if method not in _HTTP_REQUEST_ALLOWED_METHODS:
            allowed = ", ".join(sorted(_HTTP_REQUEST_ALLOWED_METHODS))
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr=f"http_request.method must be one of: {allowed}.",
            )

        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return ToolResult(exit_code=1, stdout="", stderr="http_request.url must be a string.")
        target_url = url.strip()

        headers = args.get("headers", {})
        if headers is None:
            headers = {}
        if not isinstance(headers, dict):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="http_request.headers must be an object when provided.",
            )
        normalized_headers: dict[str, str] = {}
        for key, value in headers.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return ToolResult(
                    exit_code=1,
                    stdout="",
                    stderr="http_request.headers must contain only string keys and values.",
                )
            normalized_headers[key] = value

        integration_raw = args.get("integration")
        if integration_raw is not None and not isinstance(integration_raw, str):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="http_request.integration must be a string or null.",
            )

        body = args.get("body")
        if body is not None and not isinstance(body, str):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="http_request.body must be a string or null.",
            )
        return self._dispatch_broker_action(
            {
                "action": "http_request",
                "integration": integration_raw,
                "method": method,
                "url": target_url,
                "headers": normalized_headers,
                "body": body,
            },
            label="http_request",
        )

    def _dispatch_broker_action(self, payload: dict[str, Any], *, label: str) -> ToolResult:
        """Send a broker action and map its success envelope to a ToolResult.

        Shared tail for all broker-backed tools: enforces broker availability,
        surfaces transport failures, validates the success envelope, and wraps the
        response as external data. Per-tool error strings use ``label``.
        """
        if self._broker is None:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr=f"host service broker is not configured for {label}.",
            )
        try:
            result = self._broker.call("broker", payload)
        except HostServiceError as exc:
            return ToolResult(exit_code=1, stdout="", stderr=str(exc))
        success = result.get("success")
        if success is not True and success is not False:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr=f"invalid broker response for {label}: missing success envelope.",
            )
        return ToolResult(
            exit_code=0 if success else 1,
            stdout=wrap_external_data(result),
            stderr="",
        )

    def _configured_max_results(self) -> int | None:
        web_search_cfg = self._config.get("web_search", {})
        if not isinstance(web_search_cfg, dict):
            web_search_cfg = {}
        raw_max_results = web_search_cfg.get("max_results", 10)
        if isinstance(raw_max_results, bool):
            return None
        try:
            max_results = int(raw_max_results)
        except (TypeError, ValueError):
            return None
        if max_results <= 0:
            return None
        return max_results


def _truncate_output(text: str, *, chunk_size: int = _OUTPUT_CHUNK_SIZE) -> str:
    if len(text) <= chunk_size * 2:
        return text
    omitted_chars = len(text) - (chunk_size * 2)
    return (
        f"{text[:chunk_size]}\n"
        f"...[truncated {omitted_chars} chars]...\n"
        f"{text[-chunk_size:]}"
    )


def wrap_external_data(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"--- BEGIN DATA ---\n{body}\n--- END DATA ---"
