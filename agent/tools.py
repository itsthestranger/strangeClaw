"""Built-in tool primitives for strangeclaw."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Any

from agent.broker_client import BrokerClient, HostServiceError
from agent.llm import ToolCall

_OUTPUT_CHUNK_SIZE = 4000
_DEFAULT_SHELL_TIMEOUT_SECONDS = 60.0
_HTTP_REQUEST_ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}
_KNOWN_TOOLS = ("shell", "web_search", "web_fetch", "http_request")
LOGGER = logging.getLogger(__name__)


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
        self._warn_if_deprecated_web_search_key_present()
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

    def _warn_if_deprecated_web_search_key_present(self) -> None:
        web_search = self._config.get("web_search")
        if not isinstance(web_search, dict):
            return
        api_key = web_search.get("api_key")
        if isinstance(api_key, str) and api_key.strip():
            LOGGER.warning(
                "web_search.api_key in config.yaml is deprecated and ignored. "
                "Move it to secrets.yaml under credentials._web_search.token."
            )

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
            "description": "Fetch a URL and extract readable content.",
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
        if self._broker is None:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="host service broker is not configured for web_search.",
            )
        try:
            result = self._broker.call(
                "broker",
                {
                    "action": "web_search",
                    "query": query.strip(),
                    "max_results": max_results,
                },
            )
        except HostServiceError as exc:
            return ToolResult(exit_code=1, stdout="", stderr=str(exc))
        wrapped = _wrap_external_data(result)
        if result.get("success") is False:
            return ToolResult(exit_code=1, stdout=wrapped, stderr="")
        return ToolResult(exit_code=0, stdout=wrapped, stderr="")

    def _execute_web_fetch(self, args: dict[str, Any]) -> ToolResult:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return ToolResult(exit_code=1, stdout="", stderr="web_fetch.url must be a string.")
        if self._broker is None:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="host service broker is not configured for web_fetch.",
            )
        try:
            result = self._broker.call("broker", {"action": "web_fetch", "url": url.strip()})
        except HostServiceError as exc:
            return ToolResult(exit_code=1, stdout="", stderr=str(exc))
        wrapped = _wrap_external_data(result)
        if result.get("success") is False:
            return ToolResult(exit_code=1, stdout=wrapped, stderr="")
        return ToolResult(exit_code=0, stdout=wrapped, stderr="")

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
        if self._broker is None:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="host service broker is not configured for http_request.",
            )

        try:
            result = self._broker.call(
                "broker",
                {
                    "action": "http_request",
                    "integration": integration_raw,
                    "method": method,
                    "url": target_url,
                    "headers": normalized_headers,
                    "body": body,
                },
            )
        except HostServiceError as exc:
            return ToolResult(exit_code=1, stdout="", stderr=str(exc))
        result_payload = dict(result)
        body_value = result_payload.get("body", "")
        result_payload["body"] = _wrap_external_data(
            {"body": body_value if isinstance(body_value, str) else str(body_value)}
        )
        wrapped = _wrap_external_data(result_payload)
        if result_payload.get("success") is False:
            return ToolResult(exit_code=1, stdout=wrapped, stderr="")
        return ToolResult(exit_code=0, stdout=wrapped, stderr="")

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


def _wrap_external_data(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"--- BEGIN DATA ---\n{body}\n--- END DATA ---"
