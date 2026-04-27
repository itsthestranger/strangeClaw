"""Built-in tool primitives for strangeclaw."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

import requests

from agent.llm import ToolCall

_OUTPUT_CHUNK_SIZE = 4000
_DEFAULT_SHELL_TIMEOUT_SECONDS = 60.0
_DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS = 30.0
_DEFAULT_WEB_SEARCH_USER_AGENT = "strangeclaw/0.1 (+local)"
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
        self._config = dict(config)
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
        if "web_search" in self._enabled:
            schemas.append(self._web_search_schema())
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

    def _execute_web_search(self, args: dict[str, Any]) -> ToolResult:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return ToolResult(exit_code=1, stdout="", stderr="web_search.query must be a string.")

        web_search_cfg = self._config.get("web_search", {})
        if not isinstance(web_search_cfg, dict):
            web_search_cfg = {}
        endpoint = web_search_cfg.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="web_search endpoint is not configured.",
            )

        raw_format = web_search_cfg.get("format", "brave")
        search_format = str(raw_format).strip().lower()
        raw_max_results = web_search_cfg.get("max_results", 10)
        if isinstance(raw_max_results, bool):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="web_search.max_results must be a positive integer.",
            )
        try:
            max_results = int(raw_max_results)
        except (TypeError, ValueError):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="web_search.max_results must be a positive integer.",
            )
        if max_results <= 0:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="web_search.max_results must be a positive integer.",
            )

        request_kwargs: dict[str, Any] = {
            "timeout": _DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS,
            "headers": {"User-Agent": _DEFAULT_WEB_SEARCH_USER_AGENT},
        }
        if search_format == "brave":
            api_key = web_search_cfg.get("api_key")
            if not isinstance(api_key, str) or not api_key.strip():
                return ToolResult(
                    exit_code=1,
                    stdout="",
                    stderr="web_search.api_key is required when web_search.format is brave.",
                )
            request_kwargs["headers"] = {
                "User-Agent": _DEFAULT_WEB_SEARCH_USER_AGENT,
                "X-Subscription-Token": api_key.strip(),
            }
            request_kwargs["params"] = {"q": query.strip()}
        elif search_format == "searxng":
            request_kwargs["headers"] = {
                "User-Agent": _DEFAULT_WEB_SEARCH_USER_AGENT,
                "Accept": "application/json",
            }
            request_kwargs["params"] = {"q": query.strip(), "format": "json"}
        else:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr=f"Unsupported web_search.format: {search_format}",
            )

        try:
            response = requests.get(endpoint.strip(), **request_kwargs)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr=f"web_search request failed: {exc}",
            )
        except ValueError as exc:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr=f"web_search returned invalid JSON: {exc}",
            )

        if not isinstance(payload, dict):
            return ToolResult(exit_code=1, stdout="", stderr="web_search response must be JSON.")

        if search_format == "brave":
            results = _normalize_brave_results(payload, max_results=max_results)
        else:
            results = _normalize_searxng_results(payload, max_results=max_results)

        wrapped = _wrap_external_data({"query": query.strip(), "results": results})
        return ToolResult(exit_code=0, stdout=wrapped, stderr="")


def _truncate_output(text: str, *, chunk_size: int = _OUTPUT_CHUNK_SIZE) -> str:
    if len(text) <= chunk_size * 2:
        return text
    omitted_chars = len(text) - (chunk_size * 2)
    return (
        f"{text[:chunk_size]}\n"
        f"...[truncated {omitted_chars} chars]...\n"
        f"{text[-chunk_size:]}"
    )


def _normalize_brave_results(payload: dict[str, Any], *, max_results: int) -> list[dict[str, str]]:
    raw_web = payload.get("web")
    if not isinstance(raw_web, dict):
        return []
    raw_results = raw_web.get("results")
    if not isinstance(raw_results, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("description", "")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        if not isinstance(snippet, str):
            snippet = str(snippet)
        normalized.append({"title": title, "url": url, "snippet": snippet})
        if len(normalized) >= max_results:
            break
    return normalized


def _normalize_searxng_results(payload: dict[str, Any], *, max_results: int) -> list[dict[str, str]]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        url = item.get("url")
        snippet = item.get("content", "")
        if not isinstance(title, str) or not isinstance(url, str):
            continue
        if not isinstance(snippet, str):
            snippet = str(snippet)
        normalized.append({"title": title, "url": url, "snippet": snippet})
        if len(normalized) >= max_results:
            break
    return normalized


def _wrap_external_data(payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return f"--- BEGIN DATA ---\n{body}\n--- END DATA ---"
