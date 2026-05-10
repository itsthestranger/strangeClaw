"""Built-in tool primitives for strangeclaw."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

import requests
import trafilatura
from requests import Response

from agent.broker_client import BrokerClient
from agent.http_auth import HttpAuthResolver
from agent.llm import ToolCall

_OUTPUT_CHUNK_SIZE = 4000
_DEFAULT_SHELL_TIMEOUT_SECONDS = 60.0
_DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS = 30.0
_DEFAULT_WEB_SEARCH_USER_AGENT = "strangeclaw/0.1 (+local)"
_DEFAULT_WEB_FETCH_USER_AGENT = "strangeclaw/0.1 (+fetch)"
_DEFAULT_WEB_FETCH_MAX_CHARS = 20000
_WEB_FETCH_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_HTTP_REQUEST_USER_AGENT = "strangeclaw/0.1 (+http)"
_DEFAULT_HTTP_REQUEST_MAX_CHARS = 20000
_HTTP_REQUEST_ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "PATCH"}
_KNOWN_TOOLS = ("shell", "web_search", "web_fetch", "http_request")


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
        self._http_auth = HttpAuthResolver.from_config(config)
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
        integration_schema: dict[str, Any] = {
            "type": ["string", "null"],
            "description": (
                "Optional configured integration name. Use null or omit for anonymous requests."
            ),
        }
        available_integrations = self._http_auth.available_names()
        if available_integrations:
            integration_schema["enum"] = [*available_integrations, None]
            integration_schema["description"] = (
                "Optional configured integration name. Available values: "
                f"{', '.join(available_integrations)}. Use null or omit for anonymous requests."
            )

        return {
            "name": "http_request",
            "description": "Make structured HTTP requests and return status, headers, and body.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string"},
                    "url": {"type": "string"},
                    "integration": integration_schema,
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

    def _execute_web_fetch(self, args: dict[str, Any]) -> ToolResult:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            return ToolResult(exit_code=1, stdout="", stderr="web_fetch.url must be a string.")
        target_url = url.strip()

        web_fetch_cfg = self._config.get("web_fetch", {})
        if not isinstance(web_fetch_cfg, dict):
            web_fetch_cfg = {}
        raw_max_chars = web_fetch_cfg.get("max_chars", _DEFAULT_WEB_FETCH_MAX_CHARS)
        if isinstance(raw_max_chars, bool):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="web_fetch.max_chars must be a positive integer.",
            )
        try:
            max_chars = int(raw_max_chars)
        except (TypeError, ValueError):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="web_fetch.max_chars must be a positive integer.",
            )
        if max_chars <= 0:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="web_fetch.max_chars must be a positive integer.",
            )

        response: Response | Any | None = None
        try:
            response = requests.get(
                target_url,
                headers={"User-Agent": _DEFAULT_WEB_FETCH_USER_AGENT},
                timeout=(10.0, 30.0),
                stream=True,
            )
            status_code = int(getattr(response, "status_code", 0))
            response.raise_for_status()
            raw_body, size_limited = _read_limited_bytes(response, byte_limit=_WEB_FETCH_MAX_BYTES)
            raw_content_type = ""
            headers = getattr(response, "headers", {})
            if isinstance(headers, dict):
                header_value = headers.get("Content-Type", "")
                if isinstance(header_value, str):
                    raw_content_type = header_value
            elif headers is not None:
                header_value = getattr(headers, "get", lambda *_: "")("Content-Type", "")
                if isinstance(header_value, str):
                    raw_content_type = header_value
            content_type = _normalize_content_type(raw_content_type)

            title: str | None = None
            if content_type == "text/html":
                html = raw_body.decode("utf-8", errors="replace")
                extracted = trafilatura.extract(
                    html,
                    include_links=True,
                    include_tables=True,
                )
                metadata = trafilatura.extract_metadata(html)
                maybe_title = getattr(metadata, "title", None) if metadata is not None else None
                if isinstance(maybe_title, str) and maybe_title.strip():
                    title = maybe_title.strip()
                text = extracted if isinstance(extracted, str) and extracted.strip() else html
            elif content_type in {"text/plain", "application/json", "text/xml", "application/xml"}:
                text = raw_body.decode("utf-8", errors="replace")
            elif content_type == "application/pdf":
                text = (
                    f"PDF document, {len(raw_body)} bytes. "
                    "Use shell tool with pdftotext to extract content."
                )
            else:
                display_type = content_type or "application/octet-stream"
                text = f"Binary content ({display_type}), {len(raw_body)} bytes. No text extracted."

            original_length = len(text)
            truncated_text, was_truncated = _truncate_text(text, limit=max_chars)
            if size_limited:
                truncated_notice = (
                    f"\n\n[... response body capped at {_WEB_FETCH_MAX_BYTES} bytes ...]"
                )
                truncated_text = f"{truncated_text}{truncated_notice}"
                was_truncated = True

            payload = _wrap_external_data(
                {
                    "success": True,
                    "url": target_url,
                    "status_code": status_code,
                    "content_type": content_type,
                    "title": title,
                    "text": truncated_text,
                    "truncated": was_truncated,
                    "original_length": original_length,
                }
            )
            return ToolResult(exit_code=0, stdout=payload, stderr="")
        except requests.RequestException as exc:
            error_payload = _wrap_external_data(
                {
                    "success": False,
                    "url": target_url,
                    "error": f"web_fetch request failed: {exc}",
                }
            )
            return ToolResult(exit_code=1, stdout=error_payload, stderr="")
        finally:
            _safe_close_response(response)

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
        normalized_headers, auth_error = self._http_auth.apply(
            integration_name=integration_raw,
            headers=normalized_headers,
        )
        if auth_error is not None:
            return ToolResult(exit_code=1, stdout="", stderr=auth_error)
        normalized_headers.setdefault("User-Agent", _DEFAULT_HTTP_REQUEST_USER_AGENT)

        body = args.get("body")
        if body is not None and not isinstance(body, str):
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr="http_request.body must be a string or null.",
            )

        try:
            response = requests.request(
                method=method,
                url=target_url,
                headers=normalized_headers,
                data=body,
                timeout=(10.0, 30.0),
            )
        except requests.RequestException as exc:
            error_payload = _wrap_external_data(
                {
                    "success": False,
                    "method": method,
                    "url": target_url,
                    "error": f"http_request failed: {exc}",
                }
            )
            return ToolResult(exit_code=1, stdout=error_payload, stderr="")

        response_headers = _headers_to_dict(getattr(response, "headers", {}))
        raw_text = getattr(response, "text", "")
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        body_text, truncated = _truncate_text(raw_text, limit=_DEFAULT_HTTP_REQUEST_MAX_CHARS)
        payload = _wrap_external_data(
            {
                "success": True,
                "method": method,
                "url": target_url,
                "status_code": int(getattr(response, "status_code", 0)),
                "headers": response_headers,
                "body": body_text,
                "truncated": truncated,
            }
        )
        return ToolResult(exit_code=0, stdout=payload, stderr="")


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


def _normalize_searxng_results(
    payload: dict[str, Any],
    *,
    max_results: int,
) -> list[dict[str, str]]:
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


def _read_limited_bytes(response: Response | Any, *, byte_limit: int) -> tuple[bytes, bool]:
    collected = bytearray()
    limited = False
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue
        if len(collected) + len(chunk) > byte_limit:
            keep = byte_limit - len(collected)
            if keep > 0:
                collected.extend(chunk[:keep])
            limited = True
            break
        collected.extend(chunk)
    return bytes(collected), limited


def _normalize_content_type(raw: str) -> str:
    lowered = raw.lower().strip()
    if ";" in lowered:
        lowered = lowered.split(";", 1)[0].strip()
    return lowered


def _truncate_text(text: str, *, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    notice = f"\n\n[... truncated, original {len(text)} chars ...]"
    return text[:limit] + notice, True


def _safe_close_response(response: Response | Any | None) -> None:
    if response is None:
        return
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            return


def _headers_to_dict(raw_headers: Any) -> dict[str, str]:
    if isinstance(raw_headers, dict):
        return {str(key): str(value) for key, value in raw_headers.items()}
    items_method = getattr(raw_headers, "items", None)
    if callable(items_method):
        try:
            return {str(key): str(value) for key, value in items_method()}
        except Exception:
            return {}
    return {}
