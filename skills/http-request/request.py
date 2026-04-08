#!/usr/bin/env python3
"""Structured HTTP request helper."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

BEGIN_DATA = "--- BEGIN DATA ---"
END_DATA = "--- END DATA ---"
USER_AGENT = "strangeclaw-http-request/1.0"


def _wrap_external_data(payload: str) -> str:
    return f"{BEGIN_DATA}\n{payload}\n{END_DATA}"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _load_json_arg(raw: str, label: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON for {label}") from exc


def _validate_url(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https", "file"}:
        raise ValueError("url must start with http://, https://, or file://")


def _build_request_data(
    body_json: Any, headers: dict[str, str]
) -> bytes | None:
    if body_json is None:
        return None
    if isinstance(body_json, str):
        headers.setdefault("Content-Type", "text/plain; charset=utf-8")
        return body_json.encode("utf-8")

    headers.setdefault("Content-Type", "application/json")
    encoded = json.dumps(body_json, ensure_ascii=True, separators=(",", ":"))
    return encoded.encode("utf-8")


def _make_response_payload(
    *,
    method: str,
    url: str,
    status_code: int,
    headers: dict[str, str],
    body: str,
) -> dict[str, Any]:
    wrapped_body = _wrap_external_data(
        json.dumps({"headers": headers, "body": body}, ensure_ascii=True)
    )
    return {
        "method": method,
        "url": url,
        "status_code": status_code,
        "data": wrapped_body,
    }


def main(argv: list[str]) -> int:
    if len(argv) != 7:
        print(
            "usage: request.py METHOD URL HEADERS_JSON BODY_JSON TIMEOUT_SECONDS MAX_CHARS",
            file=sys.stderr,
        )
        return 2

    method = argv[1].upper()
    url = argv[2]
    headers_obj = _load_json_arg(argv[3], "headers")
    body_json = _load_json_arg(argv[4], "body_json")

    try:
        timeout_seconds = float(argv[5])
    except ValueError as exc:
        raise ValueError("timeout_seconds must be a number") from exc

    try:
        max_chars = int(argv[6])
    except ValueError as exc:
        raise ValueError("max_chars must be an integer") from exc

    if not isinstance(headers_obj, dict):
        raise ValueError("headers must decode to a JSON object")
    headers: dict[str, str] = {}
    for key, value in headers_obj.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("headers must contain only string keys and values")
        headers[key] = value

    _validate_url(url)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    headers.setdefault("User-Agent", USER_AGENT)
    request_data = _build_request_data(body_json=body_json, headers=headers)
    request = urllib.request.Request(url=url, data=request_data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw_status = getattr(response, "status", None)
            status_code = int(raw_status) if isinstance(raw_status, int) else 200
            response_headers = {key: value for key, value in response.headers.items()}
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status_code = exc.code
        response_headers = {key: value for key, value in exc.headers.items()}
        body = exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        print(f"http-request error: {exc}", file=sys.stderr)
        return 2

    body = _truncate(body, max_chars)
    result = _make_response_payload(
        method=method,
        url=url,
        status_code=status_code,
        headers=response_headers,
        body=body,
    )
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except ValueError as exc:
        print(f"http-request error: {exc}", file=sys.stderr)
        sys.exit(2)
