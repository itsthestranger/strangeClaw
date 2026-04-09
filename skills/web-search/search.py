#!/usr/bin/env python3
"""Web search and fetch helper."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

BEGIN_DATA = "--- BEGIN DATA ---"
END_DATA = "--- END DATA ---"
USER_AGENT = "strangeclaw-web-search/1.0"


def _wrap_external_data(payload: str) -> str:
    return f"{BEGIN_DATA}\n{payload}\n{END_DATA}"


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _read_json(url: str, timeout_seconds: float = 20.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw = response.read()
    parsed = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(parsed, dict):
        raise ValueError("Search API response must be a JSON object.")
    return parsed


def _collect_related(items: Any) -> list[dict[str, str]]:
    if not isinstance(items, list):
        return []

    results: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        nested = item.get("Topics")
        if isinstance(nested, list):
            results.extend(_collect_related(nested))
            continue
        text = item.get("Text")
        url = item.get("FirstURL")
        if isinstance(text, str) and isinstance(url, str) and text and url:
            results.append({"title": text, "url": url, "snippet": text})
    return results


def _search_endpoint(query: str) -> str:
    custom_endpoint = os.environ.get("SC_WEB_SEARCH_ENDPOINT")
    params = {
        "q": query,
        "format": "json",
        "no_redirect": "1",
        "no_html": "1",
        "skip_disambig": "1",
    }
    encoded_params = urllib.parse.urlencode(params)

    if custom_endpoint:
        if "{query}" in custom_endpoint:
            return custom_endpoint.format(query=urllib.parse.quote_plus(query))
        if custom_endpoint.startswith("file://") or custom_endpoint.startswith("data:"):
            return custom_endpoint
        separator = "&" if "?" in custom_endpoint else "?"
        return f"{custom_endpoint}{separator}{encoded_params}"

    return f"https://api.duckduckgo.com/?{encoded_params}"


def _action_search(query: str, limit: int) -> dict[str, Any]:
    endpoint = _search_endpoint(query)
    payload = _read_json(endpoint)

    entries: list[dict[str, str]] = []
    abstract_text = payload.get("AbstractText")
    abstract_url = payload.get("AbstractURL")
    if (
        isinstance(abstract_text, str)
        and isinstance(abstract_url, str)
        and abstract_text
        and abstract_url
    ):
        entries.append({"title": abstract_text, "url": abstract_url, "snippet": abstract_text})

    entries.extend(_collect_related(payload.get("RelatedTopics")))
    if limit < len(entries):
        entries = entries[:limit]

    wrapped = _wrap_external_data(json.dumps(entries, ensure_ascii=True, separators=(",", ":")))
    top_result_url: str | None = entries[0]["url"] if entries else None
    return {
        "query": query,
        "count": len(entries),
        "top_result_url": top_result_url,
        "data": wrapped,
    }


def _action_fetch(url: str, max_chars: int) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https", "file"}:
        raise ValueError("fetch url must start with http://, https://, or file://")

    request = urllib.request.Request(
        url,
        headers={"Accept": "*/*", "User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        body = response.read().decode("utf-8", errors="replace")
        raw_status = getattr(response, "status", None)
        status_code = int(raw_status) if isinstance(raw_status, int) else 200
        content_type = response.headers.get("Content-Type", "")
        final_url = response.geturl()

    body = _truncate(body, max_chars)
    wrapped = _wrap_external_data(body)
    return {
        "url": url,
        "final_url": final_url,
        "status_code": int(status_code),
        "content_type": content_type,
        "data": wrapped,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: search.py <search|fetch> ...", file=sys.stderr)
        return 2

    action = argv[1]
    try:
        if action == "search":
            if len(argv) != 4:
                raise ValueError("usage: search.py search <query> <limit>")
            query = argv[2]
            limit = int(argv[3])
            result = _action_search(query=query, limit=limit)
        elif action == "fetch":
            if len(argv) != 4:
                raise ValueError("usage: search.py fetch <url> <max_chars>")
            url = argv[2]
            max_chars = int(argv[3])
            result = _action_fetch(url=url, max_chars=max_chars)
        else:
            raise ValueError(f"unsupported action: {action}")
    except (ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
        print(f"web-search error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
