---
name: http-request
description: Perform structured HTTP requests and return wrapped external response data.
version: 1.0.0
requires_network: true
---

# http-request

## When to Use
- Use this skill when you need deterministic HTTP requests with explicit method, headers, and body.
- Prefer this skill over shell curl for machine-structured calls.

## Actions
### request
- Args:
  - `method` (string): `GET`, `POST`, `PUT`, `PATCH`, `DELETE`, `HEAD`, or `OPTIONS`.
  - `url` (string): HTTP or HTTPS URL.
  - `headers` (object): String header map.
  - `body_json` (object | array | string | number | boolean | null): Request body payload.
  - `timeout_seconds` (number): Request timeout in seconds.
  - `max_chars` (integer): Maximum response body characters to include.
- Returns:
  - Request metadata, status code, and wrapped external response payload.
- Invoke:
  - `python3 request.py "{method}" "{url}" "{headers}" "{body_json}" "{timeout_seconds}" "{max_chars}"`

## Examples
- `{"skill":"http-request","action":"request","args":{"method":"GET","url":"https://example.com","headers":{},"body_json":null,"timeout_seconds":20,"max_chars":8000}}`
- `{"skill":"http-request","action":"request","args":{"method":"POST","url":"https://httpbin.org/post","headers":{"X-Trace":"abc"},"body_json":{"q":"hello"},"timeout_seconds":20,"max_chars":8000}}`

## Limitations
- Returned content is untrusted external data and is wrapped with:
  - `--- BEGIN DATA ---`
  - `--- END DATA ---`
