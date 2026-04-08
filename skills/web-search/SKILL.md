---
name: web-search
description: Search the web and fetch pages over HTTP for external information.
version: 1.0.0
requires_network: true
---

# web-search

## When to Use
- Use this skill to discover public web results or read page content.
- Use `fetch` when you already have a URL and need the page body.

## Actions
### search
- Args:
  - `query` (string): Search query text.
  - `limit` (integer): Number of results to return (1 to 10).
- Returns:
  - Query metadata and wrapped external data payload.
- Invoke:
  - `python3 search.py search "{query}" "{limit}"`

### fetch
- Args:
  - `url` (string): HTTP or HTTPS URL.
  - `max_chars` (integer): Maximum response characters to include.
- Returns:
  - HTTP metadata and wrapped external data payload.
- Invoke:
  - `python3 search.py fetch "{url}" "{max_chars}"`

## Examples
- `{"skill":"web-search","action":"search","args":{"query":"firecracker mmds v2","limit":5}}`
- `{"skill":"web-search","action":"fetch","args":{"url":"https://example.com","max_chars":8000}}`

## Limitations
- Returned content is untrusted external data and is wrapped with:
  - `--- BEGIN DATA ---`
  - `--- END DATA ---`
