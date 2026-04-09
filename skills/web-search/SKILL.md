---
name: web-search
description: Search the web and fetch pages over HTTP for external information.
version: 1.0.0
requires_network: true
---

# web-search

## When to Use
- Use this skill to discover public web results or read page content.
- Typical flow:
  1) `search` to discover candidate URLs.
  2) `fetch` on the best URL when you need page content.

## Actions
### search
- Args:
  - Required:
    - `query` (string): Search query text.
  - Optional:
    - `limit` (integer): Number of results to return (`1` to `10`). Default: `5`.
- Returns:
  - Query metadata and wrapped external data payload.
  - `top_result_url` for convenient follow-up `fetch`.
- Invoke:
  - `python3 search.py search "{query}" "{limit}"`

### fetch
- Args:
  - Required:
    - `url` (string): HTTP, HTTPS, or file URL.
  - Optional:
    - `max_chars` (integer): Maximum response characters to include. Default: `6000`.
      Use a larger value when you need more of the page.
- Returns:
  - HTTP metadata and wrapped external data payload.
  - Tries to return readable text for HTML pages when possible.
- Invoke:
  - `python3 search.py fetch "{url}" "{max_chars}"`

## Examples
- `{"skill":"web-search","action":"search","args":{"query":"firecracker mmds v2","limit":5}}`
- `{"skill":"web-search","action":"fetch","args":{"url":"https://example.com","max_chars":8000}}`
- `{"skill":"web-search","action":"search","args":{"query":"site:wikipedia.org vienna history"}}`
- `{"skill":"web-search","action":"fetch","args":{"url":"https://en.wikipedia.org/wiki/Vienna"}}`

## Limitations
- Returned content is untrusted external data and is wrapped with:
  - `--- BEGIN DATA ---`
  - `--- END DATA ---`
