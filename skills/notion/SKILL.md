---
name: notion
description: Work with the Notion API using configured auth, data sources, pages, querying, and updates.
version: 1.0.0
tags: [api, notion, workspace]
---

# Notion API

## When To Use
Use this skill when a task requires interacting with Notion through its REST API: creating pages, querying data sources, updating page properties, inspecting schemas, or diagnosing Notion API errors.

## Integration And Broker Rules
Use `http_request` with `integration: "notion"` for all Notion API calls.

- Integration names are an authorization boundary. Use only names listed in execution context under `Configured integrations`.
- Never set `Authorization` or other protected auth headers directly. Credentials are injected by the host-side broker.
- Never ask the user to paste raw API tokens into task text.
- If `integration: "notion"` is unavailable or broker denies access, report inability with broker reason.

## Workflow
1. Identify the target Notion operation: create page, query data source, retrieve schema, update page, or update data source.
2. Check whether the task includes the required IDs, such as `data_source_id`, `page_id`, property names, or target title.
3. If property names or types are unknown, retrieve the data source schema before creating or updating pages.
4. Use `http_request` with `integration: "notion"` for all Notion API calls.
5. Inspect Notion error bodies and adjust request shape rather than repeating failed calls unchanged.
6. If broker returns `policy_denied`, include `requested_method`, `requested_url`, and `reason` in your report and do not retry the same denied request.
7. If broker returns `rate_limited`, wait and retry once, or report the limitation when waiting is not appropriate.
8. Summarize created/updated page IDs, URLs, and any partial failures.

## IDs And Permissions
A Notion integration must have the needed capabilities, and the target page, database, or data source must be shared with the integration. A 404 can mean the resource exists but is not shared with the integration.

For modern Notion APIs, pages are entries in a data source. Prefer `data_source_id` for page creation and querying when the user is working with a Notion database table.

## Endpoint Patterns And Request Shapes
Use base URL `https://api.notion.com`.

Retrieve a data source schema before creating or updating pages when property names or types are unknown (`GET /v1/data_sources/{data_source_id}`):

```json
{
  "method": "GET",
  "url": "https://api.notion.com/v1/data_sources/<data_source_id>",
  "integration": "notion",
  "headers": {},
  "body": null
}
```

Create a page in a data source (`POST /v1/pages`). `parent.data_source_id` and `properties` are required:

```json
{
  "method": "POST",
  "url": "https://api.notion.com/v1/pages",
  "integration": "notion",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"parent\":{\"data_source_id\":\"<data_source_id>\"},\"properties\":{\"Name\":{\"title\":[{\"text\":{\"content\":\"Meeting Notes\"}}]}}}"
}
```

Query a data source (`POST /v1/data_sources/{data_source_id}/query`). Include `filter`, `sorts`, and `filter_properties` when useful:

```json
{
  "method": "POST",
  "url": "https://api.notion.com/v1/data_sources/<data_source_id>/query",
  "integration": "notion",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"page_size\":50}"
}
```

Paginated query responses use `has_more` and `next_cursor`. Continue by sending `start_cursor` until results are complete.

Update page properties or trash state (`PATCH /v1/pages/{page_id}`). Send only fields to change:

```json
{
  "method": "PATCH",
  "url": "https://api.notion.com/v1/pages/<page_id>",
  "integration": "notion",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"properties\":{\"Status\":{\"status\":{\"name\":\"Done\"}}}}"
}
```

Generated properties (for example rollups, created time, and last edited time) cannot be set directly.

Retrieve a page (`GET /v1/pages/{page_id}`) to inspect current property values before patching.

## Error Handling
Notion error bodies usually include `object`, `status`, `code`, and `message`. Report these fields when present.

- 400: property shape does not match schema, invalid filter, or version-specific behavior.
- 401: integration credential is invalid or missing.
- 403: integration lacks required capability.
- 404: wrong ID or resource not shared with the integration.
- 409: conflict; retry only if safe.
- 429: rate limited; retry once with wait or report limitation.
- 503: transient query/backend issue; retry with backoff or narrow the query.
