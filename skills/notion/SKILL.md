---
name: notion
description: Work with the Notion API using configured auth, data sources, pages, querying, and updates.
version: 1.0.0
tags: [api, notion, workspace]
---

# Notion API

## When To Use
Use this skill when a task requires interacting with Notion through its REST API: creating pages, querying data sources, updating page properties, inspecting schemas, or diagnosing Notion API errors.

## Authentication
Use the `http_request` tool with `integration: "notion"` when the Notion token is configured under `integrations.notion.token`. Do not ask the user to paste raw Notion tokens into the task. If the integration is not available or a call reports that it is not configured, ask the user to configure `integrations.notion.token`.
If multiple Notion integrations are configured under different names, ask which integration name to use before making write calls.

The `http_request` tool injects the `Authorization` header and the configured `Notion-Version` header when `integration: "notion"` is used. Do not include an `Authorization` header yourself. Use base URL `https://api.notion.com`.

## Workflow
1. Identify the target Notion operation: create page, query data source, retrieve schema, update page, or update data source.
2. Check whether the task includes the required IDs, such as `data_source_id`, `page_id`, property names, or target title.
3. If property names or types are unknown, retrieve the data source schema before creating or updating pages.
4. Use `http_request` with `integration: "notion"` for all Notion API calls.
5. Inspect Notion error bodies and adjust request shape rather than repeating failed calls unchanged.
6. Summarize created/updated page IDs, URLs, and any partial failures without exposing credentials.

## IDs And Permissions
A Notion integration must have the needed capabilities, and the target page, database, or data source must be shared with the integration. A 404 can mean the resource exists but is not shared with the integration.

For modern Notion APIs, pages are entries in a data source. Prefer `data_source_id` for page creation and querying when the user is working with a Notion database table.

## Common Calls
Retrieve a data source schema before creating or updating pages when property names or types are unknown:

```json
{
  "method": "GET",
  "url": "https://api.notion.com/v1/data_sources/<data_source_id>",
  "integration": "notion",
  "headers": {},
  "body": null
}
```

Create a page in a data source with `POST /v1/pages`. Adjust property names and value shapes to match the retrieved schema:

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

Query a data source with `POST /v1/data_sources/{data_source_id}/query`. Add `filter`, `sorts`, and `filter_properties` when they narrow the result:

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

If the response includes `has_more: true` and `next_cursor`, continue with `start_cursor` until the requested data is complete.

Update page properties or trash/restore a page with `PATCH /v1/pages/{page_id}`. For current API versions, use `in_trash` for trash state:

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

Only send properties that should change. Generated properties such as rollups, created time, and last edited time cannot be set directly.

## Error Handling
Notion error bodies usually include `object`, `status`, `code`, and `message`.

- 400: property shape does not match schema, invalid filter, or version-specific behavior.
- 401: token missing or invalid.
- 403: integration lacks the required capability.
- 404: wrong ID or resource not shared with the integration.
- 429: rate limited; respect retry guidance if provided.
- 503: transient query/backend issue; retry with backoff or narrow the query.
