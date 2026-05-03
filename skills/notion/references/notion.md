# Notion API Patterns

Official docs:

- Create page: https://developers.notion.com/reference/post-page
- Retrieve data source: https://developers.notion.com/reference/retrieve-a-data-source
- Query data source: https://developers.notion.com/reference/query-a-data-source
- Update data source: https://developers.notion.com/reference/update-a-data-source
- Update page / trash page: https://developers.notion.com/reference/archive-a-page

## Base URL And Credential
Use base URL `https://api.notion.com`.

Use `http_request` with `integration: "notion"`. The tool injects:

- `Authorization: Bearer <integrations.notion.token>`
- `Notion-Version: <integrations.notion.default_headers.Notion-Version>`

Do not include raw tokens or an `Authorization` header in tool arguments.

## Permissions And IDs
A Notion integration must have the needed capabilities and the target page, database, or data source must be shared with the integration. A 404 can mean the resource exists but has not been shared with the integration.

For modern Notion APIs, pages are entries in a data source. Prefer `data_source_id` for page creation and querying when the user is working with a Notion database table.

## Create A Page In A Data Source
Use `POST /v1/pages`.

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

Adjust property names to match the target data source schema. If unsure, retrieve the data source first and inspect `properties`.

## Retrieve Data Source Schema
Use this before creating or updating pages when property names or types are unknown:

```json
{
  "method": "GET",
  "url": "https://api.notion.com/v1/data_sources/<data_source_id>",
  "integration": "notion",
  "headers": {},
  "body": null
}
```

Read the returned `properties` object and use exact property names and expected property value shapes.

## Query A Data Source
Use `POST /v1/data_sources/{data_source_id}/query`.

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

Add `filter`, `sorts`, and `filter_properties` when they narrow the result. If the response includes `has_more: true` and `next_cursor`, continue with `start_cursor` until the requested data is complete.

## Update A Page
Use `PATCH /v1/pages/{page_id}` to update properties or trash/restore a page. For current API versions, use `in_trash` for trash state.

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

## Error Notes
Notion error bodies usually include `object`, `status`, `code`, and `message`. Common causes:

- 400: property shape does not match schema, invalid filter, invalid version behavior.
- 401: token missing or invalid.
- 403: integration lacks the required capability.
- 404: wrong ID or resource not shared with the integration.
- 429: rate limited; respect retry guidance if provided.
- 503: transient query/backend issue; retry with backoff or narrow the query.
