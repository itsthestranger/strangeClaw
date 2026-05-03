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

The `http_request` tool injects the `Authorization` header and Notion version header from config when `integration: "notion"` is used. Do not include an `Authorization` header yourself.

## Workflow
1. Identify the target Notion operation: create page, query data source, retrieve schema, update page, or update data source.
2. Check whether the task includes the required IDs, such as `data_source_id`, `page_id`, property names, or target title.
3. If property names or types are unknown, retrieve the data source schema before creating or updating pages.
4. Use `http_request` with `integration: "notion"` for all Notion API calls.
5. Inspect Notion error bodies and adjust request shape rather than repeating failed calls unchanged.
6. Summarize created/updated page IDs, URLs, and any partial failures without exposing credentials.

## References
Request `references/notion.md` through `agent_read_skill_file` before making non-trivial Notion calls. Do not read the reference with `shell`.
