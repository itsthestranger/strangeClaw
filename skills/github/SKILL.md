---
name: github
description: Work with the GitHub REST API using configured auth, issues, pull requests, and repository contents.
version: 1.0.0
tags: [api, github, repository]
---

# GitHub API

## When To Use
Use this skill when a task requires interacting with the GitHub REST API: creating issues, listing pull requests, reading repository contents, inspecting repository metadata, or diagnosing GitHub API errors.

## Authentication
Use the `http_request` tool with `integration: "github"` when the GitHub token is configured under `integrations.github.token`. Do not ask the user to paste raw GitHub tokens into the task. If the integration is not available or a call reports that it is not configured, ask the user to configure `integrations.github.token`.

The `http_request` tool injects the `Authorization`, `Accept`, and `X-GitHub-Api-Version` headers from config when `integration: "github"` is used. Do not include an `Authorization` header yourself. Use base URL `https://api.github.com`.

## Workflow
1. Identify the GitHub owner, repository, and operation.
2. Ask for missing owner/repo/path/branch details before calling the API.
3. Use `http_request` with `integration: "github"` for private repositories, writes, and authenticated reads.
4. Follow pagination only until the requested data is complete.
5. Inspect status codes, response body `message`, and `documentation_url` fields on errors.
6. Summarize created issues, pull request lists, or file content findings without exposing credentials.

## Repository Coordinates
Repository endpoints use path variables `owner` and `repo`. Ask the user for these if they are not in the task. Include `ref=<branch_or_sha>` when the task targets a specific branch, tag, or commit.

## Common Calls
Create an issue with `POST /repos/{owner}/{repo}/issues`:

```json
{
  "method": "POST",
  "url": "https://api.github.com/repos/<owner>/<repo>/issues",
  "integration": "github",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": "{\"title\":\"Bug report\",\"body\":\"Steps to reproduce...\"}"
}
```

Optional issue fields include `assignees`, `labels`, and `milestone` when the user provides them or asks for them.

List pull requests with `GET /repos/{owner}/{repo}/pulls`:

```json
{
  "method": "GET",
  "url": "https://api.github.com/repos/<owner>/<repo>/pulls?state=open&per_page=100",
  "integration": "github",
  "headers": {},
  "body": null
}
```

Useful query parameters include `state`, `head`, `base`, `sort`, `direction`, `per_page`, and `page`.

Read repository contents with `GET /repos/{owner}/{repo}/contents/{path}`. Omit `{path}` to list the repository root:

```json
{
  "method": "GET",
  "url": "https://api.github.com/repos/<owner>/<repo>/contents/<path>?ref=<branch_or_sha>",
  "integration": "github",
  "headers": {
    "Accept": "application/vnd.github.raw+json"
  },
  "body": null
}
```

Use `application/vnd.github.raw+json` when raw file contents are more useful than metadata. If using the default object response for a file, decode the Base64 `content` field before reasoning over the file text.

## Pagination
GitHub REST list endpoints commonly use `per_page` and `page`, and may include a `Link` response header with `rel="next"`. Continue pagination only until the requested information is complete. If `Link` contains a next URL, call that URL directly with the same headers and the same `integration`.

## Rate Limits And Errors
Important response headers can include rate-limit state such as remaining requests and reset time. For 403 responses, inspect the body and headers to distinguish missing permission from rate limiting.

- 401: token invalid or missing.
- 403: insufficient permission, secondary rate limit, or primary rate limit.
- 404: repository/resource missing or token lacks access to a private repository.
- 422: validation failed, often duplicate issue data or invalid field values.

Report GitHub's `message` field and any relevant `documentation_url`, but avoid echoing credentials.
