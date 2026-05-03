# GitHub REST API Patterns

Official docs:

- Issues: https://docs.github.com/en/rest/issues/issues
- Pull requests: https://docs.github.com/en/rest/pulls/pulls
- Repository contents: https://docs.github.com/en/rest/repos/contents

## Base URL And Credential
Use base URL `https://api.github.com`.

Use `http_request` with `integration: "github"`. The tool injects:

- `Authorization: Bearer <integrations.github.token>`
- `Accept: application/vnd.github+json`
- `X-GitHub-Api-Version: <integrations.github.default_headers.X-GitHub-Api-Version>`

Do not include raw tokens or an `Authorization` header in tool arguments.

Repository endpoints use path variables `owner` and `repo`. Ask the user for these if they are not in the task.

## Create An Issue
Use `POST /repos/{owner}/{repo}/issues`.

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

Optional fields include `assignees`, `labels`, and `milestone` when the user provides them or asks for them.

## List Pull Requests
Use `GET /repos/{owner}/{repo}/pulls`.

```json
{
  "method": "GET",
  "url": "https://api.github.com/repos/<owner>/<repo>/pulls?state=open&per_page=100",
  "integration": "github",
  "headers": {},
  "body": null
}
```

Useful query parameters: `state`, `head`, `base`, `sort`, `direction`, `per_page`, and `page`.

## Read Repository Contents
Use `GET /repos/{owner}/{repo}/contents/{path}`. Omit `{path}` to list the repository root.

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
GitHub REST list endpoints commonly use `per_page` and `page`, and may include a `Link` response header with `rel="next"`. Continue pagination only until the requested information is complete.

If `Link` contains a next URL, call that URL directly with the same headers and the same `integration`.

## Rate Limits And Errors
Important response headers can include rate-limit state such as remaining requests and reset time. For 403 responses, inspect the body and headers to distinguish missing permission from rate limiting.

Common handling:

- 401: token invalid or missing.
- 403: insufficient permission, secondary rate limit, or primary rate limit.
- 404: repository/resource missing or token lacks access to a private repository.
- 422: validation failed, often duplicate issue data or invalid field values.

Report GitHub's `message` field and any relevant `documentation_url`, but avoid echoing credentials.
