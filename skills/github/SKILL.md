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

The `http_request` tool injects the `Authorization`, `Accept`, and `X-GitHub-Api-Version` headers from config when `integration: "github"` is used. Do not include an `Authorization` header yourself.

## Workflow
1. Identify the GitHub owner, repository, and operation.
2. Ask for missing owner/repo/path/branch details before calling the API.
3. Use `http_request` with `integration: "github"` for private repositories, writes, and authenticated reads.
4. Follow pagination only until the requested data is complete.
5. Inspect status codes, response body `message`, and `documentation_url` fields on errors.
6. Summarize created issues, pull request lists, or file content findings without exposing credentials.

## References
Request `references/github.md` through `agent_read_skill_file` before making non-trivial GitHub calls. Do not read the reference with `shell`.
