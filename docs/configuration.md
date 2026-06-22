# Configuration

strangeClaw loads config from `~/.strangeclaw/config.yaml`, falling back to
`config.example.yaml` when local config is absent.

## LLMs

Configure the model under `llm`:

```yaml
llm:
  model: anthropic/claude-sonnet-4-20250514
  api_key: ${ANTHROPIC_API_KEY}
  api_base: null
  max_tokens: 4096
  temperature: 0.2
```

Local models work by setting `api_base` for the host-side provider endpoint:

```yaml
llm:
  model: ollama/llama3.1
  api_key: ""
  api_base: "http://127.0.0.1:11434"
```

In Fire mode, the guest calls the host-side LLM proxy. The guest does not need
direct access to host loopback ports.

## Tools

Enable or disable tools in `config.yaml`:

```yaml
tools:
  shell: true
  web_search: true
  web_fetch: true
  http_request: true
  spawn_subagent: false
```

Disabled tools are removed from the model-facing action surface. `spawn_subagent`
is an orchestration capability rather than a normal tool; see
[Subagents](#subagents).

## Web Search

`web_search` uses `config.yaml` for endpoint selection and
`~/.strangeclaw/secrets.yaml` for optional credentials.

Brave example:

```yaml
web_search:
  endpoint: "https://api.search.brave.com/res/v1/web/search"
  format: "brave"
  max_results: 10
```

Credential entry:

```yaml
credentials:
  _web_search:
    auth_type: header
    header_name: X-Subscription-Token
    token: "..."
    allowed_hosts: ["api.search.brave.com"]
    allowed_methods: [GET]
    allowed_paths: ["/*"]
```

SearXNG example:

```yaml
web_search:
  endpoint: "http://localhost:8080/search"
  format: "searxng"
  max_results: 10
```

For unauthenticated local SearXNG, keep a `_web_search` entry so the broker has
an explicit host/path policy. Use a non-secret placeholder token such as
`unused-local-searxng-token`.

SearXNG must allow JSON output:

```yaml
search:
  formats:
    - html
    - json
```

## Web Fetch

`web_fetch` returns the raw HTTP response as decoded text:

```yaml
web_fetch:
  max_response_bytes: 524288
```

The model receives `success`, `status_code`, `headers`, `body`, and
`truncated`.

## External API Integrations

All HTTP egress for `http_request`, `web_fetch`, and `web_search` goes through
the host-side request broker.

For authenticated APIs, the model calls `http_request` with an integration name:

```json
{
  "integration": "github",
  "method": "GET",
  "url": "https://api.github.com/user/repos",
  "headers": {},
  "body": null
}
```

The broker executes the call only if `~/.strangeclaw/secrets.yaml` contains a
matching `credentials.github` entry. It validates method, host, path, protected
headers, response size, and rate limits before injecting credentials.

Integration auth supports `bearer` and `header` modes. Query-string credential
injection is not supported.

## Secrets

Create the host-side secrets file from the example:

```bash
cp secrets.example.yaml ~/.strangeclaw/secrets.yaml
chmod 600 ~/.strangeclaw/secrets.yaml
```

Common entries:

- `credentials._web_search` for Brave or SearXNG-backed `web_search`.
- `credentials.notion` for the Notion skill.
- `credentials.github` for the GitHub skill.

Policy fields in `secrets.yaml` are the authorization boundary. Keep
`allowed_hosts`, `allowed_methods`, and `allowed_paths` narrow.

## Skills

If `skills` is omitted, strangeClaw defaults to:

```yaml
skills:
  directory: ./skills
  max_file_chars: 20000
```

Skills are loaded from `skills.directory`. Reference files read through
`agent_read_skill_file` are truncated at `max_file_chars`.

## Subagents

Subagents are disabled by default and gated by two switches. Both must be true
before the parent can delegate to a child:

```yaml
tools:
  spawn_subagent: false    # the model can see the capability

subagents:
  enabled: false           # the runtime will run a child
  max_children_per_task: 3
  max_iterations: 20
  timeout_seconds: 600
  max_context_chars: 20000
  max_result_chars: 20000
  max_files_bytes: 10485760
  journal_events: summary  # none | summary | full
```

Fields:

- `max_children_per_task`: total children one task may spawn, across all its
  turns (not a batch size — each `spawn_subagent` is one decision).
- `max_iterations`: child loop cap; a per-call `max_iterations` is clamped to it.
- `timeout_seconds`: child time budget, enforced at iteration boundaries; a
  per-call `timeout_seconds` is clamped to it.
- `max_context_chars`: cap on the parent-supplied context passed to a child.
- `max_result_chars`: cap on the child report the parent observes.
- `max_files_bytes`: cap on total child output bytes; an oversize child output
  becomes a bounded failure observation.
- `journal_events`: how much child-event detail rides in the observation and
  journal (`none`, `summary`, or `full`); all modes are bounded.

Children run sequentially in the same sandbox/session as the parent, with a
subset of the parent's tools, and cannot ask the user. In this release
subagents help with context economy and focus, not speed. See
[Architecture](architecture.md#subagents) for the runtime model and guidance.

## Telegram

1. Create a bot with `@BotFather` and copy the token.
2. Configure:

   ```yaml
   adapters:
     enabled: [telegram]

   telegram:
     token: "123456789:AA..."
     local_mode: true
     allowed_chat_ids: []
   ```

3. Run:

   ```bash
   .venv/bin/python -m main
   ```

`telegram.allowed_chat_ids` behavior:

- Empty or missing: any Telegram chat can use the bot.
- Non-empty: only listed chat IDs are allowed.

## Multiple Adapters

Enable multiple adapters in one process:

```yaml
adapters:
  enabled: [cli, telegram]
```

Notes:

- `--resume` is only allowed when exactly one adapter is enabled.
- In multi-adapter mode, Telegram session IDs are namespaced as
  `telegram-<chat_id>`.
