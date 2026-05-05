# strangeClaw

Yes, I know another OpenClaw alternative is not what the world needs right now.
But OpenClaw is a really cool project, I love the idea of AI assistants, and I wanted to understand it better.

I also really wanted to try whether it is possible to run agents inside a Firecracker microVM - I kinda like my PC and I am not trying to let an agent freestyle on my files. That is why I
created my own claw: strangeClaw.

strangeClaw is not a production-ready tool by any means, and it also does not have the skills or connections OpenClaw has, at least not yet.

Right now it supports Telegram and two different modes: `yolo` and `fire`.
`yolo` is more for trying things out quickly, and `fire` is where the Firecracker sandbox comes in.


## Quick Setup

### Modes

- `yolo`: direct host execution for trusted local workflows.
- `fire`: Firecracker microVM isolation.

### Yolo Mode Quickstart

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e ".[dev]"
   ```
2. Create local config:
   ```bash
   mkdir -p ~/.strangeclaw
   cp config.example.yaml ~/.strangeclaw/config.yaml
   ```
3. In `~/.strangeclaw/config.yaml`, set at minimum:
   ```yaml
   mode: yolo
   adapters:
     enabled: [cli]

   llm:
     model: anthropic/claude-sonnet-4-20250514
     api_key: ${ANTHROPIC_API_KEY}
   ```
4. Run strangeclaw:
   ```bash
   .venv/bin/python -m main
   ```
5. Enter a task, review/approve the plan, and wait for the final `done` output.

Resume a saved session:
```bash
.venv/bin/python -m main --resume <session_id>
```

### Fire Mode Setup

Run host setup (installs/updates prerequisites and runs checks):

```bash
bash scripts/setup-fire.sh
```

Useful variants:

```bash
# Checks only (no host changes)
bash scripts/setup-fire.sh --check-only

# Direct prerequisite checker
bash scripts/fire-check.sh

# Optional: enable IP forwarding for current runtime only
bash scripts/setup-fire.sh --enable-ip-forwarding-now

# Optional: enable + persist IP forwarding
bash scripts/setup-fire.sh --persist-ip-forwarding
```

Switch config to Fire mode:

```yaml
mode: fire
adapters:
  enabled: [cli]
```

Run with elevated privileges (required for TAP + iptables):

```bash
sudo --preserve-env=HOME .venv/bin/python -m main
```

If your API key is from env interpolation, preserve it too:

```bash
sudo --preserve-env=HOME,ANTHROPIC_API_KEY .venv/bin/python -m main
```

## Fire Mode Behavior (Current)

- Fire mode is currently per-task ephemeral: each task boots a fresh microVM,
  runs until `done`, then tears down.
- Follow-up tasks reuse host-side persisted state, not a still-running guest VM.
- `--resume` is intentionally rejected in Fire mode.

## Local LLMs In Fire Mode (Opt-In)

By default, Fire mode blocks guest-to-host traffic.
To use a host-local LLM server (Ollama/LM Studio), opt in with:

```yaml
llm:
  model: lm_studio/your-model-id
  api_key: ""
  api_base: "http://localhost:1235/v1"

firecracker:
  host_expose:
    enabled: true
    ports: [1235]
```

Notes:
- In Fire mode, `localhost` / `127.0.0.1` in `llm.api_base` is rewritten to the
  TAP gateway IP before being sent to the guest.
- Enabling `host_expose` weakens isolation; read the security analysis in
  `strangeclaw_spec.md` §13 before using it.

LM Studio loopback gotcha:
- If LM Studio listens only on `127.0.0.1:1234`, expose a host proxy port:
  ```bash
  socat TCP-LISTEN:1235,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:1234
  ```


## Features

- Fully agentic loop with plan/review, clarification, execution, and completion.
- Design decision (current): if planning references unknown skills, the agent will
  replan up to 3 times, then fail fast instead of looping indefinitely (may
  become configurable later).
- Provider-agnostic LLM layer via LiteLLM (`anthropic`, `openai`, `lm_studio`, `ollama`, and others).
- Pluggable skills loaded from `skills/<name>/` via `SKILL.md` frontmatter + optional bundled files.
- Two execution modes:
  - `yolo`: direct host execution for trusted workflows.
  - `fire`: Firecracker microVM isolation.
- CLI and Telegram adapters (including multi-adapter runs).
- Session persistence (`state.json`) plus output file export.
- Optional redacted session event journal (`events.jsonl`).
- Optional Firecracker runtime log artifact export to session outputs.
- Optional Fire-mode local LLM routing via `firecracker.host_expose`.

## Architecture

High-level runtime shape:

```text
Host (main/coordinator/adapters)
  -> Sandbox (Yolo or Fire)
    -> Agent loop (plan -> inspect -> choose -> act -> observe -> repeat)
      -> Skills + LLM calls inside sandbox
```

## Agentic Loop Contract

- Every execution turn must return exactly one structured decision (`tool` + `args`).
- During execution, free-form assistant prose is not a valid decision.
- Model-issued control decisions are:
  - `agent_done` (finish with `args.reply`)
  - `agent_clarify` (ask user input)
  - `agent_replan` (request a new plan)
  - `agent_read_skill_file` (stage-3 read from an activated skill only)
- The runtime does not choose actions for the model; it only validates, executes, and feeds observations back into history.
- Runtime-owned safety exits still exist (iteration guard, stop/shutdown, sandbox/runtime failures).

## Web Search Configuration

`web_search` is configured via `config.yaml` (not environment-variable override).
Set:

```yaml
web_search:
  endpoint: "https://api.search.brave.com/res/v1/web/search"
  format: "brave"      # or "searxng"
  integration: "brave_search"  # required for brave; host-only credential name
  max_results: 10
```

Behavior:
- `format: brave` requires `web_search.integration` and sends the query through the request broker.
- `format: searxng` sends `q=<query>&format=json`.
- For `searxng`, integration is optional (anonymous brokered request by default).
- Results are normalized into `{title, url, snippet}` for the model.

Brave credentials are host-only:
- Put Brave credentials in `~/.strangeclaw/secrets.yaml` as a broker credential
  (for example `credentials.brave_search`) and reference that name via
  `web_search.integration`.
- `web_search.api_key` in `config.yaml` is deprecated and rejected.

SearXNG local setup note:
- I run SearXNG from Docker using the official container installation guide:
  https://docs.searxng.org/admin/installation-docker.html
- The standard Compose setup exposes SearXNG on `http://localhost:8080`.
  Configure strangeclaw like this:
  ```yaml
  web_search:
    endpoint: "http://localhost:8080/search"
    format: "searxng"
    integration: null
    max_results: 10
  ```
- In SearXNG's `settings.yml`, allow JSON output or requests with
  `format=json` will be rejected. The relevant setting is:
  ```yaml
  search:
    formats:
      - html
      - json
  ```

Current direction:
- `http_request` and `web_fetch` are brokered, and `web_search` is brokered as well.
- Remaining broker work focuses on Fire transport parity, observability, and rate limits.

## Skills Config Defaults

If `skills` is omitted in `config.yaml`, strangeclaw defaults to:

```yaml
skills:
  directory: ./skills
  max_file_chars: 20000
```

## Telegram Setup

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
- Empty/missing: any Telegram chat can use the bot.
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
  `telegram-<chat_id>` to avoid collisions.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).
