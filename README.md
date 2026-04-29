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
    -> Agent loop (plan -> act -> observe -> iterate)
      -> Skills + LLM calls inside sandbox
```

## Web Search Endpoint Override

`web-search` uses DuckDuckGo by default. Override with:

```bash
export SC_WEB_SEARCH_ENDPOINT="https://your-endpoint.example/search"
```

Behavior:
- Applies only to `web-search` `search` action.
- If value contains `{query}`, strangeclaw substitutes URL-encoded query text.
- If value starts with `file://` or `data:`, it is used as-is.
- Otherwise, strangeclaw appends standard query params (`q`, `format`,
  `no_redirect`, `no_html`, `skip_disambig`).

Expected endpoint response: JSON object with optional `AbstractText`,
`AbstractURL`, and `RelatedTopics` entries.

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
