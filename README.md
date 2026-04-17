# strangeclaw

strangeclaw is a minimal, self-hosted autonomous AI agent designed around four
drivers: simplicity, security, maintainability, and expandability.

It runs a fully agentic loop:
`Inspect -> Choose -> Act -> Observe -> Repeat`.

## Quickstart (Yolo mode)

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
3. Configure the LLM in `~/.strangeclaw/config.yaml`:
   - Hosted API models: set `llm.api_key` either directly or via `${ENV_VAR}`.
   - Local models (LM Studio/Ollama): API key is optional and can be empty.
   - `llm.model` uses `{provider}/{model_name}` format.
     Examples:
     - Anthropic: `anthropic/claude-sonnet-4-20250514`
     - OpenAI: `openai/gpt-4.1-mini`
     - LM Studio: `lm_studio/qwen2.5-coder-7b-instruct`
4. Run the agent:
   ```bash
   .venv/bin/python -m main
   ```
5. Enter a task, review/approve the plan, and wait for the final `done` output.

Resume a saved session:
```bash
.venv/bin/python -m main --resume <session_id>
```

## Web Search Endpoint Override

`web-search` uses DuckDuckGo by default. You can override the search endpoint with:

```bash
export SC_WEB_SEARCH_ENDPOINT="https://your-endpoint.example/search"
```

Behavior:
- Applies to `web-search` `search` action only.
- If the value contains `{query}`, strangeclaw substitutes it with URL-encoded query text.
- If the value starts with `file://` or `data:`, it is used as-is.
- Otherwise, strangeclaw appends standard query params (`q`, `format`, `no_redirect`, `no_html`, `skip_disambig`).

Expected endpoint response is a JSON object. Supported fields:
- `AbstractText` + `AbstractURL` (optional top result)
- `RelatedTopics` array with items like `{"Text": "...", "FirstURL": "..."}` (supports nested `Topics`)

## Modes

- `yolo`: direct host execution for trusted local workflows.
- `fire`: Firecracker microVM isolation (in progress).

## Fire Mode Prerequisite Setup

Run host setup (installs/updates prerequisites and then runs checks):

```bash
bash scripts/setup-fire.sh
```

`setup-fire.sh` keeps IP forwarding unchanged by default (runtime-managed policy).
Use explicit flags only when you want setup to change it:

```bash
# Enable for current runtime only (no persistent sysctl file)
bash scripts/setup-fire.sh --enable-ip-forwarding-now

# Enable and persist in /etc/sysctl.d/99-strangeclaw-fire.conf
bash scripts/setup-fire.sh --persist-ip-forwarding
```

Run checks only (no host changes):

```bash
bash scripts/setup-fire.sh --check-only
```

Run the Fire prerequisite checker directly:

```bash
bash scripts/fire-check.sh
```

## Fire Mode With Local LLMs (Opt-In)

By default, Fire mode blocks guest-to-host traffic. To use a host-local LLM
server (Ollama, LM Studio), you must opt in with `firecracker.host_expose`.

Example config:

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
- In Fire mode, strangeclaw rewrites `localhost` / `127.0.0.1` `api_base` to
  the TAP gateway IP before sending config to the guest.
- Fire mode networking setup (`ip tuntap`, `iptables`) requires elevated
  privileges. Run Fire verification/usage commands with `sudo` and preserve your
  user config path:
  ```bash
  sudo --preserve-env=HOME .venv/bin/python scripts/verify_b24.py --goal "Say hello briefly." --approval-mode auto
  ```
- For security trade-offs of `host_expose`, see `strangeclaw_spec.md` §13.

### LM Studio Loopback Gotcha

LM Studio often listens on loopback only (`127.0.0.1:<port>`). In that case,
the Fire guest cannot reach it through exposed host ports.

Check listener:

```bash
ss -ltnp | grep 1234
```

If LM Studio is loopback-only, use a host proxy port that listens on all
interfaces:

```bash
sudo dnf install -y socat
socat TCP-LISTEN:1235,bind=0.0.0.0,reuseaddr,fork TCP:127.0.0.1:1234
```

Run the `socat` command in a separate terminal and keep it running while
strangeclaw is using Fire mode with the proxied local LLM.

Then point strangeclaw to the proxy port (`api_base: http://localhost:1235/v1`)
and expose that port (`host_expose.ports: [1235]`).

## Telegram Setup

1. Create a bot with BotFather:
   - Open Telegram and start a chat with `@BotFather`.
   - Run `/newbot`.
   - Follow prompts for bot name and username.
   - Copy the API token BotFather returns.
2. Update `~/.strangeclaw/config.yaml`:
   ```yaml
   adapters:
     enabled: [telegram]

   telegram:
     token: "123456789:AA..."
     local_mode: true
     allowed_chat_ids: []
   ```
3. Start strangeclaw:
   ```bash
   .venv/bin/python -m main
   ```
4. Open your bot chat in Telegram and send a task message.

`telegram.allowed_chat_ids` behavior:
- Empty or missing list: any Telegram chat can use the bot (good for local personal use).
- Non-empty list: only listed chat IDs are allowed; all other chats receive a polite "not authorized" reply.

Security note:
- Treat `telegram.token` as a secret like your LLM API key.
- Do not commit bot tokens to git or share them in logs/screenshots.

## Multiple Adapters

Enable multiple adapters in one process:

```yaml
adapters:
  enabled: [cli, telegram]
```

Notes:
- `--resume` is only allowed when exactly one adapter is enabled.
- In multi-adapter mode, Telegram session IDs are namespaced (`telegram-<chat_id>`) to avoid persistence collisions with other adapters.

## Telegram Session Behavior

When `telegram` is enabled in `adapters.enabled`, strangeclaw maps Telegram session identity to the
Telegram `chat_id`.

- A new task message in a chat starts a run for that chat.
- While a run is active, extra messages are only accepted when the agent is
  explicitly waiting for plan feedback or clarification.
- After `done`, the in-memory run state is cleared for that chat.
- Follow-up tasks in the same chat start a new run and automatically reuse the
  last completed task state from that chat (with re-planning forced).
- To create a truly separate session, use a different Telegram chat (different
  `chat_id`).

## Telegram Security Defaults

- For local development, keep `telegram.local_mode: true`.
- For non-local deployments, set `telegram.local_mode: false` and configure
  `telegram.allowed_chat_ids` to an explicit allowlist.
- Runtime limits are enforced:
  - `telegram.max_active_sessions`
  - `telegram.max_output_total_bytes`
  - `telegram.max_output_file_bytes`

## Validation

```bash
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest -q
```
