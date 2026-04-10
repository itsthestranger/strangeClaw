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

## Telegram Session Behavior

When using `adapter: telegram`, strangeclaw currently maps `session_id` to the
Telegram `chat_id`.

- A new task message in a chat starts a run for that chat.
- While a run is active, extra messages are only accepted when the agent is
  explicitly waiting for plan feedback or clarification.
- After `done`, the in-memory run state is cleared for that chat.
- Follow-up tasks in the same chat start a new run (no automatic resume of the
  prior task state).
- To create a truly separate session, use a different Telegram chat (different
  `chat_id`).

## Validation

```bash
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/pytest -q
```
