# Fire Mode

Fire mode runs the strangeClaw agent inside a Firecracker microVM. The host owns
adapters, session coordination, Firecracker lifecycle, request-broker policy, and
LLM provider access. The guest owns the agent loop and enabled tools.

## Session Model

- One Firecracker VM runs per active session.
- The first task in a session starts the VM and waits for `agent_ready`.
- Follow-up tasks in the same session reuse the running VM.
- Files, installed tools, and other guest filesystem changes persist between
  tasks in the same session.
- Agent execution state resets per task; continuity is carried by `task.state`
  and by files that remain in the guest.
- Idle VMs are stopped after `firecracker.session_idle_timeout_seconds`
  seconds. The default is `1800`; set `0` to disable idle reaping.
- A new session starts from a fresh per-session rootfs copy. Fire mode does not
  support `--resume` across sessions.

The coordinator owns VM lifecycle. Task workers drive one task at a time and do
not stop the session VM when a task completes. `stop_session` and coordinator
shutdown stop the VM explicitly.

## Guest Internet

The guest has full NAT internet access through the host TAP device and
Firecracker network configuration. DNS is provided through the MMDS-delivered
network settings and written into the guest resolver config during boot.

The broker is not an internet access controller. Its job is credential injection
and policy enforcement for broker-backed tools:

- `http_request`
- `web_fetch`
- `web_search`

Unauthenticated public internet access can still happen directly from the guest
through enabled tools such as `shell`.

## Host Services

Fire mode uses the same host-services channel for broker calls and LLM calls.
The guest sends `broker_request` events over the existing vsock JSONL stream; the
host returns matching `broker_response` events. These transport events are not
shown to adapters and are not written to session state.

Registered services:

- `broker`: request broker for HTTP/search/API tool calls.
- `llm`: host-side LLM service used by the Fire guest.

The guest receives `host_services.llm_timeout_seconds` so it can wait long
enough for host-proxied model calls. `host_services.llm_max_request_bytes` is
host-only and enforced by the LLM service before provider calls.

## LLM Proxy

Fire guests use `LLMProxyRuntime`. The guest does not need LiteLLM or provider
configuration. It forwards `complete` and `count_tokens` requests to the host
`llm` service, and the host process calls the configured LiteLLM provider.

Local models work through the same proxy. For Ollama, configure the host:

```yaml
llm:
  model: ollama/llama3.1
  api_key: ""
  api_base: "http://127.0.0.1:11434"
```

The guest never needs access to host loopback ports. If the host LLM service
times out or returns an error, the proxy raises `LLMRuntimeError`; the agent
turns that into an `agent_decision_error` observation, appends it to history, and
lets the model decide the next step on a later turn.

## Subagents

When subagents are enabled, a child agent runs inside the **same** Firecracker VM
and session as the parent. It shares the guest filesystem (so it can read files
an earlier task or the parent created in the same session) and uses the **same**
host-side broker and LLM proxy as the parent. This introduces no host
credentials and no new host permissions: credential injection and LLM provider
access stay host-only, exactly as for the parent.

Children run one at a time, synchronously, so their broker and LLM calls use the
same vsock host-services channel with never more than one outstanding request.
`broker_request`/`broker_response` events and the child's internal events are not
shown to adapters and are not written to session state. Non-secret `subagents`
settings are delivered to the guest via MMDS; credentials and the LLM API key are
not. See [Architecture](architecture.md#subagents) for the full model.

## Request Broker

All authenticated API access goes through the host request broker. Credentials
live in `~/.strangeclaw/secrets.yaml` under `credentials.<name>`. The model can
request an integration by name, for example:

```json
{
  "integration": "github",
  "method": "GET",
  "url": "https://api.github.com/user/repos",
  "headers": {},
  "body": null
}
```

The broker validates the requested method, host, path, protected headers,
response-size cap, and rate limit before injecting credentials. Denials are
returned as observations for the model to reason about. Token values are redacted
before any broker result leaves the host broker.

`web_fetch` is intentionally a pass-through tool. It returns the HTTP response
as:

```json
{
  "success": true,
  "status_code": 200,
  "headers": {},
  "body": "<decoded UTF-8 response body>",
  "truncated": false
}
```

The host does not parse HTML, extract text, summarize content, or special-case
PDFs/images. If the agent needs content processing, it should do that inside the
guest with enabled tools.

`web_search` remains normalized because it consumes known search API JSON
formats. Results are returned as `{title, url, snippet}` items.

## Credential Isolation

Host-only:

- LLM provider credentials from `config.yaml`.
- External API/search tokens from `~/.strangeclaw/secrets.yaml`.
- Broker policy records and injected outbound request headers.
- `host_services.llm_max_request_bytes`.

Guest-visible:

- Tool toggles and loop/context settings.
- Non-secret `web_search` settings: endpoint, format, max results.
- `web_fetch.max_response_bytes`.
- `skills` settings.
- `host_services.llm_timeout_seconds`.
- Network settings needed to configure guest NAT/DNS.

MMDS contains no secret values. Fire startup calls `_assert_no_secrets()` before
writing MMDS data; it checks both integration token values and the configured LLM
API key value. Broker responses are also recursively redacted before they become
tool observations, action events, history, or persisted session state.

## Rootfs Rebuilds

Build the Fire rootfs with:

```bash
bash scripts/build-fire-rootfs.sh
```

Rebuild whenever guest code, built-in skills, guest dependencies, or
`firecracker/rootfs/entrypoint.sh` changes. A rebuild is especially required
after changes to:

- The guest config file path or MMDS payload shape.
- Guest network bootstrap in `entrypoint.sh`.
- Agent loop, tool, skill, transport, or proxy code copied into the guest.
- Guest Python dependencies.
- Built-in skills under `skills/`.

## Common Failures

`_web_search` absent:
The `web_search` tool returns a broker denial. Add `credentials._web_search` to
`~/.strangeclaw/secrets.yaml` with host/path policy matching your configured
search endpoint.

`credentials.<name>` absent:
Authenticated `http_request` calls with that integration name are denied. Add the
credential record and narrow policy fields in `secrets.yaml`.

Policy denied:
The broker returns `policy_denied` with the requested method, requested URL, and
reason. The model should not retry the same denied request unchanged.

Rate limited:
The broker returns `rate_limited`. The model can wait and retry once, or report
the limitation to the user.

LLM proxy timeout or service failure:
The guest observes `agent_decision_error` with category `llm_runtime_error`.
This is a model-visible observation, not a transport crash.

VM not ready:
If `agent_ready` is not received before `firecracker.boot_timeout`, Fire startup
fails with boot diagnostics including Firecracker log tail where available.

Stale Fire resources:
After abnormal host termination, inspect with:

```bash
sudo bash scripts/cleanup-fire.sh --dry-run
```

Then remove strangeClaw-owned resources with:

```bash
sudo bash scripts/cleanup-fire.sh
```
