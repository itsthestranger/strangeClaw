# strangeClaw

strangeClaw is a small, self-hosted autonomous agent experiment. I started it to
understand how agent systems are built, where their limitations are, and where
they can be improved.

The agent accepts a task, plans, executes tools, observes results, and repeats
until the model chooses to finish, ask for clarification, or replan. One idea I
wanted to try was running the agent inside a **Firecracker microVM** and keeping
credentials on the host behind a **request broker**.

## Status

strangeClaw is work in progress and not production-ready. It is a personal
project with a working Yolo mode, a mode where the agent runs inside a
Firecracker microVM, and a design that is still expected to evolve.

## Why This Exists

The project is a way for me to learn by building. I wanted to build my own agent
loop instead of using an agent framework, then use that as a base to try security
ideas that seemed worth exploring.

The main security experiment is running the agent inside a **Firecracker
microVM**, so commands and tools run behind a VM boundary instead of directly on
the host. The **request broker** is another part of that: keep API credentials
on the host, inject them only when a request passes policy, and let the agent
observe denials instead of giving it direct access to secrets.

It is built around four constraints:

- Simplicity: a small Python codebase with explicit module boundaries.
- Security: the agent can run inside a **Firecracker microVM**; credentials stay
  on the host behind the **request broker**.
- Maintainability: tools are fixed framework capabilities, skills are plain
  workflow documents.
- Expandability: adapters, skills, and host services can be added without
  changing the core loop.

## Architecture

```text
HOST
┌────────────────────────────────────────────────────────────────┐
│ User / Adapter / Coordinator                                   │
│                                                                │
│ Host secrets.yaml ──► Request Broker ──► External APIs         │
│                    policy check                                │
│                    credential injection                        │
│                    response redaction                          │
│                                                                │
│ SANDBOX                                                        │
│ ┌────────────────────────────────────────────────────────────┐ │
│ │ Agent loop                                                 │ │
│ │ Inspect → Choose → Act → Observe → Repeat                  │ │
│ │                                                            │ │
│ │ Tools + skills context                                     │ │
│ │                                                            │ │
│ │ In Fire mode: no host filesystem, no API secrets,          │ │
│ │ no LLM credentials. Risky work stays inside the VM.        │ │
│ └────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

## What It Supports

- A strict agentic loop: Inspect -> Choose -> Act -> Observe -> Repeat.
- Two execution modes:
  - `yolo`: direct host execution for trusted local workflows.
  - `fire`: Firecracker microVM isolation.
- CLI and Telegram adapters.
- Provider-agnostic LLM access through LiteLLM.
- Host-side LLM proxy in Fire mode, so LLM credentials stay off the guest.
- Host-side request broker for `web_search`, `web_fetch`, and `http_request`.
- Skills loaded from `skills/<name>/SKILL.md` using the Agent Skills format.
- Per-session state, output files, optional event journals, and Fire runtime log
  export.

## Current Limitations

- Fire mode requires Linux with KVM plus elevated privileges for TAP and
  iptables management.
- Fire rootfs images must be rebuilt when guest code, built-in skills, or guest
  dependencies change.
- Fire mode does not support resume across sessions; files persist only while a
  session VM is running.
- `shell` is powerful and high risk. Use Yolo mode only for trusted workflows,
  and review tool settings before running untrusted tasks.
- The project is intended for local experimentation, not production deployment
  or multi-user SaaS.

## Future Work

- Expand the built-in skill set, especially for coding workflows, research, and
  personal knowledge work.
- Add more integration-focused skills and broker policy examples for common
  APIs.
- Improve custom skill delivery in Fire mode without rebuilding the rootfs.
- Add more adapters and better observability for session replay and debugging.

## Documentation

- [Setup](./docs/setup.md): install, configure, and run Yolo or Fire mode.
- [Architecture](./docs/architecture.md): runtime model, agent loop, tools,
  skills, sessions, and security boundaries.
- [Configuration](./docs/configuration.md): LLMs, web search, secrets,
  integrations, Telegram, and adapters.
- [Fire Mode](./docs/fire-mode.md): Firecracker-specific behavior,
  troubleshooting, host services, credential isolation, and cleanup.

## Quick Start

For a local trusted run:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
mkdir -p ~/.strangeclaw
cp config.example.yaml ~/.strangeclaw/config.yaml
.venv/bin/python -m main
```

Set `mode: yolo` and your `llm` settings in
`~/.strangeclaw/config.yaml`. See [Setup](./docs/setup.md) for the full
walkthrough.

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).
