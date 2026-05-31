# strangeClaw

strangeClaw is a small, self-hosted autonomous agent experiment. I started it to
understand how agent systems work when you keep the moving parts visible:
planning, tool execution, sandboxing, credentials, and user interaction.

The agent accepts a task, plans, executes tools, observes results, and repeats
until the model chooses to finish, ask for clarification, or replan. The main
question behind the project is whether a useful personal agent can stay simple
while running untrusted work inside a stronger sandbox than a container.

It is not production-ready. It is built around four constraints:

- Simplicity: a small Python codebase with explicit module boundaries.
- Security: Firecracker mode runs the agent in a microVM; credentials stay on
  the host where possible.
- Maintainability: tools are fixed framework capabilities, skills are plain
  workflow documents.
- Expandability: adapters, skills, and host services can be added without
  changing the core loop.

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
