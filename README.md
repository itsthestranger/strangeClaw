# strangeclaw

strangeclaw is a minimal, self-hosted autonomous AI agent designed around four
drivers: simplicity, security, maintainability, and expandability.

It accepts a task, builds a plan, executes tools, observes results, and loops
until done. The same core agent runs in two modes:

- `yolo`: direct host execution for trusted local workflows
- `fire`: Firecracker microVM isolation for stronger host security

Architecture and implementation details are defined in
`strangeclaw_spec.md` and tracked in `strangeclaw_backlog.md`.
