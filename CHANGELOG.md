# Changelog

Pre-1.0 semantic versioning: minor versions may add features.

## [0.2.0] - 2026-06-26

### Added
- Sequential in-VM subagents: the agent can delegate a subtask to a child agent
  in the same sandbox that returns one bounded result. Disabled by default;
  gated by both `tools.spawn_subagent` and `subagents.enabled`.
- `SubagentRunner` runs children synchronously in the parent's thread, with
  per-call iteration/timeout caps and a per-task fan-out cap.
- Child output is isolated under `subagents/<child_id>/` with traversal/symlink
  rejection and a size cap; subagent journaling supports `none`/`summary`/`full`.

### Changed
- Children are non-interactive and may use only a subset of the parent's tools;
  `spawn_subagent` recursion is disabled.

## [0.1.0] - 2026

Initial release: Yolo and Fire modes, request broker, host-side LLM proxy, CLI
and Telegram adapters, and the Agent Skills format.
