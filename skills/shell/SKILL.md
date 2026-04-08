---
name: shell
description: Execute shell commands in the sandbox and return stdout/stderr.
version: 1.0.0
requires_network: false
---

# shell

## When to Use
- Use this skill when you need to run local commands, inspect files, or execute scripts in the sandbox.

## Actions
### run
- Args:
  - `command` (string): Shell command to run with `bash -lc`.
- Returns:
  - `exit_code`, `stdout`, and `stderr`.
- Invoke:
  - `bash -lc "{command}"`

## Examples
- `{"skill":"shell","action":"run","args":{"command":"python3 --version"}}`
- `{"skill":"shell","action":"run","args":{"command":"ls -la /output"}}`

## Limitations
- Command execution is constrained by runtime timeout and sandbox permissions.
