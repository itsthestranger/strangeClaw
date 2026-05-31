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

Fire mode needs host prerequisites, Firecracker kernel/rootfs assets, and a
guest rootfs containing the current strangeClaw agent code. Run these steps from
the repository root.

1. Install the Python environment as in the Yolo quickstart:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e ".[dev]"
   ```
2. Install/check host prerequisites. On Ubuntu, Linux Mint, Fedora, and other
   `apt-get`/`dnf` hosts:
   ```bash
   bash scripts/setup-fire.sh
   ```
   This installs basic host packages where supported, grants `/dev/kvm` access,
   installs the pinned Firecracker binary to `/usr/local/bin/firecracker`, checks
   `tun`, checks the container runtime, and prints a post-setup report.

   The setup script deliberately does not install packages with `pacman`. Arch
   Linux and CachyOS are rolling-release systems; running a package database
   refresh without a full system upgrade can create a partial-upgrade state. Use
   the manual Arch path below instead.
3. Enable IPv4 forwarding if the prerequisite report says `ip_forward` is the
   remaining blocker. For the current boot only:
   ```bash
   bash scripts/setup-fire.sh --enable-ip-forwarding-now
   ```
   To persist it in `/etc/sysctl.d`:
   ```bash
   bash scripts/setup-fire.sh --persist-ip-forwarding
   ```
4. Download the pinned Firecracker kernel/rootfs artifacts and run the kernel
   vsock checks:
   ```bash
   bash scripts/fetch-fire-assets.sh
   ```
   This reads `firecracker/VERSION`, downloads the matching CI kernel and Ubuntu
   rootfs assets into `firecracker/kernel/` and `firecracker/rootfs/`, generates
   a temporary `firecracker/rootfs/agent.ext4`, and verifies that the kernel can
   expose `/dev/vsock`.

   If this script reports missing commands such as `unsquashfs`, `mkfs.ext4`,
   `debugfs`, or `e2fsck`, install the distro packages that provide them
   (`squashfs-tools` and `e2fsprogs` on most Linux distributions) and rerun it.
   If the downloaded kernel does not provide built-in vsock support, build a
   fallback kernel:
   ```bash
   bash scripts/build-fire-kernel.sh
   ```

   If you only want to download assets and skip the runtime smoke checks:
   ```bash
   bash scripts/fetch-fire-assets.sh --download-only
   ```
5. Build the actual strangeClaw guest rootfs:
   ```bash
   bash scripts/build-fire-rootfs.sh
   ```
   This uses Docker or Podman to build `firecracker/rootfs/agent.ext4` from
   `firecracker/rootfs/Dockerfile`, exports the container filesystem, creates an
   ext4 image with `mkfs.ext4 -d`, and verifies critical guest files with
   `debugfs`.

   If `setup-fire.sh` warned that Docker/Podman is missing, install one of them
   before this step. The script auto-detects either runtime; you can force one
   with `--runtime docker` or `--runtime podman`.

   Re-run this script whenever you need Fire mode to pick up changed agent code,
   built-in skills, guest dependencies, or `firecracker/rootfs/entrypoint.sh`.
6. Create/update local config:
   ```bash
   mkdir -p ~/.strangeclaw
   cp config.example.yaml ~/.strangeclaw/config.yaml
   ```
   Set Fire mode and confirm the Firecracker paths match the generated files:
   ```yaml
   mode: fire
   adapters:
     enabled: [cli]

   firecracker:
     binary: /usr/local/bin/firecracker
     kernel: ./firecracker/kernel/vmlinux
     rootfs: ./firecracker/rootfs/agent.ext4
   ```
7. Add any host-side secrets. LLM credentials stay in `config.yaml`; external
   API/search credentials live in `~/.strangeclaw/secrets.yaml`:
   ```bash
   cp secrets.example.yaml ~/.strangeclaw/secrets.yaml
   chmod 600 ~/.strangeclaw/secrets.yaml
   ```
8. Run a verification check before starting normal tasks:
   ```bash
   sudo --preserve-env=HOME,ANTHROPIC_API_KEY .venv/bin/python scripts/verify_fire.py --check network
   ```
   `--check network` verifies TAP/firewall setup and cleanup without running an
   LLM task. For a full boot + agent lifecycle check, use:
   ```bash
   sudo --preserve-env=HOME,ANTHROPIC_API_KEY .venv/bin/python scripts/verify_fire.py --check lifecycle
   ```
   Preserve whichever environment variables your `~/.strangeclaw/config.yaml`
   references, not only `ANTHROPIC_API_KEY`.
9. Run strangeClaw in Fire mode:
   ```bash
   sudo --preserve-env=HOME,ANTHROPIC_API_KEY .venv/bin/python -m main
   ```
   For local models only `HOME` needs to be preserved:
   ```bash
   sudo --preserve-env=HOME .venv/bin/python -m main
   ```
   Fire mode needs elevated privileges for TAP device and iptables management.

Useful prerequisite-check variants:

```bash
# Checks only (no host changes)
bash scripts/setup-fire.sh --check-only

# Direct prerequisite checker
bash scripts/fire-check.sh
```

#### Fire Mode Cleanup

If the host process is terminated abnormally, Firecracker processes, TAP
devices, iptables rules, or `/tmp/strangeclaw-*` runtime directories can be left
behind. Inspect first:

```bash
sudo bash scripts/cleanup-fire.sh --dry-run
```

Then remove strangeClaw-owned Fire resources:

```bash
sudo bash scripts/cleanup-fire.sh
```

The cleanup script is best-effort and conservative: it only targets Firecracker
processes with a strangeClaw `--api-sock`, TAP names matching `fc` plus 12 hex
characters, matching per-session iptables rules, and verified stale Fire runtime
paths under `strangeclaw-*` temp directories.

#### Arch Linux / CachyOS Manual Fire Setup

Do the package-manager work yourself on Arch-family systems. This avoids
automated partial upgrades and keeps kernel/module changes explicit.

1. Fully upgrade first:
   ```bash
   sudo pacman -Syu
   ```
2. If the upgrade installed a new kernel, `systemd`, `glibc`, `kmod`, or low-level
   networking packages, reboot before continuing. This keeps `/dev/kvm`,
   `modprobe`, and loaded modules aligned with the running kernel:
   ```bash
   sudo reboot
   ```
3. Install the host prerequisites:
   ```bash
   sudo pacman -S --needed acl curl iproute2 iptables ca-certificates kmod tar docker e2fsprogs squashfs-tools
   ```
   If `pacman` asks about an iptables provider, choose the nft-compatible
   `iptables` package, not a legacy-only alternative.
4. Verify the commands strangeClaw needs:
   ```bash
   command -v setfacl curl ip iptables modprobe tar sha256sum unsquashfs mkfs.ext4 debugfs e2fsck
   iptables -V
   ```
   `iptables -V` should mention `nf_tables`. If it does not, fix the iptables
   backend before running Fire mode.
5. Verify KVM is available:
   ```bash
   ls -l /dev/kvm
   test -d /sys/module/kvm && echo "kvm module loaded"
   ```
   If `/dev/kvm` is missing, enable virtualization in firmware and make sure the
   appropriate KVM module for your CPU is loaded by the running kernel.
6. Run the strangeClaw setup script. On Arch-family systems it skips package
   installation, grants `/dev/kvm` access when possible, installs the pinned
   Firecracker binary from GitHub after checksum verification, optionally loads
   `tun`, and runs the post-setup checks:
   ```bash
   bash scripts/setup-fire.sh
   ```
7. If the only remaining failure is IPv4 forwarding, enable it for the current
   boot or persist it as described above.

## Fire Mode Behavior (Current)

- Fire mode is session-persistent: the first task in a session boots a microVM,
  and follow-up tasks in that same session reuse the same running VM.
- Files and installed tooling inside the guest persist across tasks in the same
  session.
- Idle Fire sessions are reaped by timeout (`firecracker.session_idle_timeout_seconds`,
  default `1800`; set `0` to disable reaping).
- Across sessions the VM filesystem is ephemeral; start a new session when you
  want a clean guest.
- `--resume` is intentionally rejected in Fire mode.

## Local LLMs In Fire Mode

Fire mode uses the host-side LLM proxy. The guest never receives the LLM API key
or LiteLLM configuration; it sends model calls over the existing host-services
channel, and the host process calls the configured provider.

For local/self-hosted models such as Ollama, LM Studio, or vLLM, configure
`llm.api_base` on the host. The guest calls the host LLM service over vsock, and
the host process talks to the local model.

Ollama quick-start:

```yaml
llm:
  model: ollama/llama3.1
  api_key: ""
  api_base: "http://127.0.0.1:11434"
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
- Host-side request broker for `web_search`, `web_fetch`, and `http_request`
  so external API credentials stay out of the sandbox.

## Architecture

High-level runtime shape:

```text
Host (main/coordinator/adapters)
  -> Sandbox (Yolo or Fire)
    -> Agent loop (plan -> inspect -> choose -> act -> observe -> repeat)
      -> Tools + skills + LLM calls inside sandbox
      -> HTTP/search/API calls via host-side broker
```

## Security Model

`yolo` mode has no isolation and is for trusted local workflows. `fire` mode
runs the agent in a Firecracker microVM; the Firecracker VMM boundary is the
primary host protection.

The Fire guest has NAT internet access. The request broker is not an internet
gatekeeper: it injects credentials and enforces host-defined policy for
`http_request`, `web_fetch`, and `web_search`. Public guest internet traffic can
still happen directly through enabled tools such as `shell`.

External API/search credentials stay in `~/.strangeclaw/secrets.yaml` on the
host. LLM provider credentials stay in the host process and are used by the
host-side LLM proxy. Fire MMDS contains guest runtime settings only, not
integration tokens or LLM provider configuration.

## Tools vs Skills

Tools are capabilities. They are built into strangeclaw, can be enabled or
disabled in `config.yaml`, and are the permission boundary:

- `shell`: run shell commands. High risk.
- `web_search`: search via the host broker. Low risk.
- `web_fetch`: fetch a public URL response via the host broker. Low risk.
- `http_request`: make structured HTTP/API calls via the host broker. Medium risk.

`web_fetch` performance note:
The broker now returns raw HTTP response data (`status_code`, `headers`, `body`,
`truncated`) without host-side content extraction. This reduces host-side
complexity and parser attack surface, but HTML-heavy pages can increase token
usage and response latency. For web-heavy workflows, prefer optional guest-side
parsing steps (for example via `shell`) before summarization.

Skills are instructions and workflow context. A skill is a directory under
`skills/` with a `SKILL.md` file using YAML frontmatter, plus optional
`references/`, `scripts/`, and `assets/` files. Installing a skill grants no new
permissions. A skill can only cause effects through enabled tools.

Adding a skill means dropping a directory into `skills/`. During planning, the
agent sees only each skill's name and description. During execution, referenced
skills are activated and their docs are added to context. Bundled files are read
on demand through `agent_read_skill_file`; bundled scripts are only executable if
the `shell` tool is enabled.

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

## Tool Configuration

Enable or disable tools in `config.yaml`:

```yaml
tools:
  shell: true
  web_search: true
  web_fetch: true
  http_request: true
```

Disabled tools are removed from the model's action surface. If a skill depends
on a disabled tool, that skill remains readable documentation but cannot perform
the blocked action.

## Web Search Setup

`web_search` uses two config locations:

- `config.yaml` selects the endpoint and response format.
- `~/.strangeclaw/secrets.yaml` holds the optional `_web_search` credential.

Brave quick-start in `config.yaml`:

```yaml
web_search:
  endpoint: "https://api.search.brave.com/res/v1/web/search"
  format: "brave"
  max_results: 10
```

Add the Brave key to `~/.strangeclaw/secrets.yaml`:

```yaml
credentials:
  _web_search:
    auth_type: header
    header_name: X-Subscription-Token
    token: "..."
    allowed_hosts: ["api.search.brave.com"]
    allowed_methods: [GET]
    allowed_paths: ["/*"]
```

Behavior:
- `format: brave` sends `q=<query>` and uses `X-Subscription-Token`.
- `format: searxng` sends `q=<query>&format=json`.
- Results are normalized into `{title, url, snippet}` for the model.

SearXNG local setup note:
- I run SearXNG from Docker using the official container installation guide:
  https://docs.searxng.org/admin/installation-docker.html
- The standard Compose setup exposes SearXNG on `http://localhost:8080`.
  Configure strangeclaw like this:
  ```yaml
  web_search:
    endpoint: "http://localhost:8080/search"
    format: "searxng"
    max_results: 10
  ```
- For local SearXNG without authentication, keep the `_web_search` entry in
  `secrets.yaml` so the broker knows the allowed host and policy. The current
  secrets loader requires a non-empty `token`, so use a non-secret placeholder
  value if your SearXNG instance does not need one (for example
  `"unused-local-searxng-token"`).
- In SearXNG's `settings.yml`, allow JSON output or requests with
  `format=json` will be rejected. The relevant setting is:
  ```yaml
  search:
    formats:
      - html
      - json
  ```

## External API Integrations

All HTTP egress for `http_request`, `web_fetch`, and `web_search` goes through
the host-side request broker.

For authenticated APIs, the model calls `http_request` with an integration name:

```json
{
  "integration": "github",
  "method": "GET",
  "url": "https://api.github.com/user/repos",
  "headers": {},
  "body": null
}
```

The broker executes the call only if `~/.strangeclaw/secrets.yaml` contains a
matching `credentials.github` entry. It validates method, host, path, protected
headers, response size, and rate limits before injecting the credential. The
model never sees token values.
As a deliberate security decision, integration auth supports only `bearer` and
`header` modes. Query-string credential injection (`auth_type: query`) is not
supported.

A skill without a matching credentials entry is inert at execution time. For
example, the `notion` skill can teach the model Notion API shapes, but Notion
calls are denied until `credentials.notion` exists.

## Setting Up `secrets.yaml`

Start from the example file:

```bash
cp secrets.example.yaml ~/.strangeclaw/secrets.yaml
chmod 600 ~/.strangeclaw/secrets.yaml
```

Then fill in only the credentials you want to authorize. Common entries are:

- `credentials._web_search` for Brave or SearXNG-backed `web_search`.
- `credentials.notion` for the `notion` skill.
- `credentials.github` for the `github` skill.

Policy fields in `secrets.yaml` are the authorization boundary. Keep
`allowed_hosts`, `allowed_methods`, and `allowed_paths` as narrow as practical.

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
