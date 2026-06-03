# Setup

This guide covers installing dependencies, configuring the agent, and running
strangeClaw in Yolo or Fire mode.

## Modes

- `yolo`: direct host execution. Fast, convenient, and not isolated.
- `fire`: Firecracker microVM isolation. Slower to set up, safer for untrusted
  tasks.

## Yolo Mode

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

4. Run strangeClaw:

   ```bash
   .venv/bin/python -m main
   ```

5. Enter a task, review or approve the plan, and wait for the final result.

Resume a saved Yolo session:

```bash
.venv/bin/python -m main --resume <session_id>
```

## Fire Mode

Fire mode needs host prerequisites, Firecracker kernel/rootfs assets, and a
guest rootfs containing the current strangeClaw code. Run these commands from
the repository root.

See [Fire Mode](./fire-mode.md) for architecture details and troubleshooting.

1. Install the Python environment as in the Yolo setup:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e ".[dev]"
   ```

2. Install or check host prerequisites. On Ubuntu, Linux Mint, Fedora, and other
   `apt-get`/`dnf` hosts:

   ```bash
   bash scripts/setup-fire.sh
   ```

   This checks `/dev/kvm`, grants KVM access where possible, installs the pinned
   Firecracker binary, checks `tun`, checks a container runtime, and prints a
   status report.

3. Enable IPv4 forwarding if the prerequisite report says it is the remaining
   blocker:

   ```bash
   bash scripts/setup-fire.sh --enable-ip-forwarding-now
   ```

   To persist it:

   ```bash
   bash scripts/setup-fire.sh --persist-ip-forwarding
   ```

4. Download Firecracker kernel/rootfs assets and run kernel vsock checks:

   ```bash
   bash scripts/fetch-fire-assets.sh
   ```

   If the script reports missing tools, install the packages that provide
   `unsquashfs`, `mkfs.ext4`, `debugfs`, or `e2fsck` and rerun it. If the
   downloaded kernel does not provide built-in vsock support, build a fallback
   kernel:

   ```bash
   bash scripts/build-fire-kernel.sh
   ```

   Download only, without smoke checks:

   ```bash
   bash scripts/fetch-fire-assets.sh --download-only
   ```

5. Build the strangeClaw guest rootfs:

   ```bash
   bash scripts/build-fire-rootfs.sh
   ```

   Re-run this whenever Fire mode needs changed guest code, built-in skills,
   guest dependencies, or `firecracker/rootfs/entrypoint.sh`.

6. Create or update local config:

   ```bash
   mkdir -p ~/.strangeclaw
   cp config.example.yaml ~/.strangeclaw/config.yaml
   ```

   Set Fire mode and confirm the Firecracker paths:

   ```yaml
   mode: fire
   adapters:
     enabled: [cli]

   firecracker:
     binary: /usr/local/bin/firecracker
     kernel: ./firecracker/kernel/vmlinux
     rootfs: ./firecracker/rootfs/agent.ext4
   ```

7. Add host-side secrets for external API/search access:

   ```bash
   cp secrets.example.yaml ~/.strangeclaw/secrets.yaml
   chmod 600 ~/.strangeclaw/secrets.yaml
   ```

   LLM credentials stay in `config.yaml`; external API/search credentials live
   in `~/.strangeclaw/secrets.yaml`.

8. Run a verification check:

   ```bash
   sudo --preserve-env=HOME,ANTHROPIC_API_KEY .venv/bin/python scripts/verify_fire.py --check network
   ```

   Full boot and agent lifecycle check:

   ```bash
   sudo --preserve-env=HOME,ANTHROPIC_API_KEY .venv/bin/python scripts/verify_fire.py --check lifecycle
   ```

   Preserve whichever environment variables your config references.

9. Run strangeClaw in Fire mode:

   ```bash
   sudo --preserve-env=HOME,ANTHROPIC_API_KEY .venv/bin/python -m main
   ```

   For local models only `HOME` may be needed:

   ```bash
   sudo --preserve-env=HOME .venv/bin/python -m main
   ```

Fire mode needs elevated privileges for TAP device and iptables management.

## Fire Prerequisite Checks

```bash
# Checks only, no host changes
bash scripts/setup-fire.sh --check-only

# Direct prerequisite checker
bash scripts/fire-check.sh
```

## Fire Cleanup

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

The cleanup script is conservative: it targets strangeClaw-owned Firecracker
processes, TAP names, iptables rules, and stale runtime paths.

## Arch Linux Notes

The setup script deliberately does not install packages with `pacman`. Do
package-manager work manually on Arch-family systems to avoid partial upgrades.
The Fire setup has been tested on CachyOS and Fedora 42.

1. Fully upgrade first:

   ```bash
   sudo pacman -Syu
   ```

2. Reboot if the upgrade installed a new kernel, `systemd`, `glibc`, `kmod`, or
   low-level networking packages:

   ```bash
   sudo reboot
   ```

3. Install host prerequisites:

   ```bash
   sudo pacman -S --needed acl curl iproute2 iptables ca-certificates kmod tar docker e2fsprogs squashfs-tools
   ```

   If prompted for an iptables provider, choose the nft-compatible `iptables`
   package.

4. Verify required commands:

   ```bash
   command -v setfacl curl ip iptables modprobe tar sha256sum unsquashfs mkfs.ext4 debugfs e2fsck
   iptables -V
   ```

5. Verify KVM:

   ```bash
   ls -l /dev/kvm
   test -d /sys/module/kvm && echo "kvm module loaded"
   ```

6. Run the setup script for Firecracker install/checks:

   ```bash
   bash scripts/setup-fire.sh
   ```
