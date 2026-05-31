"""Tests for the Fire cleanup recovery script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cleanup-fire.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _install_fake_commands(fake_bin: Path, log_path: Path) -> None:
    _write_executable(
        fake_bin / "sudo",
        """#!/usr/bin/env bash
printf 'sudo %s\\n' "$*" >> "$CLEANUP_LOG"
exec "$@"
""",
    )
    _write_executable(
        fake_bin / "pgrep",
        """#!/usr/bin/env bash
exit 1
""",
    )
    _write_executable(
        fake_bin / "ip",
        """#!/usr/bin/env bash
case "$*" in
  "-o link show")
    cat "$IP_LINK_OUTPUT"
    ;;
  "-4 -o addr show dev "*)
    cat "$IP_ADDR_OUTPUT"
    ;;
  "link del "*)
    printf 'ip %s\\n' "$*" >> "$CLEANUP_LOG"
    ;;
esac
""",
    )
    _write_executable(
        fake_bin / "iptables-save",
        """#!/usr/bin/env bash
if [[ "$*" == "-t filter" ]]; then
  cat "$FILTER_SAVE"
elif [[ "$*" == "-t nat" ]]; then
  cat "$NAT_SAVE"
fi
""",
    )
    _write_executable(
        fake_bin / "iptables",
        """#!/usr/bin/env bash
printf 'iptables %s\\n' "$*" >> "$CLEANUP_LOG"
""",
    )
    log_path.write_text("", encoding="utf-8")


def _script_env(fake_bin: Path, log_path: Path, fixture_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["CLEANUP_LOG"] = str(log_path)
    env["IP_LINK_OUTPUT"] = str(fixture_dir / "ip-link.txt")
    env["IP_ADDR_OUTPUT"] = str(fixture_dir / "ip-addr.txt")
    env["FILTER_SAVE"] = str(fixture_dir / "filter-save.txt")
    env["NAT_SAVE"] = str(fixture_dir / "nat-save.txt")
    return env


def test_cleanup_fire_removes_orphaned_iptables_rules_without_live_tap(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "commands.log"
    _install_fake_commands(fake_bin, log_path)

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "ip-link.txt").write_text("", encoding="utf-8")
    (fixture_dir / "ip-addr.txt").write_text("", encoding="utf-8")
    (fixture_dir / "filter-save.txt").write_text(
        "\n".join(
            [
                "*filter",
                "-A INPUT -i fc123456789abc -j DROP",
                "-A FORWARD -i fc123456789abc -o eth0 -j ACCEPT",
                "-A FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT",
                "COMMIT",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (fixture_dir / "nat-save.txt").write_text(
        "\n".join(
            [
                "*nat",
                "-A POSTROUTING -o eth0 -s 172.16.0.2/32 -j MASQUERADE",
                "-A POSTROUTING -o eth0 -s 10.0.0.2/32 -j MASQUERADE",
                "COMMIT",
                "",
            ]
        ),
        encoding="utf-8",
    )
    temp_root = tmp_path / "runtime"
    temp_root.mkdir()

    result = subprocess.run(
        ["bash", str(SCRIPT), "--temp-root", str(temp_root)],
        capture_output=True,
        env=_script_env(fake_bin, log_path, fixture_dir),
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log = log_path.read_text(encoding="utf-8")
    assert "iptables -D INPUT -i fc123456789abc -j DROP" in log
    assert "iptables -D FORWARD -i fc123456789abc -o eth0 -j ACCEPT" in log
    assert (
        "iptables -t nat -D POSTROUTING -o eth0 -s 172.16.0.2/32 -j MASQUERADE"
        in log
    )
    assert "iptables -D FORWARD -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT" in log
    assert "10.0.0.2" not in log


def test_cleanup_fire_only_removes_verified_runtime_paths(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_path = tmp_path / "commands.log"
    _install_fake_commands(fake_bin, log_path)

    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    (fixture_dir / "ip-link.txt").write_text("", encoding="utf-8")
    (fixture_dir / "ip-addr.txt").write_text("", encoding="utf-8")
    (fixture_dir / "filter-save.txt").write_text("", encoding="utf-8")
    (fixture_dir / "nat-save.txt").write_text("", encoding="utf-8")

    temp_root = tmp_path / "runtime"
    temp_root.mkdir()
    fire_runtime = temp_root / "strangeclaw-session-abc"
    fire_runtime.mkdir()
    (fire_runtime / "firecracker.socket").write_text("", encoding="utf-8")
    unrelated = temp_root / "strangeclaw-not-fire"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("keep", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPT), "--temp-root", str(temp_root)],
        capture_output=True,
        env=_script_env(fake_bin, log_path, fixture_dir),
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not fire_runtime.exists()
    assert unrelated.exists()
    assert (unrelated / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert "Skipping unrecognized strangeClaw temp path" in result.stdout
