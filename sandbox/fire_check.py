"""Fire mode prerequisite checks and report rendering."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class FireCheckResult:
    """A single Fire-mode host prerequisite check result."""

    name: str
    status: str  # PASS | WARN | FAIL
    details: str


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_pinned_firecracker_version() -> str:
    version_path = _project_root() / "firecracker" / "VERSION"
    try:
        return version_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_os_release() -> Mapping[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def _command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def _command_output(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError:
        return (1, "")
    except subprocess.TimeoutExpired:
        return (1, "")

    output = (completed.stdout or completed.stderr or "").strip()
    return (completed.returncode, output)


def run_fire_checks() -> list[FireCheckResult]:
    """Run all host prerequisite checks for Fire mode."""
    results: list[FireCheckResult] = []

    os_release = _read_os_release()
    distro_id = os_release.get("ID", "").strip().lower()
    distro_version = os_release.get("VERSION_ID", "").strip()
    if os.name != "posix":
        results.append(FireCheckResult("host_os", "FAIL", "Linux host required."))
    elif distro_id != "ubuntu":
        pretty = os_release.get("PRETTY_NAME", "unknown distribution")
        results.append(
            FireCheckResult(
                "host_os",
                "WARN",
                f"Ubuntu 22.04/24.04 recommended; detected {pretty}.",
            )
        )
    elif distro_version not in {"22.04", "24.04"}:
        pretty = os_release.get("PRETTY_NAME", f"Ubuntu {distro_version or 'unknown'}")
        results.append(
            FireCheckResult(
                "host_os",
                "WARN",
                f"Ubuntu 22.04/24.04 recommended; detected {pretty}.",
            )
        )
    else:
        results.append(
            FireCheckResult(
                "host_os",
                "PASS",
                f"Supported host: Ubuntu {distro_version}.",
            )
        )

    kvm_path = Path("/dev/kvm")
    if not kvm_path.exists():
        results.append(FireCheckResult("kvm_device", "FAIL", "/dev/kvm not found."))
        results.append(
            FireCheckResult("kvm_access", "FAIL", "Cannot validate access without /dev/kvm.")
        )
    else:
        results.append(FireCheckResult("kvm_device", "PASS", "/dev/kvm is present."))
        has_rw = os.access(kvm_path, os.R_OK | os.W_OK)
        if has_rw:
            results.append(FireCheckResult("kvm_access", "PASS", "Read/write access is available."))
        else:
            results.append(
                FireCheckResult(
                    "kvm_access",
                    "FAIL",
                    "Read/write access missing. Grant access with setfacl or kvm group.",
                )
            )

    kvm_loaded = Path("/sys/module/kvm").exists()
    if kvm_loaded:
        results.append(FireCheckResult("kvm_module", "PASS", "KVM kernel module is loaded."))
    else:
        results.append(FireCheckResult("kvm_module", "FAIL", "KVM kernel module is not loaded."))

    for check_name, command_name in (
        ("setfacl", "setfacl"),
        ("curl", "curl"),
        ("iproute2", "ip"),
        ("iptables", "iptables"),
    ):
        if _command_exists(command_name):
            results.append(FireCheckResult(check_name, "PASS", f"Found `{command_name}`."))
        else:
            results.append(FireCheckResult(check_name, "FAIL", f"Missing `{command_name}`."))

    ip_forwarding_path = Path("/proc/sys/net/ipv4/ip_forward")
    if not ip_forwarding_path.exists():
        results.append(FireCheckResult("ip_forward", "FAIL", "Cannot read ip_forward sysctl."))
    else:
        enabled = ip_forwarding_path.read_text(encoding="utf-8").strip() == "1"
        if enabled:
            results.append(FireCheckResult("ip_forward", "PASS", "IPv4 forwarding is enabled."))
        else:
            results.append(FireCheckResult("ip_forward", "FAIL", "IPv4 forwarding is disabled."))

    tun_loaded = Path("/sys/module/tun").exists() and Path("/dev/net/tun").exists()
    if tun_loaded:
        results.append(FireCheckResult("tun", "PASS", "tun module is loaded."))
    else:
        results.append(FireCheckResult("tun", "FAIL", "tun module or /dev/net/tun is missing."))

    firecracker_binary = Path("/usr/local/bin/firecracker")
    if not firecracker_binary.exists():
        results.append(
            FireCheckResult(
                "firecracker_binary",
                "FAIL",
                "Missing /usr/local/bin/firecracker.",
            )
        )
    elif not os.access(firecracker_binary, os.X_OK):
        results.append(
            FireCheckResult(
                "firecracker_binary",
                "FAIL",
                "/usr/local/bin/firecracker is not executable.",
            )
        )
    else:
        results.append(
            FireCheckResult(
                "firecracker_binary",
                "PASS",
                "Found /usr/local/bin/firecracker.",
            )
        )
        pinned = _read_pinned_firecracker_version()
        rc, output = _command_output([str(firecracker_binary), "--version"])
        if rc != 0:
            results.append(
                FireCheckResult(
                    "firecracker_version",
                    "FAIL",
                    "Unable to execute `firecracker --version`.",
                )
            )
        elif pinned and pinned in output:
            results.append(
                FireCheckResult(
                    "firecracker_version",
                    "PASS",
                    f"Version matches pinned release {pinned}.",
                )
            )
        elif pinned:
            results.append(
                FireCheckResult(
                    "firecracker_version",
                    "FAIL",
                    f"Expected {pinned}, detected `{output}`.",
                )
            )
        else:
            results.append(
                FireCheckResult(
                    "firecracker_version",
                    "WARN",
                    "Pinned version file missing; version not validated.",
                )
            )

    has_docker = _command_exists("docker")
    has_podman = _command_exists("podman")
    if has_docker or has_podman:
        runtime = "docker" if has_docker else "podman"
        results.append(FireCheckResult("container_runtime", "PASS", f"Found `{runtime}`."))
    else:
        results.append(
            FireCheckResult(
                "container_runtime",
                "WARN",
                "Docker/Podman not found (needed for rootfs build).",
            )
        )

    return results


def print_fire_check_report(
    results: list[FireCheckResult],
    *,
    stream: TextIO,
) -> None:
    """Print a tabular report and summary."""
    name_width = max(len("check"), *(len(result.name) for result in results))
    status_width = len("status")

    stream.write(f"{'check':<{name_width}}  {'status':<{status_width}}  details\n")
    stream.write(f"{'-' * name_width}  {'-' * status_width}  {'-' * 40}\n")
    for result in results:
        line = (
            f"{result.name:<{name_width}}  {result.status:<{status_width}}  "
            f"{result.details}\n"
        )
        stream.write(line)

    passed = sum(1 for result in results if result.status == "PASS")
    warned = sum(1 for result in results if result.status == "WARN")
    failed = sum(1 for result in results if result.status == "FAIL")
    stream.write("\n")
    stream.write(f"Summary: PASS={passed} WARN={warned} FAIL={failed}\n")


def run_fire_check_command(*, stream: TextIO) -> bool:
    """Run checks, print report, and return success state."""
    results = run_fire_checks()
    print_fire_check_report(results, stream=stream)
    return all(result.status != "FAIL" for result in results)
