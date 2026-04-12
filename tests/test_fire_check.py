"""Tests for Fire mode host prerequisite checks."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

import sandbox.fire_check as fire_check


def test_run_fire_checks_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fire_check,
        "_read_os_release",
        lambda: {"ID": "ubuntu", "VERSION_ID": "24.04", "PRETTY_NAME": "Ubuntu 24.04"},
    )
    monkeypatch.setattr(fire_check, "_read_pinned_firecracker_version", lambda: "v1.8.2")
    monkeypatch.setattr(
        fire_check,
        "_command_exists",
        lambda name: name in {"setfacl", "curl", "ip", "iptables", "docker"},
    )
    monkeypatch.setattr(
        fire_check,
        "_command_output",
        lambda command: (0, "Firecracker v1.8.2"),
    )

    present = {
        "/dev/kvm",
        "/sys/module/kvm",
        "/proc/sys/net/ipv4/ip_forward",
        "/sys/module/tun",
        "/dev/net/tun",
        "/usr/local/bin/firecracker",
    }

    def fake_exists(path_obj: Path) -> bool:
        return str(path_obj) in present

    def fake_read_text(path_obj: Path, encoding: str = "utf-8") -> str:
        del encoding
        if str(path_obj) == "/proc/sys/net/ipv4/ip_forward":
            return "1\n"
        return ""

    def fake_access(path_obj: Path, mode: int) -> bool:
        del mode
        return str(path_obj) in {"/dev/kvm", "/usr/local/bin/firecracker"}

    monkeypatch.setattr(fire_check.Path, "exists", fake_exists)
    monkeypatch.setattr(fire_check.Path, "read_text", fake_read_text)
    monkeypatch.setattr(fire_check.os, "access", fake_access)

    results = fire_check.run_fire_checks()
    assert results
    assert all(result.status == "PASS" for result in results)


def test_run_fire_checks_collects_failures_and_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fire_check,
        "_read_os_release",
        lambda: {"ID": "debian", "VERSION_ID": "12", "PRETTY_NAME": "Debian 12"},
    )
    monkeypatch.setattr(fire_check, "_command_exists", lambda name: False)
    monkeypatch.setattr(fire_check, "_read_pinned_firecracker_version", lambda: "v1.8.2")
    monkeypatch.setattr(fire_check.os, "access", lambda path_obj, mode: False)
    monkeypatch.setattr(fire_check.Path, "exists", lambda path_obj: False)

    results = fire_check.run_fire_checks()
    statuses = {result.name: result.status for result in results}
    assert statuses["host_os"] == "WARN"
    assert statuses["kvm_device"] == "FAIL"
    assert statuses["setfacl"] == "FAIL"
    assert statuses["container_runtime"] == "WARN"


def test_print_fire_check_report_renders_summary() -> None:
    results = [
        fire_check.FireCheckResult(name="a", status="PASS", details="ok"),
        fire_check.FireCheckResult(name="b", status="WARN", details="warn"),
        fire_check.FireCheckResult(name="c", status="FAIL", details="fail"),
    ]
    stream = io.StringIO()

    fire_check.print_fire_check_report(results, stream=stream)
    text = stream.getvalue()

    assert "check" in text
    assert "status" in text
    assert "Summary: PASS=1 WARN=1 FAIL=1" in text


def test_run_fire_check_command_success_and_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = io.StringIO()
    monkeypatch.setattr(
        fire_check,
        "run_fire_checks",
        lambda: [fire_check.FireCheckResult(name="a", status="PASS", details="ok")],
    )
    assert fire_check.run_fire_check_command(stream=stream) is True

    stream = io.StringIO()
    monkeypatch.setattr(
        fire_check,
        "run_fire_checks",
        lambda: [fire_check.FireCheckResult(name="a", status="FAIL", details="nope")],
    )
    assert fire_check.run_fire_check_command(stream=stream) is False
