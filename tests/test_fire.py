"""Tests for Firecracker sandbox helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from sandbox.fire import (
    DEFAULT_BOOT_ARGS,
    FirecrackerAPIClient,
    FirecrackerAPIError,
    FirecrackerConfig,
    FirecrackerConfigError,
    FirePrebootConfig,
    configure_microvm_preboot,
    load_firecracker_config,
)


def test_load_firecracker_config_validates_and_resolves(tmp_path: Path) -> None:
    binary = tmp_path / "firecracker"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    kernel = tmp_path / "vmlinux"
    kernel.write_text("kernel", encoding="utf-8")
    rootfs = tmp_path / "agent.ext4"
    rootfs.write_text("rootfs", encoding="utf-8")

    config = {
        "firecracker": {
            "binary": str(binary),
            "kernel": str(kernel),
            "rootfs": str(rootfs),
            "vcpu": 1,
            "mem_mb": 512,
            "host_iface": None,
            "boot_timeout": 30,
        }
    }
    loaded = load_firecracker_config(config)

    assert loaded.binary == binary.resolve()
    assert loaded.kernel == kernel.resolve()
    assert loaded.rootfs == rootfs.resolve()
    assert loaded.vcpu == 1
    assert loaded.mem_mb == 512
    assert loaded.host_iface is None
    assert loaded.boot_timeout == 30


def test_load_firecracker_config_rejects_missing_kernel(tmp_path: Path) -> None:
    binary = tmp_path / "firecracker"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    rootfs = tmp_path / "agent.ext4"
    rootfs.write_text("rootfs", encoding="utf-8")

    config = {
        "firecracker": {
            "binary": str(binary),
            "kernel": str(tmp_path / "missing-vmlinux"),
            "rootfs": str(rootfs),
            "vcpu": 1,
            "mem_mb": 512,
            "host_iface": "eth0",
            "boot_timeout": 30,
        }
    }

    with pytest.raises(FirecrackerConfigError, match="firecracker\\.kernel file not found"):
        load_firecracker_config(config)


def test_load_firecracker_config_rejects_non_executable_binary(tmp_path: Path) -> None:
    binary = tmp_path / "firecracker"
    binary.write_text("not executable", encoding="utf-8")
    kernel = tmp_path / "vmlinux"
    kernel.write_text("kernel", encoding="utf-8")
    rootfs = tmp_path / "agent.ext4"
    rootfs.write_text("rootfs", encoding="utf-8")

    config = {
        "firecracker": {
            "binary": str(binary),
            "kernel": str(kernel),
            "rootfs": str(rootfs),
            "vcpu": 1,
            "mem_mb": 512,
            "host_iface": "eth0",
            "boot_timeout": 30,
        }
    }

    with pytest.raises(FirecrackerConfigError, match="binary is not executable"):
        load_firecracker_config(config)


def test_configure_microvm_preboot_calls_put_in_required_sequence(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    rootfs_copy = tmp_path / "rootfs-copy.ext4"
    rootfs_copy.write_text("copy", encoding="utf-8")
    preboot = FirePrebootConfig(
        rootfs_path=rootfs_copy,
        tap_name="fc-testtap",
        guest_mac="06:00:AC:10:00:02",
        guest_cid=52,
        vsock_uds_path=tmp_path / "fire.vsock",
        log_path=tmp_path / "firecracker.log",
        network={
            "ip": "172.16.0.2",
            "gateway": "172.16.0.1",
            "netmask": "255.255.255.252",
            "dns": ["8.8.8.8", "1.1.1.1"],
        },
        llm={
            "model": "anthropic/claude-sonnet-4-20250514",
            "api_key": "sk-test",
            "max_tokens": 4096,
            "temperature": 0.2,
        },
    )

    calls: list[tuple[str, dict[str, Any]]] = []

    class RecordingClient:
        def put(self, path: str, payload: Mapping[str, Any]) -> None:
            calls.append((path, dict(payload)))

    configure_microvm_preboot(
        api_client=RecordingClient(),
        config=config,
        preboot=preboot,
    )

    assert [path for path, _ in calls] == [
        "/boot-source",
        "/drives/rootfs",
        "/machine-config",
        "/network-interfaces/eth0",
        "/vsock",
        "/logger",
        "/mmds/config",
        "/mmds",
        "/actions",
    ]
    assert calls[0][1]["boot_args"] == DEFAULT_BOOT_ARGS
    assert calls[3][1]["guest_mac"] == "06:00:AC:10:00:02"
    assert calls[5][1]["log_path"] == str(preboot.log_path)
    assert calls[7][1] == {"network": preboot.network, "llm": preboot.llm}
    assert calls[8][1] == {"action_type": "InstanceStart"}


def test_configure_microvm_preboot_rejects_invalid_guest_mac(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    rootfs_copy = tmp_path / "rootfs-copy.ext4"
    rootfs_copy.write_text("copy", encoding="utf-8")
    preboot = FirePrebootConfig(
        rootfs_path=rootfs_copy,
        tap_name="fc-testtap",
        guest_mac="bad-mac",
        guest_cid=52,
        vsock_uds_path=tmp_path / "fire.vsock",
        log_path=tmp_path / "firecracker.log",
        network={
            "ip": "172.16.0.2",
            "gateway": "172.16.0.1",
            "netmask": "255.255.255.252",
            "dns": ["8.8.8.8"],
        },
        llm={
            "model": "openai/gpt-4.1-mini",
            "api_key": "sk-test",
            "max_tokens": 1024,
            "temperature": 0.2,
        },
    )

    with pytest.raises(FirecrackerConfigError, match="Invalid guest MAC format"):
        configure_microvm_preboot(
            api_client=_NoopClient(),
            config=config,
            preboot=preboot,
        )


def test_firecracker_api_client_put_raises_on_non_2xx() -> None:
    response = _FakeResponse(status=400, body='{"fault_message":"bad request"}')
    connection = _FakeConnection(response=response)

    def factory(api_socket_path: str, timeout_seconds: float) -> _FakeConnection:
        assert api_socket_path == "/tmp/fire.sock"
        assert timeout_seconds == 3.0
        return connection

    client = FirecrackerAPIClient(
        api_socket_path="/tmp/fire.sock",
        timeout_seconds=3.0,
        connection_factory=factory,
    )

    with pytest.raises(FirecrackerAPIError, match="HTTP 400"):
        client.put("/machine-config", {"vcpu_count": 1, "mem_size_mib": 512})
    assert connection.closed is True
    assert connection.request_calls[0]["method"] == "PUT"
    assert connection.request_calls[0]["url"] == "/machine-config"


def _build_firecracker_config(tmp_path: Path) -> FirecrackerConfig:
    binary = tmp_path / "firecracker"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    kernel = tmp_path / "vmlinux"
    kernel.write_text("kernel", encoding="utf-8")
    rootfs = tmp_path / "agent.ext4"
    rootfs.write_text("rootfs", encoding="utf-8")
    return FirecrackerConfig(
        binary=binary,
        kernel=kernel,
        rootfs=rootfs,
        vcpu=1,
        mem_mb=512,
        host_iface="eth0",
        boot_timeout=30,
    )


class _NoopClient:
    def put(self, path: str, payload: Mapping[str, Any]) -> None:
        del path
        del payload


@dataclass
class _FakeResponse:
    status: int
    body: str

    def read(self) -> bytes:
        return self.body.encode("utf-8")


class _FakeConnection:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.request_calls: list[dict[str, Any]] = []
        self.closed = False

    def request(
        self,
        method: str,
        url: str,
        body: str,
        headers: Mapping[str, str],
    ) -> None:
        self.request_calls.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "headers": headers,
            }
        )

    def getresponse(self) -> _FakeResponse:
        return self._response

    def close(self) -> None:
        self.closed = True
