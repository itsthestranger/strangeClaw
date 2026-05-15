"""Tests for Firecracker sandbox helpers."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from agent.protocol import encode_event
from sandbox.fire import (
    DEFAULT_BOOT_ARGS,
    FirecrackerAPIClient,
    FirecrackerAPIError,
    FirecrackerConfig,
    FirecrackerConfigError,
    FirePrebootConfig,
    FireSandbox,
    IptablesManager,
    TapDeviceManager,
    TapNetworkAllocation,
    VMBootError,
    _assert_no_secrets,
    _default_popen_factory,
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


def test_load_firecracker_config_parses_host_expose(tmp_path: Path) -> None:
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
            "host_iface": "eth0",
            "boot_timeout": 30,
            "host_expose": {"enabled": True, "ports": [11434, 11434]},
        }
    }
    loaded = load_firecracker_config(config)

    assert loaded.host_expose_enabled is True
    assert loaded.host_expose_ports == (11434,)
    assert loaded.log_export_enabled is False
    assert loaded.log_export_max_bytes == 32 * 1024


def test_load_firecracker_config_parses_log_export(tmp_path: Path) -> None:
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
            "host_iface": "eth0",
            "boot_timeout": 30,
            "log_export": {"enabled": True, "max_bytes": 4096},
        }
    }
    loaded = load_firecracker_config(config)

    assert loaded.log_export_enabled is True
    assert loaded.log_export_max_bytes == 4096


def test_default_popen_factory_detaches_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeProcess:
        stderr = None

        def poll(self) -> int | None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def terminate(self) -> None:
            return

        def kill(self) -> None:
            return

    def fake_popen(args: list[str], **kwargs: Any) -> _FakeProcess:
        captured["args"] = list(args)
        captured["kwargs"] = dict(kwargs)
        return _FakeProcess()

    monkeypatch.setattr("sandbox.fire.subprocess.Popen", fake_popen)

    process = _default_popen_factory(["firecracker", "--api-sock", "/tmp/fc.sock"])

    assert isinstance(process, _FakeProcess)
    assert captured["args"] == ["firecracker", "--api-sock", "/tmp/fc.sock"]
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["stdout"] is subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is subprocess.PIPE
    assert captured["kwargs"]["text"] is True


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
        agent_config={
            "tools": {
                "shell": True,
                "web_search": True,
                "web_fetch": True,
                "http_request": True,
            },
            "web_search": {
                "endpoint": "https://api.search.brave.com/res/v1/web/search",
                "format": "brave",
            },
            "web_fetch": {"max_chars": 20000},
            "skills": {"directory": "./skills", "max_file_chars": 20000},
            "approval_mode": "review",
            "max_iterations": 50,
            "context": {"token_budget": 4000, "summary_threshold": 10, "max_output_chars": 8000},
            "llm": {
                "model": "anthropic/claude-sonnet-4-20250514",
                "api_key": "sk-test",
                "max_tokens": 4096,
                "temperature": 0.2,
            },
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
    assert calls[7][1] == {"network": preboot.network, "config": preboot.agent_config}
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
        agent_config={
            "llm": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "sk-test",
                "max_tokens": 1024,
                "temperature": 0.2,
            }
        },
    )

    with pytest.raises(FirecrackerConfigError, match="Invalid guest MAC format"):
        configure_microvm_preboot(
            api_client=_NoopClient(),
            config=config,
            preboot=preboot,
        )


def test_assert_no_secrets_raises_on_token_leak() -> None:
    with pytest.raises(ValueError, match=r"credential leaked into MMDS: notion"):
        _assert_no_secrets(
            mmds_json='{"config":{"llm":{"api_key":"safe"},"web_search":{"endpoint":"x"}},"token":"notion-secret"}',
            credentials={
                "notion": {"token": "notion-secret"},
                "_web_search": {"token": "search-secret"},
            },
        )


def test_configure_microvm_preboot_rejects_credentials_in_mmds_payload(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    preboot = _build_preboot_config(
        tmp_path,
        agent_config={
            "llm": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "sk-host",
                "max_tokens": 1024,
                "temperature": 0.2,
            },
            "web_search": {
                "endpoint": "https://api.search.brave.com/res/v1/web/search",
                "format": "brave",
            },
            "skills": {"directory": "./skills", "max_file_chars": 20000},
            "leak": "ghp-secret-leak",
        },
    )

    with pytest.raises(ValueError, match=r"credential leaked into MMDS: github"):
        configure_microvm_preboot(
            api_client=_NoopClient(),
            config=config,
            preboot=preboot,
            credentials={"github": {"token": "ghp-secret-leak"}},
        )


def test_configure_microvm_preboot_rejects_invalid_tap_name(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    preboot = _build_preboot_config(
        tmp_path,
        tap_name="tap-name-too-long!",
    )

    with pytest.raises(FirecrackerConfigError, match="tap_name has invalid format"):
        configure_microvm_preboot(
            api_client=_NoopClient(),
            config=config,
            preboot=preboot,
        )


def test_configure_microvm_preboot_rejects_invalid_network_ip(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    preboot = _build_preboot_config(
        tmp_path,
        network={
            "ip": "",
            "gateway": "172.16.0.1",
            "netmask": "255.255.255.252",
            "dns": ["8.8.8.8"],
        },
    )

    with pytest.raises(FirecrackerConfigError, match=r"network\.ip must be a non-empty string"):
        configure_microvm_preboot(
            api_client=_NoopClient(),
            config=config,
            preboot=preboot,
        )


def test_configure_microvm_preboot_rejects_invalid_llm_max_tokens(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    preboot = _build_preboot_config(
        tmp_path,
        agent_config={
            "llm": {
                "model": "openai/gpt-4.1-mini",
                "api_key": "sk-test",
                "max_tokens": 0,
                "temperature": 0.2,
            }
        },
    )

    with pytest.raises(FirecrackerConfigError, match=r"llm\.max_tokens must be a positive integer"):
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


def test_tap_device_manager_allocates_sequential_subnets(tmp_path: Path) -> None:
    runner = _FakeCommandRunner(
        route_json='[{"dev":"eth0"}]',
        route_text="default via 192.168.1.1 dev eth0 proto dhcp",
    )
    manager = TapDeviceManager(
        host_iface=None,
        counter_path=tmp_path / "tap_counter",
        command_runner=runner,
    )

    first = manager.create(session_id="session-1")
    second = manager.create(session_id="session-2")

    assert first.tap_ip == "172.16.0.1"
    assert first.guest_ip == "172.16.0.2"
    assert second.tap_ip == "172.16.0.5"
    assert second.guest_ip == "172.16.0.6"
    assert first.host_iface == "eth0"
    assert second.host_iface == "eth0"
    assert len(first.tap_name) <= 15
    assert len(second.tap_name) <= 15
    assert first.tap_name != second.tap_name


def test_tap_device_manager_retries_on_ip_collision(tmp_path: Path) -> None:
    runner = _FakeCommandRunner(
        route_json='[{"dev":"eth9"}]',
        route_text="default via 10.0.0.1 dev eth9",
        addr_outputs=[
            "2: stale0    inet 172.16.0.1/30 brd 172.16.0.3 scope global stale0",
            "",
        ],
    )
    counter_path = tmp_path / "tap_counter"
    manager = TapDeviceManager(
        host_iface="eth9",
        counter_path=counter_path,
        command_runner=runner,
    )

    allocation = manager.create(session_id="session-collision")

    assert allocation.session_index == 1
    assert allocation.tap_ip == "172.16.0.5"
    assert allocation.guest_ip == "172.16.0.6"
    assert counter_path.read_text(encoding="utf-8").strip() == "2"


def test_tap_device_manager_detects_host_iface_from_text_route(tmp_path: Path) -> None:
    runner = _FakeCommandRunner(
        route_json="not-json",
        route_text="default via 192.168.0.1 dev wlan0 proto dhcp src 192.168.0.200",
    )
    manager = TapDeviceManager(
        host_iface=None,
        counter_path=tmp_path / "tap_counter",
        command_runner=runner,
    )

    allocation = manager.create(session_id="session-route-fallback")

    assert allocation.host_iface == "wlan0"


def test_tap_device_manager_rejects_invalid_session_id(tmp_path: Path) -> None:
    runner = _FakeCommandRunner(
        route_json='[{"dev":"eth0"}]',
        route_text="default via 192.168.1.1 dev eth0",
    )
    manager = TapDeviceManager(
        host_iface=None,
        counter_path=tmp_path / "tap_counter",
        command_runner=runner,
    )

    with pytest.raises(FirecrackerConfigError, match="Invalid session_id format"):
        manager.create(session_id="bad/session")


def test_tap_device_manager_destroy_never_raises(tmp_path: Path) -> None:
    runner = _FakeCommandRunner(
        route_json='[{"dev":"eth0"}]',
        route_text="default via 192.168.1.1 dev eth0",
        fail_link_del=True,
    )
    manager = TapDeviceManager(
        host_iface="eth0",
        counter_path=tmp_path / "tap_counter",
        command_runner=runner,
    )

    manager.destroy("fc123456789abc")


def test_iptables_manager_apply_adds_required_rules() -> None:
    runner = _ScriptedCommandRunner(
        returncodes={
            (
                "iptables",
                "-C",
                "FORWARD",
                "-m",
                "conntrack",
                "--ctstate",
                "RELATED,ESTABLISHED",
                "-j",
                "ACCEPT",
            ): [1]
        }
    )
    manager = IptablesManager(command_runner=runner)

    manager.apply(_build_allocation())

    assert (
        "iptables",
        "-A",
        "FORWARD",
        "-m",
        "conntrack",
        "--ctstate",
        "RELATED,ESTABLISHED",
        "-j",
        "ACCEPT",
    ) in runner.calls_as_tuples()
    assert (
        "iptables",
        "-t",
        "nat",
        "-A",
        "POSTROUTING",
        "-o",
        "eth0",
        "-s",
        "172.16.0.2",
        "-j",
        "MASQUERADE",
    ) in runner.calls_as_tuples()
    assert (
        "iptables",
        "-A",
        "FORWARD",
        "-i",
        "fc123456789abc",
        "-o",
        "eth0",
        "-j",
        "ACCEPT",
    ) in runner.calls_as_tuples()
    assert (
        "iptables",
        "-A",
        "INPUT",
        "-i",
        "fc123456789abc",
        "-j",
        "DROP",
    ) in runner.calls_as_tuples()


def test_iptables_manager_apply_skips_conntrack_when_existing() -> None:
    runner = _ScriptedCommandRunner()
    manager = IptablesManager(command_runner=runner)

    manager.apply(_build_allocation())

    assert (
        "iptables",
        "-A",
        "FORWARD",
        "-m",
        "conntrack",
        "--ctstate",
        "RELATED,ESTABLISHED",
        "-j",
        "ACCEPT",
    ) not in runner.calls_as_tuples()


def test_iptables_manager_apply_inserts_host_expose_accept_before_drop() -> None:
    runner = _ScriptedCommandRunner(
        returncodes={
            (
                "iptables",
                "-C",
                "FORWARD",
                "-m",
                "conntrack",
                "--ctstate",
                "RELATED,ESTABLISHED",
                "-j",
                "ACCEPT",
            ): [0]
        }
    )
    manager = IptablesManager(command_runner=runner)

    manager.apply(_build_allocation(), host_expose_ports=(11434,))

    calls = runner.calls_as_tuples()
    expose_call = (
        "iptables",
        "-I",
        "INPUT",
        "-i",
        "fc123456789abc",
        "-p",
        "tcp",
        "--dport",
        "11434",
        "-j",
        "ACCEPT",
    )
    drop_call = (
        "iptables",
        "-A",
        "INPUT",
        "-i",
        "fc123456789abc",
        "-j",
        "DROP",
    )
    assert expose_call in calls
    assert drop_call in calls
    assert calls.index(expose_call) < calls.index(drop_call)


def test_iptables_manager_cleanup_removes_conntrack_when_no_fire_taps() -> None:
    runner = _ScriptedCommandRunner(
        outputs={
            ("ip", "-o", "link", "show"): "",
        }
    )
    manager = IptablesManager(command_runner=runner)

    manager.cleanup(_build_allocation())

    assert (
        "iptables",
        "-D",
        "FORWARD",
        "-m",
        "conntrack",
        "--ctstate",
        "RELATED,ESTABLISHED",
        "-j",
        "ACCEPT",
    ) in runner.calls_as_tuples()


def test_iptables_manager_cleanup_keeps_conntrack_with_active_fire_taps() -> None:
    runner = _ScriptedCommandRunner(
        outputs={
            (
                "ip",
                "-o",
                "link",
                "show",
            ): "7: fc0123456789ab: <BROADCAST,MULTICAST> mtu 1500 state DOWN",
        }
    )
    manager = IptablesManager(command_runner=runner)

    manager.cleanup(_build_allocation())

    assert (
        "iptables",
        "-D",
        "FORWARD",
        "-m",
        "conntrack",
        "--ctstate",
        "RELATED,ESTABLISHED",
        "-j",
        "ACCEPT",
    ) not in runner.calls_as_tuples()


def test_iptables_manager_cleanup_removes_host_expose_accept_rules() -> None:
    runner = _ScriptedCommandRunner(
        outputs={
            ("ip", "-o", "link", "show"): "",
        }
    )
    manager = IptablesManager(command_runner=runner)

    manager.cleanup(_build_allocation(), host_expose_ports=(11434,))

    assert (
        "iptables",
        "-D",
        "INPUT",
        "-i",
        "fc123456789abc",
        "-p",
        "tcp",
        "--dport",
        "11434",
        "-j",
        "ACCEPT",
    ) in runner.calls_as_tuples()


def test_iptables_manager_cleanup_never_raises_on_command_errors() -> None:
    runner = _ScriptedCommandRunner(default_returncode=1)
    manager = IptablesManager(command_runner=runner)

    manager.cleanup(_build_allocation())


def test_iptables_manager_apply_raises_on_command_failure() -> None:
    runner = _ScriptedCommandRunner(
        returncodes={
            (
                "iptables",
                "-t",
                "nat",
                "-A",
                "POSTROUTING",
                "-o",
                "eth0",
                "-s",
                "172.16.0.2",
                "-j",
                "MASQUERADE",
            ): [1]
        }
    )
    manager = IptablesManager(command_runner=runner)

    with pytest.raises(
        FirecrackerConfigError,
        match="Command failed: iptables -t nat -A POSTROUTING",
    ):
        manager.apply(_build_allocation())


def test_fire_sandbox_run_sends_task_after_agent_ready(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    command_runner = _CopyingCommandRunner()
    tap_manager = _FakeTapManager(_build_allocation())
    iptables_manager = _FakeIptablesManager()
    cid_manager = _FakeCidManager(cid=52)
    api_factory = _FakeApiFactory()
    process_factory = _FakePopenFactory()
    socket_factory = _FakeSocketFactory(
        sockets=[
            _FakeConnectedSocket(
                recv_chunks=[b'OK 100\n{"type":"agent_ready"}\n'],
            )
        ]
    )

    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="openai/gpt-4.1-mini",
            api_key="sk-host",
            max_tokens=1024,
            temperature=0.2,
        ),
        tap_manager=tap_manager,
        iptables_manager=iptables_manager,
        cid_manager=cid_manager,
        api_client_factory=api_factory.make,
        command_runner=command_runner,
        popen_factory=process_factory,
        unix_socket_factory=socket_factory,
        temp_root=tmp_path,
        install_exit_handlers=False,
    )

    task = {
        "type": "task",
        "text": "hello",
        "session_id": "sess-1",
        "approval_mode": "review",
        "llm": {
            "model": "openai/gpt-4.1-mini",
            "api_key": "sk-inline",
            "max_tokens": 2048,
            "temperature": 0.1,
        },
    }
    sandbox.run(task)

    sent_payloads = socket_factory.sockets[0].sent
    assert sent_payloads[0] == b"CONNECT 5000\n"
    assert b'"type":"task"' in sent_payloads[1]
    assert b'"llm"' not in sent_payloads[1]
    assert tap_manager.create_calls == ["sess-1"]
    assert iptables_manager.applied is True
    assert iptables_manager.apply_ports == ()
    assert cid_manager.allocated == 52
    assert api_factory.paths() == [
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
    mmds_payload = api_factory.payload_for("/mmds")
    assert mmds_payload["config"]["web_search"] == {
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "format": "brave",
        "max_results": 10,
    }
    assert "api_key" not in mmds_payload["config"]["web_search"]
    assert "credentials" not in mmds_payload["config"]
    assert "integrations" not in mmds_payload["config"]

    sandbox.stop()
    assert iptables_manager.cleaned is True
    assert iptables_manager.cleanup_ports == ()
    assert tap_manager.destroyed == ["fc123456789abc"]
    assert cid_manager.released == [52]


def test_fire_sandbox_autonomous_event_stream_replan_read_and_done(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    command_runner = _CopyingCommandRunner()
    tap_manager = _FakeTapManager(_build_allocation())
    iptables_manager = _FakeIptablesManager()
    cid_manager = _FakeCidManager(cid=52)
    api_factory = _FakeApiFactory()
    process_factory = _FakePopenFactory()

    event_stream: list[dict[str, Any]] = [
        {"type": "agent_ready"},
        {
            "type": "message",
            "role": "plan",
            "content": {
                "goal": "integration task",
                "steps": ["inspect"],
                "referenced_skills": [],
            },
        },
        {
            "type": "message",
            "role": "plan",
            "content": {
                "goal": "integration task",
                "steps": ["read reference", "finish"],
                "referenced_skills": ["demo"],
            },
        },
        {
            "type": "action",
            "tool": "agent_read_skill_file",
            "args": {"skill": "demo", "path": "references/notes.md"},
            "result": {"exit_code": 0, "stdout": "skill-reference-content\n", "stderr": ""},
        },
        {
            "type": "action",
            "tool": "shell",
            "args": {"command": "printf integrated"},
            "result": {"exit_code": 0, "stdout": "integrated", "stderr": ""},
        },
        {
            "type": "done",
            "success": True,
            "reply": "autonomous done",
            "state": {
                "goal": "integration task",
                "plan": {"steps": ["read reference", "finish"], "referenced_skills": ["demo"]},
                "history": [],
                "summary": "",
            },
            "files": [
                {
                    "path": "artifact.txt",
                    "content_b64": "aW50ZWdyYXRlZA==",
                    "size_bytes": 10,
                }
            ],
        },
    ]
    stream_chunk = (
        b"OK 100\n" + "".join(encode_event(event) for event in event_stream).encode("utf-8")
    )
    socket_factory = _FakeSocketFactory(
        sockets=[_FakeConnectedSocket(recv_chunks=[stream_chunk])]
    )

    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="openai/gpt-4.1-mini",
            api_key="sk-host",
            max_tokens=1024,
            temperature=0.2,
        ),
        tap_manager=tap_manager,
        iptables_manager=iptables_manager,
        cid_manager=cid_manager,
        api_client_factory=api_factory.make,
        command_runner=command_runner,
        popen_factory=process_factory,
        unix_socket_factory=socket_factory,
        temp_root=tmp_path,
        install_exit_handlers=False,
    )

    task = {
        "type": "task",
        "text": "integration task",
        "session_id": "sess-c33g-fire",
        "approval_mode": "auto",
        "llm": {
            "model": "inline/model",
            "api_key": "sk-inline",
            "max_tokens": 2048,
            "temperature": 0.7,
        },
    }
    sandbox.run(task)
    events: list[dict[str, Any]] = []
    while True:
        event = sandbox.receive(timeout_seconds=1.0)
        assert event is not None
        events.append(event)
        if event.get("type") == "done":
            break

    # Capture preboot payload assertions before stop() issues CtrlAltDel API calls.
    mmds_payload = api_factory.payload_for("/mmds")

    try:
        # Fire mode must use MMDS-delivered config and must not forward task-inline llm.
        sent_payloads = socket_factory.sockets[0].sent
        assert sent_payloads[0] == b"CONNECT 5000\n"
        assert b'"type":"task"' in sent_payloads[1]
        assert b'"approval_mode":"auto"' in sent_payloads[1]
        assert b'"llm"' not in sent_payloads[1]
        assert mmds_payload["config"]["llm"]["api_key"] == "sk-host"
        assert mmds_payload["config"]["llm"]["model"] == "openai/gpt-4.1-mini"
        assert mmds_payload["config"]["skills"]["max_file_chars"] == 20000

        # Deterministic autonomous loop stream includes replan, skill-file read, and done.
        plan_events = [
            event
            for event in events
            if event.get("type") == "message" and event.get("role") == "plan"
        ]
        assert len(plan_events) == 2
        read_actions = [
            event
            for event in events
            if event.get("type") == "action"
            and event.get("tool") == "agent_read_skill_file"
        ]
        assert read_actions
        assert read_actions[0]["result"]["exit_code"] == 0

        done_event = events[-1]
        assert done_event["type"] == "done"
        assert done_event["success"] is True
        assert done_event["reply"] == "autonomous done"
        assert done_event["files"]
        assert done_event["files"][0]["path"] == "artifact.txt"
        assert not done_event["files"][0]["path"].startswith("/")
    finally:
        sandbox.stop()


def test_fire_sandbox_handles_broker_requests_without_adapter_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reflected_token = "fire-broker-secret-token"
    monkeypatch.setattr(
        "sandbox.fire.load_secrets",
        lambda: {
            "notion": {
                "name": "notion",
                "auth_type": "bearer",
                "token": reflected_token,
                "allowed_hosts": ["api.notion.com"],
                "allowed_methods": ["GET"],
                "allowed_paths": ["/v1/*"],
                "protected_headers": ["Authorization"],
                "default_headers": {},
                "max_response_bytes": 4096,
                "rate_limit": None,
            }
        },
    )
    monkeypatch.setattr(
        "sandbox.broker.RequestBroker._handle_list_integrations",
        lambda self, payload: {  # noqa: ARG005
            "success": True,
            "integrations": ["notion"],
            "echo": reflected_token,
        },
    )

    config = _build_firecracker_config(tmp_path)
    command_runner = _CopyingCommandRunner()
    tap_manager = _FakeTapManager(_build_allocation())
    iptables_manager = _FakeIptablesManager()
    cid_manager = _FakeCidManager(cid=52)
    api_factory = _FakeApiFactory()
    process_factory = _FakePopenFactory()

    event_stream: list[dict[str, Any]] = [
        {"type": "agent_ready"},
        {
            "type": "broker_request",
            "request_id": "req-1",
            "service": "broker",
            "payload": {"action": "list_integrations"},
        },
        {"type": "message", "role": "status", "content": "working"},
        {
            "type": "done",
            "success": True,
            "reply": "ok",
            "state": {"goal": "x"},
            "files": [],
        },
    ]
    stream_chunk = (
        b"OK 100\n" + "".join(encode_event(event) for event in event_stream).encode("utf-8")
    )
    socket_factory = _FakeSocketFactory(
        sockets=[_FakeConnectedSocket(recv_chunks=[stream_chunk])]
    )
    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="openai/gpt-4.1-mini",
            api_key="sk-host",
            max_tokens=1024,
            temperature=0.2,
        ),
        tap_manager=tap_manager,
        iptables_manager=iptables_manager,
        cid_manager=cid_manager,
        api_client_factory=api_factory.make,
        command_runner=command_runner,
        popen_factory=process_factory,
        unix_socket_factory=socket_factory,
        temp_root=tmp_path,
        install_exit_handlers=False,
    )

    try:
        sandbox.run(
            {
                "type": "task",
                "text": "broker roundtrip",
                "session_id": "sess-broker-fire",
                "approval_mode": "auto",
            }
        )
        event_one = sandbox.receive(timeout_seconds=1.0)
        event_two = sandbox.receive(timeout_seconds=1.0)
        assert event_one == {"type": "message", "role": "status", "content": "working"}
        assert event_two is not None
        assert event_two["type"] == "done"

        sent_payloads = socket_factory.sockets[0].sent
        decoded_sent = [
            payload.decode("utf-8", errors="replace")
            for payload in sent_payloads
            if payload.startswith(b"{")
        ]
        assert any('"type":"broker_response"' in payload for payload in decoded_sent)
        assert any('"request_id":"req-1"' in payload for payload in decoded_sent)
        broker_payloads = [
            payload for payload in decoded_sent if '"type":"broker_response"' in payload
        ]
        assert broker_payloads
        for payload in broker_payloads:
            assert reflected_token not in payload
            assert "[REDACTED]" in payload
    finally:
        sandbox.stop()


def test_fire_sandbox_connect_retries_with_new_socket(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    vsock_path = tmp_path / "fire.vsock"
    vsock_path.write_text("", encoding="utf-8")

    first = _FakeConnectedSocket(connect_error=OSError("connection refused"))
    second = _FakeConnectedSocket(recv_chunks=[b"OK 101\n"])
    socket_factory = _FakeSocketFactory([first, second])
    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="openai/gpt-4.1-mini",
            api_key="sk-host",
            max_tokens=1024,
            temperature=0.2,
        ),
        tap_manager=_FakeTapManager(_build_allocation()),
        iptables_manager=_FakeIptablesManager(),
        cid_manager=_FakeCidManager(cid=52),
        api_client_factory=_FakeApiFactory().make,
        command_runner=_CopyingCommandRunner(),
        popen_factory=_FakePopenFactory(),
        unix_socket_factory=socket_factory,
        temp_root=tmp_path,
        sleep=lambda _seconds: None,
        install_exit_handlers=False,
    )
    sandbox._vsock_uds_path = vsock_path  # noqa: SLF001

    conn = sandbox._connect_vsock_with_retry(timeout_seconds=1.0)  # noqa: SLF001
    assert conn is second
    assert len(socket_factory.sockets) == 2


def test_fire_sandbox_run_raises_boot_error_with_diagnostics(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    command_runner = _CopyingCommandRunner()
    tap_manager = _FakeTapManager(_build_allocation())
    iptables_manager = _FakeIptablesManager()
    cid_manager = _FakeCidManager(cid=52)
    api_factory = _FakeApiFactory()
    process_factory = _FakePopenFactory()
    socket_factory = _FakeSocketFactory([_FakeConnectedSocket(recv_chunks=[b"BAD\n"])])

    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="openai/gpt-4.1-mini",
            api_key="sk-host",
            max_tokens=1024,
            temperature=0.2,
        ),
        tap_manager=tap_manager,
        iptables_manager=iptables_manager,
        cid_manager=cid_manager,
        api_client_factory=api_factory.make,
        command_runner=command_runner,
        popen_factory=process_factory,
        unix_socket_factory=socket_factory,
        temp_root=tmp_path,
        install_exit_handlers=False,
    )

    with pytest.raises(VMBootError, match="firecracker log tail"):
        sandbox.run(
            {
                "type": "task",
                "text": "hello",
                "session_id": "sess-2",
                "approval_mode": "review",
            }
        )


def test_fire_sandbox_boot_diagnostics_redacts_sensitive_log_content(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    command_runner = _CopyingCommandRunner()
    tap_manager = _FakeTapManager(_build_allocation())
    iptables_manager = _FakeIptablesManager()
    cid_manager = _FakeCidManager(cid=52)
    process_factory = _SecretPopenFactory()
    socket_factory = _FakeSocketFactory([_FakeConnectedSocket(recv_chunks=[b"BAD\n"])])
    api_factory = _SecretApiFactory()

    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="openai/gpt-4.1-mini",
            api_key="sk-host",
            max_tokens=1024,
            temperature=0.2,
        ),
        tap_manager=tap_manager,
        iptables_manager=iptables_manager,
        cid_manager=cid_manager,
        api_client_factory=api_factory.make,
        command_runner=command_runner,
        popen_factory=process_factory,
        unix_socket_factory=socket_factory,
        temp_root=tmp_path,
        install_exit_handlers=False,
    )

    with pytest.raises(VMBootError) as excinfo:
        sandbox.run(
            {
                "type": "task",
                "text": "hello",
                "session_id": "sess-redact",
                "approval_mode": "review",
            }
        )

    err_text = str(excinfo.value)
    assert "sk-secret-stderr" not in err_text
    assert "sk-secret-log" not in err_text
    assert "[REDACTED]" in err_text


def test_fire_sandbox_rewrites_localhost_api_base_for_host_expose(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _build_firecracker_config(
        tmp_path,
        host_expose_enabled=True,
        host_expose_ports=(11434,),
    )
    command_runner = _CopyingCommandRunner()
    tap_manager = _FakeTapManager(_build_allocation())
    iptables_manager = _FakeIptablesManager()
    cid_manager = _FakeCidManager(cid=52)
    api_factory = _FakeApiFactory()
    process_factory = _FakePopenFactory()
    socket_factory = _FakeSocketFactory(
        sockets=[_FakeConnectedSocket(recv_chunks=[b'OK 100\n{"type":"agent_ready"}\n'])]
    )

    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="lm_studio/local-model",
            api_key="",
            max_tokens=1024,
            temperature=0.2,
            api_base="http://localhost:11434/v1",
        ),
        tap_manager=tap_manager,
        iptables_manager=iptables_manager,
        cid_manager=cid_manager,
        api_client_factory=api_factory.make,
        command_runner=command_runner,
        popen_factory=process_factory,
        unix_socket_factory=socket_factory,
        temp_root=tmp_path,
        install_exit_handlers=False,
    )

    with caplog.at_level("INFO"):
        sandbox.run(
            {
                "type": "task",
                "text": "hello",
                "session_id": "sess-expose",
                "approval_mode": "review",
            }
        )

    mmds_payload = api_factory.payload_for("/mmds")
    assert mmds_payload["config"]["llm"]["api_base"] == "http://172.16.0.1:11434/v1"
    assert (
        mmds_payload["config"]["llm"]["provider_settings"]["api_base"]
        == "http://172.16.0.1:11434/v1"
    )
    assert iptables_manager.apply_ports == (11434,)
    assert "host_expose enabled for ports [11434]" in caplog.text

    sandbox.stop()


def test_fire_sandbox_warns_when_localhost_api_base_without_host_expose(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _build_firecracker_config(tmp_path)
    command_runner = _CopyingCommandRunner()
    tap_manager = _FakeTapManager(_build_allocation())
    iptables_manager = _FakeIptablesManager()
    cid_manager = _FakeCidManager(cid=52)
    api_factory = _FakeApiFactory()
    process_factory = _FakePopenFactory()
    socket_factory = _FakeSocketFactory(
        sockets=[_FakeConnectedSocket(recv_chunks=[b'OK 100\n{"type":"agent_ready"}\n'])]
    )

    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="lm_studio/local-model",
            api_key="",
            max_tokens=1024,
            temperature=0.2,
            api_base="http://127.0.0.1:11434/v1",
        ),
        tap_manager=tap_manager,
        iptables_manager=iptables_manager,
        cid_manager=cid_manager,
        api_client_factory=api_factory.make,
        command_runner=command_runner,
        popen_factory=process_factory,
        unix_socket_factory=socket_factory,
        temp_root=tmp_path,
        install_exit_handlers=False,
    )

    with caplog.at_level("WARNING"):
        sandbox.run(
            {
                "type": "task",
                "text": "hello",
                "session_id": "sess-no-expose",
                "approval_mode": "review",
            }
        )

    mmds_payload = api_factory.payload_for("/mmds")
    assert mmds_payload["config"]["llm"]["api_base"] == "http://127.0.0.1:11434/v1"
    assert "host_expose is disabled" in caplog.text
    assert iptables_manager.apply_ports == ()

    sandbox.stop()


def test_fire_sandbox_stop_never_raises_and_is_idempotent(tmp_path: Path) -> None:
    config = _build_firecracker_config(tmp_path)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="openai/gpt-4.1-mini",
            api_key="sk-host",
            max_tokens=1024,
            temperature=0.2,
        ),
        tap_manager=_RaisingTapManager(_build_allocation()),
        iptables_manager=_RaisingIptablesManager(),
        cid_manager=_RaisingCidManager(cid=52),
        api_client_factory=_FakeApiFactory().make,
        command_runner=_CopyingCommandRunner(),
        popen_factory=_FakePopenFactory(),
        unix_socket_factory=_FakeSocketFactory([]),
        temp_root=tmp_path,
        install_exit_handlers=False,
    )
    sandbox._allocation = _build_allocation()  # noqa: SLF001
    sandbox._guest_cid = 52  # noqa: SLF001
    sandbox._session_temp_dir = runtime_dir  # noqa: SLF001
    sandbox._process = cast(Any, _FakeProcess())  # noqa: SLF001

    sandbox.stop()
    sandbox.stop()

    assert not runtime_dir.exists()
    assert sandbox._allocation is None  # noqa: SLF001
    assert sandbox._guest_cid is None  # noqa: SLF001
    assert sandbox._session_temp_dir is None  # noqa: SLF001


def test_fire_sandbox_stop_exports_redacted_log_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _build_firecracker_config(
        tmp_path,
        log_export_enabled=True,
        log_export_max_bytes=128,
    )
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    log_path = runtime_dir / "firecracker.log"
    log_path.write_text(
        "\n".join(
            [
                "line-before-tail",
                "token=abc123",
                "Authorization: Bearer super-secret",
                "api_key=sk-test-secret-123456",
            ]
        ),
        encoding="utf-8",
    )

    sandbox = FireSandbox(
        firecracker_config=config,
        agent_config=_agent_config(
            model="openai/gpt-4.1-mini",
            api_key="sk-host",
            max_tokens=1024,
            temperature=0.2,
        ),
        tap_manager=_RaisingTapManager(_build_allocation()),
        iptables_manager=_RaisingIptablesManager(),
        cid_manager=_RaisingCidManager(cid=52),
        api_client_factory=_FakeApiFactory().make,
        command_runner=_CopyingCommandRunner(),
        popen_factory=_FakePopenFactory(),
        unix_socket_factory=_FakeSocketFactory([]),
        temp_root=tmp_path,
        install_exit_handlers=False,
    )
    sandbox._session_id = "sess-log-export"  # noqa: SLF001
    sandbox._log_path = log_path  # noqa: SLF001
    sandbox._session_temp_dir = runtime_dir  # noqa: SLF001
    sandbox._process = cast(Any, _FakeProcess())  # noqa: SLF001

    sandbox.stop()

    artifact = (
        tmp_path
        / ".strangeclaw"
        / "sessions"
        / "sess-log-export"
        / "outputs"
        / "system"
        / "firecracker.log.tail.txt"
    )
    assert artifact.exists()
    content = artifact.read_text(encoding="utf-8")
    assert "abc123" not in content
    assert "super-secret" not in content
    assert "sk-test-secret-123456" not in content
    assert "[REDACTED]" in content
    assert len(content.encode("utf-8")) <= 128 + 1


def _build_firecracker_config(
    tmp_path: Path,
    *,
    host_expose_enabled: bool = False,
    host_expose_ports: tuple[int, ...] = (),
    log_export_enabled: bool = False,
    log_export_max_bytes: int = 32 * 1024,
) -> FirecrackerConfig:
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
        host_expose_enabled=host_expose_enabled,
        host_expose_ports=host_expose_ports,
        log_export_enabled=log_export_enabled,
        log_export_max_bytes=log_export_max_bytes,
    )


def _build_preboot_config(tmp_path: Path, **overrides: Any) -> FirePrebootConfig:
    rootfs_copy = tmp_path / "rootfs-copy.ext4"
    rootfs_copy.write_text("copy", encoding="utf-8")
    params: dict[str, Any] = {
        "rootfs_path": rootfs_copy,
        "tap_name": "fc-testtap",
        "guest_mac": "06:00:AC:10:00:02",
        "guest_cid": 52,
        "vsock_uds_path": tmp_path / "fire.vsock",
        "log_path": tmp_path / "firecracker.log",
        "network": {
            "ip": "172.16.0.2",
            "gateway": "172.16.0.1",
            "netmask": "255.255.255.252",
            "dns": ["8.8.8.8", "1.1.1.1"],
        },
        "agent_config": _agent_config(
            model="openai/gpt-4.1-mini",
            api_key="sk-test",
            max_tokens=1024,
            temperature=0.2,
        ),
    }
    params.update(overrides)
    return FirePrebootConfig(**params)


def _agent_config(
    model: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    api_base: str | None = None,
) -> dict[str, Any]:
    llm: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if api_base is not None:
        llm["api_base"] = api_base
    return {
        "llm": llm,
        "tools": {
            "shell": True,
            "web_search": True,
            "web_fetch": True,
            "http_request": True,
        },
        "web_search": {
            "endpoint": "https://api.search.brave.com/res/v1/web/search",
            "format": "brave",
            "max_results": 10,
        },
        "web_fetch": {"max_chars": 20000},
        "skills": {"directory": "./skills", "max_file_chars": 20000},
        "approval_mode": "review",
        "max_iterations": 50,
        "context": {"token_budget": 4000, "summary_threshold": 10, "max_output_chars": 8000},
    }


def _build_allocation() -> TapNetworkAllocation:
    return TapNetworkAllocation(
        session_id="session-test",
        session_index=0,
        tap_name="fc123456789abc",
        tap_ip="172.16.0.1",
        guest_ip="172.16.0.2",
        host_iface="eth0",
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


class _FakeCommandRunner:
    def __init__(
        self,
        *,
        route_json: str,
        route_text: str,
        addr_outputs: list[str] | None = None,
        fail_link_del: bool = False,
    ) -> None:
        self._route_json = route_json
        self._route_text = route_text
        self._addr_outputs = list(addr_outputs or [])
        self._fail_link_del = fail_link_del
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        key = tuple(args)
        if key == ("ip", "-j", "route", "list", "default"):
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout=self._route_json,
                stderr="",
            )
        if key == ("ip", "route", "show", "default"):
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout=self._route_text,
                stderr="",
            )
        if key == ("ip", "-4", "-o", "addr", "show"):
            stdout = ""
            if self._addr_outputs:
                stdout = self._addr_outputs.pop(0)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")
        if len(args) >= 4 and args[:3] == ["ip", "link", "del"] and self._fail_link_del:
            return subprocess.CompletedProcess(
                args=args,
                returncode=1,
                stdout="",
                stderr="Cannot find device",
            )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


class _ScriptedCommandRunner:
    def __init__(
        self,
        *,
        returncodes: dict[tuple[str, ...], list[int]] | None = None,
        outputs: dict[tuple[str, ...], str] | None = None,
        default_returncode: int = 0,
    ) -> None:
        self._returncodes = {key: list(values) for key, values in (returncodes or {}).items()}
        self._outputs = dict(outputs or {})
        self._default_returncode = default_returncode
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(args))
        key = tuple(args)
        returncode = self._default_returncode
        if key in self._returncodes and self._returncodes[key]:
            returncode = self._returncodes[key].pop(0)
        stdout = self._outputs.get(key, "")
        stderr = "" if returncode == 0 else "error"
        return subprocess.CompletedProcess(
            args=args,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def calls_as_tuples(self) -> list[tuple[str, ...]]:
        return [tuple(call) for call in self.calls]


class _CopyingCommandRunner:
    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        if len(args) == 4 and args[0] == "cp":
            source = Path(args[2])
            dest = Path(args[3])
            dest.write_bytes(source.read_bytes())
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


class _FakeTapManager:
    def __init__(self, allocation: TapNetworkAllocation) -> None:
        self._allocation = allocation
        self.create_calls: list[str] = []
        self.destroyed: list[str] = []

    def create(self, *, session_id: str, max_retries: int = 16) -> TapNetworkAllocation:
        del max_retries
        self.create_calls.append(session_id)
        return self._allocation

    def destroy(self, tap_name: str) -> None:
        self.destroyed.append(tap_name)


class _FakeIptablesManager:
    def __init__(self) -> None:
        self.applied = False
        self.cleaned = False
        self.apply_ports: tuple[int, ...] = ()
        self.cleanup_ports: tuple[int, ...] = ()

    def apply(
        self,
        allocation: TapNetworkAllocation,
        *,
        host_expose_ports: tuple[int, ...] = (),
    ) -> None:
        del allocation
        self.applied = True
        self.apply_ports = host_expose_ports

    def cleanup(
        self,
        allocation: TapNetworkAllocation,
        *,
        host_expose_ports: tuple[int, ...] = (),
    ) -> None:
        del allocation
        self.cleaned = True
        self.cleanup_ports = host_expose_ports


class _RaisingIptablesManager:
    def apply(
        self,
        allocation: TapNetworkAllocation,
        *,
        host_expose_ports: tuple[int, ...] = (),
    ) -> None:
        del allocation
        del host_expose_ports

    def cleanup(
        self,
        allocation: TapNetworkAllocation,
        *,
        host_expose_ports: tuple[int, ...] = (),
    ) -> None:
        del allocation
        del host_expose_ports
        raise RuntimeError("cleanup failed")


class _FakeCidManager:
    def __init__(self, cid: int) -> None:
        self.allocated = cid
        self.released: list[int] = []

    def allocate(self, *, attempts: int = 10) -> int:
        del attempts
        return self.allocated

    def release(self, cid: int) -> None:
        self.released.append(cid)


class _RaisingCidManager:
    def __init__(self, cid: int) -> None:
        self.allocated = cid

    def allocate(self, *, attempts: int = 10) -> int:
        del attempts
        return self.allocated

    def release(self, cid: int) -> None:
        del cid
        raise RuntimeError("release failed")


class _FakeApiFactory:
    def __init__(self) -> None:
        self._clients: list[_FakeApiClient] = []

    def make(self, api_socket: str) -> _FakeApiClient:
        client = _FakeApiClient(api_socket=Path(api_socket))
        self._clients.append(client)
        return client

    def paths(self) -> list[str]:
        if not self._clients:
            return []
        return [path for path, _payload in self._clients[-1].calls]

    def payload_for(self, path: str) -> dict[str, Any]:
        if not self._clients:
            raise AssertionError("No API clients recorded.")
        for call_path, payload in self._clients[-1].calls:
            if call_path == path:
                return payload
        raise AssertionError(f"No payload recorded for path {path}.")


class _FakeApiClient:
    def __init__(self, api_socket: Path) -> None:
        self._api_socket = api_socket
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._vsock_uds: Path | None = None

    def put(self, path: str, payload: Mapping[str, Any]) -> None:
        self.calls.append((path, dict(payload)))
        if path == "/vsock":
            uds_path = payload.get("uds_path")
            if isinstance(uds_path, str):
                self._vsock_uds = Path(uds_path)
        if path == "/actions" and self._vsock_uds is not None:
            self._vsock_uds.write_text("", encoding="utf-8")


class _SecretApiFactory:
    def __init__(self) -> None:
        self._clients: list[_SecretApiClient] = []

    def make(self, api_socket: str) -> _SecretApiClient:
        client = _SecretApiClient(api_socket=Path(api_socket))
        self._clients.append(client)
        return client


class _SecretApiClient(_FakeApiClient):
    def put(self, path: str, payload: Mapping[str, Any]) -> None:
        super().put(path, payload)
        if path == "/logger":
            log_path = payload.get("log_path")
            if isinstance(log_path, str):
                Path(log_path).write_text(
                    "Authorization: Bearer sk-secret-log\n",
                    encoding="utf-8",
                )


class _FakePopenFactory:
    def __init__(self) -> None:
        self.processes: list[_FakeProcess] = []

    def __call__(self, args: list[str]) -> _FakeProcess:
        api_index = args.index("--api-sock") + 1
        api_path = Path(args[api_index])
        api_path.write_text("", encoding="utf-8")
        process = _FakeProcess()
        self.processes.append(process)
        return process


class _SecretPopenFactory:
    def __call__(self, args: list[str]) -> _FakeProcess:
        api_index = args.index("--api-sock") + 1
        api_path = Path(args[api_index])
        api_path.write_text("", encoding="utf-8")
        process = _FakeProcess()
        process.stderr = _FakeStderr("api_key=sk-secret-stderr")
        return process


class _FakeProcess:
    def __init__(self) -> None:
        self._running = True
        self.stderr = _FakeStderr("")

    def poll(self) -> int | None:
        return None if self._running else 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self._running = False
        return 0

    def terminate(self) -> None:
        self._running = False

    def kill(self) -> None:
        self._running = False


class _FakeStderr:
    def __init__(self, content: str) -> None:
        self._content = content

    def read(self) -> str:
        return self._content


class _FakeSocketFactory:
    def __init__(self, sockets: list[_FakeConnectedSocket]) -> None:
        self.sockets = sockets
        self._index = 0

    def __call__(self) -> _FakeConnectedSocket:
        if self._index >= len(self.sockets):
            raise AssertionError("No more fake sockets configured.")
        sock = self.sockets[self._index]
        self._index += 1
        return sock


class _FakeConnectedSocket:
    def __init__(
        self,
        *,
        recv_chunks: list[bytes] | None = None,
        connect_error: Exception | None = None,
    ) -> None:
        self._recv_chunks = list(recv_chunks or [])
        self._connect_error = connect_error
        self.sent: list[bytes] = []
        self.closed = False

    def settimeout(self, timeout: float | None) -> None:
        del timeout

    def connect(self, address: Any) -> None:
        del address
        if self._connect_error is not None:
            raise self._connect_error

    def sendall(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self, bufsize: int) -> bytes:
        del bufsize
        if not self._recv_chunks:
            return b""
        return self._recv_chunks.pop(0)

    def close(self) -> None:
        self.closed = True


class _RaisingTapManager:
    def __init__(self, allocation: TapNetworkAllocation) -> None:
        self._allocation = allocation

    def create(self, *, session_id: str, max_retries: int = 16) -> TapNetworkAllocation:
        del session_id
        del max_retries
        return self._allocation

    def destroy(self, tap_name: str) -> None:
        del tap_name
        raise RuntimeError("destroy failed")
