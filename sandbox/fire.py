"""Firecracker sandbox primitives and interface."""

from __future__ import annotations

import atexit
import fcntl
import http.client
import json
import logging
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

from agent.broker_client import HostServiceError
from agent.protocol import decode_event, encode_event
from host_secrets import load_secrets
from sandbox.broker import RequestBroker
from sandbox.host_services import HostServiceServer
from sandbox.llm_service import LLMService

DEFAULT_BOOT_ARGS = "console=ttyS0 reboot=k panic=1 pci=off init=/sbin/init"
_HOST_IFACE_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,15}$")
_MAC_ADDR_PATTERN = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
_DEFAULT_TAP_COUNTER_PATH = Path("~/.strangeclaw/tap_counter").expanduser()
_TAP_NETWORK_PREFIX = "172.16"
_TAP_NETMASK = "255.255.255.252"
_TAP_CIDR_SUFFIX = 30
_MAX_TAP_SUBNETS = 16384
_FIRE_TAP_NAME_PATTERN = re.compile(r"^fc[0-9a-f]{12}$")
_GUEST_VSOCK_PORT = 5000
_DEFAULT_API_SOCKET_WAIT_SECONDS = 5.0
_DEFAULT_VSOCK_RETRY_SECONDS = 0.5
_DEFAULT_FIRE_DNS = ["8.8.8.8", "1.1.1.1"]
_CID_MIN = 3
_CID_MAX = 4294967294
_CID_RETRY_ATTEMPTS = 10
_DEFAULT_CID_LOCK_PATH = Path("~/.strangeclaw/firecracker_cids.json").expanduser()
_DEFAULT_LOG_EXPORT_MAX_BYTES = 32 * 1024
_LOG_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"(?i)\b(authorization)\b(\s*[:=]\s*)(bearer\s+[^\s,;]+)"),
        r"\1\2[REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|authorization|token|secret|password)\b(\s*[:=]\s*)([^\s,;]+)"
        ),
        r"\1\2[REDACTED]",
    ),
    (
        re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+"),
        r"\1 [REDACTED]",
    ),
    (
        re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b"),
        "[REDACTED]",
    ),
)
LOGGER = logging.getLogger(__name__)


class FirecrackerConfigError(ValueError):
    """Raised when firecracker config is invalid."""


class FirecrackerAPIError(RuntimeError):
    """Raised when a Firecracker API call fails."""


class VMBootError(RuntimeError):
    """Raised when FireSandbox cannot boot/connect a Firecracker VM."""


@dataclass(frozen=True)
class FirecrackerConfig:
    """Validated host-side Firecracker config."""

    binary: Path
    kernel: Path
    rootfs: Path
    vcpu: int
    mem_mb: int
    host_iface: str | None
    boot_timeout: float
    log_export_enabled: bool = False
    log_export_max_bytes: int = _DEFAULT_LOG_EXPORT_MAX_BYTES

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> FirecrackerConfig:
        """Load and validate Firecracker config values from app config."""
        fire_section = config.get("firecracker")
        if not isinstance(fire_section, Mapping):
            raise FirecrackerConfigError("Config field firecracker must be a mapping.")

        binary = _require_file_path(
            fire_section.get("binary"),
            field_name="firecracker.binary",
            executable=True,
        )
        kernel = _require_file_path(
            fire_section.get("kernel"),
            field_name="firecracker.kernel",
            executable=False,
        )
        rootfs = _require_file_path(
            fire_section.get("rootfs"),
            field_name="firecracker.rootfs",
            executable=False,
        )
        vcpu = _require_positive_int(fire_section.get("vcpu"), "firecracker.vcpu")
        mem_mb = _require_positive_int(fire_section.get("mem_mb"), "firecracker.mem_mb")
        boot_timeout = _require_positive_float(
            fire_section.get("boot_timeout"),
            "firecracker.boot_timeout",
        )
        host_iface = _validate_host_iface(fire_section.get("host_iface"))
        log_export_enabled, log_export_max_bytes = _parse_log_export(fire_section)

        return cls(
            binary=binary,
            kernel=kernel,
            rootfs=rootfs,
            vcpu=vcpu,
            mem_mb=mem_mb,
            host_iface=host_iface,
            boot_timeout=boot_timeout,
            log_export_enabled=log_export_enabled,
            log_export_max_bytes=log_export_max_bytes,
        )


def load_firecracker_config(config: Mapping[str, Any]) -> FirecrackerConfig:
    """Load validated Firecracker config from app config."""
    return FirecrackerConfig.from_mapping(config)


@dataclass(frozen=True)
class FirePrebootConfig:
    """Pre-boot runtime values needed for Firecracker API configuration."""

    rootfs_path: Path
    tap_name: str
    guest_mac: str
    guest_cid: int
    vsock_uds_path: Path
    log_path: Path
    network: dict[str, Any]
    agent_config: dict[str, Any]


class _HTTPResponseLike(Protocol):
    status: int

    def read(self) -> bytes: ...


class _HTTPConnectionLike(Protocol):
    def request(
        self,
        method: str,
        url: str,
        body: str,
        headers: Mapping[str, str],
    ) -> None: ...

    def getresponse(self) -> _HTTPResponseLike: ...

    def close(self) -> None: ...


class FirecrackerRequestClient(Protocol):
    """Minimal client contract for Firecracker API requests."""

    def put(self, path: str, payload: Mapping[str, Any]) -> None: ...


class CommandRunner(Protocol):
    """Minimal subprocess runner contract for host command execution."""

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]: ...


class TapManagerLike(Protocol):
    """Contract for TAP allocation lifecycle dependencies."""

    def create(self, *, session_id: str, max_retries: int = 16) -> TapNetworkAllocation: ...

    def destroy(self, tap_name: str) -> None: ...


class FirewallManagerLike(Protocol):
    """Contract for host firewall lifecycle dependencies."""

    def apply(
        self,
        allocation: TapNetworkAllocation,
    ) -> None: ...

    def cleanup(
        self,
        allocation: TapNetworkAllocation,
    ) -> None: ...


class CidManagerLike(Protocol):
    """Contract for guest CID lease lifecycle dependencies."""

    def allocate(self, *, attempts: int = _CID_RETRY_ATTEMPTS) -> int: ...

    def release(self, cid: int) -> None: ...


class PopenProcess(Protocol):
    """Minimal process contract used by FireSandbox lifecycle."""

    stderr: Any | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class PopenFactory(Protocol):
    """Factory for spawning Firecracker processes."""

    def __call__(self, args: list[str]) -> PopenProcess: ...


class ConnectedSocket(Protocol):
    """Minimal connected-socket contract for host vsock stream."""

    def settimeout(self, timeout: float | None) -> None: ...

    def connect(self, address: Any) -> None: ...

    def sendall(self, payload: bytes) -> None: ...

    def recv(self, bufsize: int) -> bytes: ...

    def close(self) -> None: ...


class UnixSocketFactory(Protocol):
    """Factory for host AF_UNIX stream sockets."""

    def __call__(self) -> ConnectedSocket: ...


@dataclass(frozen=True)
class TapNetworkAllocation:
    """Allocated network tuple for a Firecracker TAP device."""

    session_id: str
    session_index: int
    tap_name: str
    tap_ip: str
    guest_ip: str
    host_iface: str

    @property
    def tap_cidr(self) -> str:
        return f"{self.tap_ip}/{_TAP_CIDR_SUFFIX}"

    @property
    def netmask(self) -> str:
        return _TAP_NETMASK

    @property
    def gateway(self) -> str:
        return self.tap_ip


def sanitize_session_id(session_id: str) -> str:
    """Validate and return a safe session id for host-side resource names."""
    if not isinstance(session_id, str) or not session_id:
        raise FirecrackerConfigError("session_id must be a non-empty string.")
    if not _SESSION_ID_PATTERN.fullmatch(session_id):
        raise FirecrackerConfigError(
            "Invalid session_id format. Expected characters [a-zA-Z0-9-] only."
        )
    return session_id


def tap_name_for_session(session_id: str) -> str:
    """Return deterministic TAP name: fc + first 12 hex chars of SHA-256(session_id)."""
    sanitized = sanitize_session_id(session_id)
    return f"fc{sha256(sanitized.encode('utf-8')).hexdigest()[:12]}"


def _default_command_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=False)


def _default_popen_factory(args: list[str]) -> PopenProcess:
    return subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _default_unix_socket_factory() -> ConnectedSocket:
    return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)


class TapDeviceManager:
    """Allocates and manages TAP devices for Firecracker sessions."""

    def __init__(
        self,
        *,
        host_iface: str | None,
        counter_path: Path | None = None,
        command_runner: CommandRunner | None = None,
    ) -> None:
        if host_iface is not None and not _HOST_IFACE_PATTERN.fullmatch(host_iface):
            raise FirecrackerConfigError(
                "Config field firecracker.host_iface has invalid format: "
                f"{host_iface!r}. Expected 1-15 chars [A-Za-z0-9._-]."
            )
        self._host_iface_override = host_iface
        self._counter_path = (counter_path or _DEFAULT_TAP_COUNTER_PATH).expanduser()
        self._command_runner: CommandRunner = command_runner or _default_command_runner

    def create(self, *, session_id: str, max_retries: int = 16) -> TapNetworkAllocation:
        """Create TAP device and return allocation metadata."""
        if max_retries <= 0:
            raise ValueError("max_retries must be greater than zero.")

        sanitized_session_id = sanitize_session_id(session_id)
        tap_name = tap_name_for_session(sanitized_session_id)
        host_iface = self._detect_host_iface()
        last_collision: tuple[str, str] | None = None

        for _ in range(max_retries):
            session_index = self._next_session_index()
            tap_ip, guest_ip = _tap_guest_ips_from_index(session_index)
            used_ips = self._list_assigned_ipv4_addrs()
            if tap_ip in used_ips or guest_ip in used_ips:
                last_collision = (tap_ip, guest_ip)
                continue

            allocation = TapNetworkAllocation(
                session_id=sanitized_session_id,
                session_index=session_index,
                tap_name=tap_name,
                tap_ip=tap_ip,
                guest_ip=guest_ip,
                host_iface=host_iface,
            )
            self._create_tap_interface(allocation)
            return allocation

        collision_text = ""
        if last_collision is not None:
            collision_text = (
                f" Last attempted pair tap_ip={last_collision[0]} guest_ip={last_collision[1]}."
            )
        raise FirecrackerConfigError(
            f"Failed to allocate TAP subnet without IP collision after {max_retries} attempts."
            f"{collision_text}"
        )

    def destroy(self, tap_name: str) -> None:
        """Best-effort TAP teardown. Never raises."""
        if not isinstance(tap_name, str) or not tap_name:
            return
        try:
            self._run_command(["ip", "link", "del", tap_name], check=False)
        except Exception:
            return

    def _create_tap_interface(self, allocation: TapNetworkAllocation) -> None:
        self._run_command(
            ["ip", "tuntap", "add", "dev", allocation.tap_name, "mode", "tap"],
            check=True,
        )
        try:
            self._run_command(
                ["ip", "addr", "add", allocation.tap_cidr, "dev", allocation.tap_name],
                check=True,
            )
            self._run_command(
                ["ip", "link", "set", "dev", allocation.tap_name, "up"],
                check=True,
            )
        except FirecrackerConfigError:
            self.destroy(allocation.tap_name)
            raise

    def _detect_host_iface(self) -> str:
        if self._host_iface_override:
            return self._host_iface_override

        route_json = self._run_command(
            ["ip", "-j", "route", "list", "default"],
            check=False,
        )
        iface = _parse_default_route_iface_json(route_json)
        if iface is not None:
            return iface

        route_text = self._run_command(["ip", "route", "show", "default"], check=False)
        iface = _parse_default_route_iface_text(route_text)
        if iface is not None:
            return iface

        raise FirecrackerConfigError(
            "Unable to detect host outbound interface from default route. "
            "Set firecracker.host_iface in config."
        )

    def _run_command(self, args: list[str], *, check: bool) -> str:
        result = self._command_runner(args)
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or (
                f"exit code {result.returncode}"
            )
            raise FirecrackerConfigError(f"Command failed: {' '.join(args)}: {detail}")
        return result.stdout

    def _list_assigned_ipv4_addrs(self) -> set[str]:
        output = self._run_command(["ip", "-4", "-o", "addr", "show"], check=False)
        ips: set[str] = set()
        for line in output.splitlines():
            match = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+)/\d+\b", line)
            if match is not None:
                ips.add(match.group(1))
        return ips

    def _next_session_index(self) -> int:
        self._counter_path.parent.mkdir(parents=True, exist_ok=True)
        with self._counter_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                raw_value = handle.read().strip()
                current = 0
                if raw_value:
                    try:
                        current = int(raw_value)
                    except ValueError as exc:
                        raise FirecrackerConfigError(
                            f"Invalid TAP counter value in {self._counter_path}: {raw_value!r}"
                        ) from exc
                if current < 0:
                    raise FirecrackerConfigError(
                        f"Invalid TAP counter value in {self._counter_path}: {raw_value!r}"
                    )
                handle.seek(0)
                handle.truncate()
                handle.write(f"{current + 1}\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return current


class IptablesManager:
    """Manage Firecracker TAP firewall rules via iptables."""

    def __init__(self, *, command_runner: CommandRunner | None = None) -> None:
        self._command_runner: CommandRunner = command_runner or _default_command_runner

    def apply(
        self,
        allocation: TapNetworkAllocation,
    ) -> None:
        """Apply per-session firewall/NAT rules and shared conntrack rule."""
        _validate_iptables_context(allocation)
        self._ensure_conntrack_rule()
        self._run_checked(
            [
                "iptables",
                "-t",
                "nat",
                "-A",
                "POSTROUTING",
                "-o",
                allocation.host_iface,
                "-s",
                allocation.guest_ip,
                "-j",
                "MASQUERADE",
            ]
        )
        self._run_checked(
            [
                "iptables",
                "-A",
                "FORWARD",
                "-i",
                allocation.tap_name,
                "-o",
                allocation.host_iface,
                "-j",
                "ACCEPT",
            ]
        )
        self._run_checked(
            [
                "iptables",
                "-A",
                "INPUT",
                "-i",
                allocation.tap_name,
                "-j",
                "DROP",
            ]
        )

    def cleanup(
        self,
        allocation: TapNetworkAllocation,
    ) -> None:
        """Best-effort teardown for per-session rules. Never raises."""
        _validate_iptables_context(allocation)
        self._run_unchecked(
            [
                "iptables",
                "-t",
                "nat",
                "-D",
                "POSTROUTING",
                "-o",
                allocation.host_iface,
                "-s",
                allocation.guest_ip,
                "-j",
                "MASQUERADE",
            ]
        )
        self._run_unchecked(
            [
                "iptables",
                "-D",
                "FORWARD",
                "-i",
                allocation.tap_name,
                "-o",
                allocation.host_iface,
                "-j",
                "ACCEPT",
            ]
        )
        self._run_unchecked(
            [
                "iptables",
                "-D",
                "INPUT",
                "-i",
                allocation.tap_name,
                "-j",
                "DROP",
            ]
        )
        if not self._has_active_fire_taps():
            self._run_unchecked(
                [
                    "iptables",
                    "-D",
                    "FORWARD",
                    "-m",
                    "conntrack",
                    "--ctstate",
                    "RELATED,ESTABLISHED",
                    "-j",
                    "ACCEPT",
                ]
            )

    def _ensure_conntrack_rule(self) -> None:
        check_args = [
            "iptables",
            "-C",
            "FORWARD",
            "-m",
            "conntrack",
            "--ctstate",
            "RELATED,ESTABLISHED",
            "-j",
            "ACCEPT",
        ]
        existing = self._run_unchecked(check_args)
        if existing.returncode == 0:
            return
        self._run_checked(
            [
                "iptables",
                "-A",
                "FORWARD",
                "-m",
                "conntrack",
                "--ctstate",
                "RELATED,ESTABLISHED",
                "-j",
                "ACCEPT",
            ]
        )

    def _has_active_fire_taps(self) -> bool:
        result = self._run_unchecked(["ip", "-o", "link", "show"])
        if result.returncode != 0:
            return True
        for line in result.stdout.splitlines():
            tap_name = _parse_link_name(line)
            if tap_name is None:
                continue
            if _FIRE_TAP_NAME_PATTERN.fullmatch(tap_name):
                return True
        return False

    def _run_checked(self, args: list[str]) -> None:
        result = self._command_runner(args)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or (
                f"exit code {result.returncode}"
            )
            raise FirecrackerConfigError(f"Command failed: {' '.join(args)}: {detail}")

    def _run_unchecked(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return self._command_runner(args)
        except Exception:
            return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="")


class CidLeaseManager:
    """Cross-process guest CID allocator backed by a lock file."""

    def __init__(
        self,
        *,
        lock_path: Path | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._lock_path = (lock_path or _DEFAULT_CID_LOCK_PATH).expanduser()
        self._rng = rng or random.SystemRandom()

    def allocate(self, *, attempts: int = _CID_RETRY_ATTEMPTS) -> int:
        if attempts <= 0:
            raise ValueError("attempts must be greater than zero.")

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                state = self._read_state(handle)
                in_use = {int(item) for item in state.get("active", [])}
                for _ in range(attempts):
                    candidate = int(self._rng.randint(_CID_MIN, _CID_MAX))
                    if candidate in in_use:
                        continue
                    in_use.add(candidate)
                    self._write_state(handle, sorted(in_use))
                    return candidate
                raise FirecrackerConfigError(
                    f"Unable to allocate guest_cid after {attempts} attempts."
                )
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def release(self, cid: int) -> None:
        if not isinstance(cid, int):
            return
        if not self._lock_path.exists():
            return

        try:
            with self._lock_path.open("a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    state = self._read_state(handle)
                    in_use = {int(item) for item in state.get("active", [])}
                    if cid not in in_use:
                        return
                    in_use.remove(cid)
                    self._write_state(handle, sorted(in_use))
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            return

    def _read_state(self, handle: Any) -> dict[str, Any]:
        handle.seek(0)
        raw = handle.read().strip()
        if not raw:
            return {"active": []}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FirecrackerConfigError(
                f"Invalid CID lock file content in {self._lock_path}: {raw!r}"
            ) from exc
        if not isinstance(parsed, dict):
            raise FirecrackerConfigError(
                f"Invalid CID lock file structure in {self._lock_path}: expected object."
            )
        active = parsed.get("active", [])
        if not isinstance(active, list):
            raise FirecrackerConfigError(
                f"Invalid CID lock file structure in {self._lock_path}: active must be list."
            )
        return {"active": active}

    def _write_state(self, handle: Any, active: list[int]) -> None:
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"active": active}, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTP connection over a Unix socket path."""

    def __init__(self, api_socket_path: str, timeout_seconds: float) -> None:
        super().__init__("localhost", timeout=timeout_seconds)
        self._api_socket_path = api_socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(cast(float | None, self.timeout))
        sock.connect(self._api_socket_path)
        self.sock = sock


def _default_connection_factory(
    api_socket_path: str,
    timeout_seconds: float,
) -> _HTTPConnectionLike:
    return _UnixHTTPConnection(api_socket_path=api_socket_path, timeout_seconds=timeout_seconds)


class FirecrackerAPIClient:
    """Small Firecracker API client that talks over --api-sock Unix socket."""

    def __init__(
        self,
        *,
        api_socket_path: str,
        timeout_seconds: float = 5.0,
        connection_factory: Callable[[str, float], _HTTPConnectionLike] | None = None,
    ) -> None:
        if not api_socket_path:
            raise ValueError("api_socket_path must be non-empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        self._api_socket_path = api_socket_path
        self._timeout_seconds = float(timeout_seconds)
        self._connection_factory = connection_factory or _default_connection_factory

    def put(self, path: str, payload: Mapping[str, Any]) -> None:
        """Send a JSON PUT request and raise on non-2xx responses."""
        if not path.startswith("/"):
            raise ValueError("path must start with '/'.")
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        conn = self._connection_factory(self._api_socket_path, self._timeout_seconds)
        response_body = ""
        status = 0
        try:
            conn.request(
                "PUT",
                path,
                body=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            response = conn.getresponse()
            status = int(response.status)
            response_body = response.read().decode("utf-8", errors="replace").strip()
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise FirecrackerAPIError(
                f"Failed Firecracker API PUT {path} via {self._api_socket_path}: {exc}"
            ) from exc
        finally:
            conn.close()

        if status < 200 or status >= 300:
            detail = response_body if response_body else "<empty response body>"
            raise FirecrackerAPIError(
                f"Firecracker API PUT {path} failed with HTTP {status}: {detail}"
            )


def configure_microvm_preboot(
    *,
    api_client: FirecrackerRequestClient,
    config: FirecrackerConfig,
    preboot: FirePrebootConfig,
    credentials: Mapping[str, Any] | None = None,
    llm_config: Mapping[str, Any] | None = None,
) -> None:
    """Configure and start the microVM via sequential pre-boot API PUT calls."""
    _validate_preboot_config(preboot)

    requests: list[tuple[str, dict[str, Any]]] = [
        (
            "/boot-source",
            {
                "kernel_image_path": str(config.kernel),
                "boot_args": DEFAULT_BOOT_ARGS,
            },
        ),
        (
            "/drives/rootfs",
            {
                "drive_id": "rootfs",
                "path_on_host": str(preboot.rootfs_path),
                "is_root_device": True,
                "is_read_only": False,
            },
        ),
        (
            "/machine-config",
            {
                "vcpu_count": config.vcpu,
                "mem_size_mib": config.mem_mb,
            },
        ),
        (
            "/network-interfaces/eth0",
            {
                "iface_id": "eth0",
                "guest_mac": preboot.guest_mac,
                "host_dev_name": preboot.tap_name,
            },
        ),
        (
            "/vsock",
            {
                "guest_cid": preboot.guest_cid,
                "uds_path": str(preboot.vsock_uds_path),
            },
        ),
        (
            "/logger",
            {
                "log_path": str(preboot.log_path),
                "level": "Info",
                "show_level": True,
                "show_log_origin": True,
            },
        ),
        (
            "/mmds/config",
            {
                "network_interfaces": ["eth0"],
                "version": "V2",
            },
        ),
        (
            "/mmds",
            {
                "network": preboot.network,
                "config": preboot.agent_config,
            },
        ),
        (
            "/actions",
            {
                "action_type": "InstanceStart",
            },
        ),
    ]
    for path, payload in requests:
        if path == "/mmds":
            mmds_json = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            _assert_no_secrets(mmds_json, credentials or {}, llm_config=llm_config)
        api_client.put(path, payload)


class FireSandbox:
    """Firecracker sandbox interface."""

    def __init__(
        self,
        *,
        firecracker_config: FirecrackerConfig,
        llm_config: Mapping[str, Any] | None = None,
        agent_config: Mapping[str, Any] | None = None,
        tap_manager: TapManagerLike | None = None,
        iptables_manager: FirewallManagerLike | None = None,
        cid_manager: CidManagerLike | None = None,
        api_client_factory: Callable[[str], FirecrackerRequestClient] | None = None,
        command_runner: CommandRunner | None = None,
        popen_factory: PopenFactory | None = None,
        unix_socket_factory: UnixSocketFactory | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        temp_root: Path | None = None,
        install_exit_handlers: bool = True,
    ) -> None:
        self._firecracker_config = firecracker_config
        self._agent_config_template = _coerce_agent_config_template(
            llm_config=llm_config,
            agent_config=agent_config,
        )

        self._command_runner = command_runner or _default_command_runner
        self._tap_manager = tap_manager or TapDeviceManager(
            host_iface=self._firecracker_config.host_iface,
            command_runner=self._command_runner,
        )
        self._iptables_manager = iptables_manager or IptablesManager(
            command_runner=self._command_runner
        )
        self._cid_manager = cid_manager or CidLeaseManager()
        self._api_client_factory = api_client_factory or (
            lambda api_sock: FirecrackerAPIClient(
                api_socket_path=api_sock,
                timeout_seconds=5.0,
            )
        )
        self._popen_factory = popen_factory or _default_popen_factory
        self._unix_socket_factory = unix_socket_factory or _default_unix_socket_factory
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleep or time.sleep
        self._temp_root = temp_root or Path("/tmp")

        self._process: PopenProcess | None = None
        self._api_socket_path: Path | None = None
        self._vsock_uds_path: Path | None = None
        self._log_path: Path | None = None
        self._session_temp_dir: Path | None = None
        self._rootfs_copy_path: Path | None = None
        self._session_id: str | None = None
        self._allocation: TapNetworkAllocation | None = None
        self._guest_cid: int | None = None
        self._vsock_conn: ConnectedSocket | None = None
        self._recv_buffer = ""
        self._stopping = False
        self._host_service_server: HostServiceServer | None = None

        self._install_exit_handlers = install_exit_handlers
        self._atexit_registered = False
        self._previous_signal_handlers: dict[int, Any] = {}
        self._credentials: dict[str, Any] = {}
        self._start_session_id = "default"
        if self._install_exit_handlers:
            self._register_exit_handlers()

    def __enter__(self) -> FireSandbox:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        del exc_type
        del exc
        del tb
        self.stop()

    def run(self, task: dict[str, Any]) -> None:
        """Compatibility wrapper: start the VM and send one task."""
        session_id = sanitize_session_id(str(task.get("session_id", "")))
        self._start_session_id = session_id
        self.start()
        self.send_task(task)

    def start(self, session_id: str | None = None) -> None:
        """Boot the Firecracker VM and wait for the guest agent to be ready."""
        if self.is_running():
            raise RuntimeError("FireSandbox is already running.")
        if self._process is not None:
            self.stop()
        if self._install_exit_handlers and not self._atexit_registered:
            self._register_exit_handlers()

        if session_id is not None:
            self._start_session_id = sanitize_session_id(str(session_id))
        session_id = sanitize_session_id(self._start_session_id)
        self._session_id = session_id

        try:
            credentials = load_secrets()
            self._credentials = credentials
            request_broker = RequestBroker(
                credentials=credentials,
                config=self._agent_config_template,
            )
            host_services = HostServiceServer()
            host_services.register("broker", request_broker.handle)
            host_services.register("llm", LLMService(self._agent_config_template).handle)
            host_services.start()
            self._host_service_server = host_services
            self._prepare_runtime_paths(session_id=session_id)
            self._copy_rootfs()
            self._allocation = self._tap_manager.create(session_id=session_id)
            self._iptables_manager.apply(self._allocation)
            self._guest_cid = self._cid_manager.allocate()
            self._launch_firecracker()
            self._wait_for_api_socket(timeout_seconds=_DEFAULT_API_SOCKET_WAIT_SECONDS)
            fire_agent_config_payload = self._prepare_agent_config_for_fire()
            self._configure_microvm(agent_config_payload=fire_agent_config_payload)
            self._vsock_conn = self._connect_vsock_with_retry(
                timeout_seconds=self._firecracker_config.boot_timeout
            )
            agent_ready = self.receive(timeout_seconds=self._firecracker_config.boot_timeout)
            if agent_ready is None or agent_ready.get("type") != "agent_ready":
                raise VMBootError(
                    "Did not receive agent_ready after successful vsock CONNECT handshake."
                )
        except Exception as exc:
            diagnostics = self._build_boot_diagnostics()
            self.stop()
            if isinstance(exc, VMBootError):
                raise VMBootError(f"{exc}{diagnostics}") from exc
            raise VMBootError(f"Failed to start FireSandbox: {exc}{diagnostics}") from exc

    def is_running(self) -> bool:
        """Return whether the VM process and guest event stream are active."""
        return bool(
            self._process is not None
            and self._process.poll() is None
            and self._vsock_conn is not None
        )

    def send_task(self, task: dict[str, Any]) -> None:
        """Send a task event to an already-running guest agent."""
        if not self.is_running():
            raise RuntimeError("FireSandbox is not running.")
        self.send(dict(task))

    def send(self, event: dict[str, Any]) -> None:
        """Send an event to the agent."""
        conn = self._require_vsock_conn()
        try:
            payload = encode_event(event).encode("utf-8")
            conn.sendall(payload)
        except OSError as exc:
            raise RuntimeError(f"Failed to send event over vsock: {exc}") from exc

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        """Receive one non-broker event from the agent."""
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be >= 0.")

        deadline: float | None = None
        if timeout_seconds is not None:
            deadline = self._monotonic() + timeout_seconds

        while True:
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    return None
            event = self._receive_raw_event(timeout_seconds=remaining)
            if event is None:
                return None
            event_type = event.get("type")
            if event_type == "broker_request":
                self._handle_broker_request_event(event)
                continue
            if event_type == "broker_response":
                LOGGER.warning(
                    "Received unexpected broker_response from guest; discarding."
                )
                continue
            return event

    def stop(self) -> None:
        """Stop the VM and clean up resources."""
        if self._stopping:
            return
        self._stopping = True
        try:
            self._safe_teardown_step(self._close_vsock_conn)
            self._safe_teardown_step(self._graceful_shutdown_process)
            self._safe_teardown_step(self._export_firecracker_log_artifact)
            if self._host_service_server is not None:
                host_services = self._host_service_server
                self._safe_teardown_step(host_services.stop)
                self._host_service_server = None
            if self._allocation is not None:
                allocation = self._allocation
                self._safe_teardown_step(lambda: self._iptables_manager.cleanup(allocation))
                self._safe_teardown_step(lambda: self._tap_manager.destroy(allocation.tap_name))
                self._allocation = None
            if self._guest_cid is not None:
                guest_cid = self._guest_cid
                self._safe_teardown_step(lambda: self._cid_manager.release(guest_cid))
                self._guest_cid = None
            if self._session_temp_dir is not None:
                session_temp_dir = self._session_temp_dir
                self._safe_teardown_step(
                    lambda: shutil.rmtree(session_temp_dir, ignore_errors=True)
                )
            self._session_temp_dir = None
            self._api_socket_path = None
            self._vsock_uds_path = None
            self._log_path = None
            self._rootfs_copy_path = None
            self._session_id = None
            self._recv_buffer = ""
            self._credentials = {}
        finally:
            self._process = None
            self._host_service_server = None
            self._safe_teardown_step(self._unregister_exit_handlers)
            self._stopping = False

    def _receive_raw_event(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        conn = self._require_vsock_conn()

        line = self._extract_line()
        if line is not None:
            return decode_event(line)
        deadline: float | None = None
        if timeout_seconds is not None:
            deadline = self._monotonic() + timeout_seconds

        while True:
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - self._monotonic()
                if remaining <= 0:
                    return None

            conn.settimeout(remaining)
            try:
                chunk = conn.recv(65 * 1024)
            except TimeoutError:
                return None
            except OSError as exc:
                if self._stopping:
                    return None
                raise RuntimeError(f"Failed to receive event over vsock: {exc}") from exc

            if not chunk:
                if self._stopping:
                    return None
                exit_code = self._process.poll() if self._process is not None else None
                if exit_code is None:
                    raise RuntimeError("Guest vsock connection closed unexpectedly.")
                raise RuntimeError(
                    "Guest vsock connection closed (firecracker exited with code "
                    f"{exit_code})."
                )

            self._recv_buffer += chunk.decode("utf-8", errors="strict")
            line = self._extract_line()
            if line is not None:
                return decode_event(line)

    def _handle_broker_request_event(self, event: dict[str, Any]) -> None:
        request_id = event.get("request_id")
        if not isinstance(request_id, str):
            return
        if self._host_service_server is None:
            self.send(
                {
                    "type": "broker_response",
                    "request_id": request_id,
                    "success": False,
                    "error": "host services unavailable",
                }
            )
            return

        try:
            response = self._host_service_server.handle_incoming(event)
        except Exception as exc:
            response = {
                "type": "broker_response",
                "request_id": request_id,
                "success": False,
                "error": str(exc),
            }
        try:
            self.send(response)
        except Exception as exc:
            raise HostServiceError(f"Failed to send broker_response to guest: {exc}") from exc

    def _safe_teardown_step(self, operation: Callable[[], None]) -> None:
        try:
            operation()
        except Exception:
            return

    def _prepare_agent_config_for_fire(self) -> dict[str, Any]:
        return _sanitize_agent_config_for_mmds(self._agent_config_template)

    def _prepare_runtime_paths(self, *, session_id: str) -> None:
        self._session_temp_dir = Path(
            tempfile.mkdtemp(prefix=f"strangeclaw-{session_id}-", dir=str(self._temp_root))
        )
        self._api_socket_path = self._session_temp_dir / "firecracker.socket"
        self._vsock_uds_path = self._session_temp_dir / "fire.vsock"
        self._log_path = self._session_temp_dir / "firecracker.log"
        self._rootfs_copy_path = self._session_temp_dir / "rootfs.ext4"

    def _copy_rootfs(self) -> None:
        if self._rootfs_copy_path is None:
            raise RuntimeError("Rootfs copy path was not initialized.")
        result = self._command_runner(
            [
                "cp",
                "--reflink=auto",
                str(self._firecracker_config.rootfs),
                str(self._rootfs_copy_path),
            ]
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or (
                f"exit code {result.returncode}"
            )
            raise FirecrackerConfigError(f"Failed to copy rootfs image: {detail}")

    def _launch_firecracker(self) -> None:
        if self._api_socket_path is None:
            raise RuntimeError("API socket path was not initialized.")
        args = [
            str(self._firecracker_config.binary),
            "--api-sock",
            str(self._api_socket_path),
        ]
        self._process = self._popen_factory(args)

    def _wait_for_api_socket(self, *, timeout_seconds: float) -> None:
        if self._api_socket_path is None:
            raise RuntimeError("API socket path was not initialized.")
        deadline = self._monotonic() + timeout_seconds
        while self._monotonic() < deadline:
            if self._api_socket_path.exists():
                return
            if self._process is not None and self._process.poll() is not None:
                raise VMBootError("Firecracker process exited before API socket became available.")
            self._sleep(0.1)
        raise VMBootError(
            f"Timed out waiting for Firecracker API socket: {self._api_socket_path}"
        )

    def _configure_microvm(self, *, agent_config_payload: dict[str, Any]) -> None:
        if self._api_socket_path is None:
            raise RuntimeError("API socket path was not initialized.")
        if self._rootfs_copy_path is None:
            raise RuntimeError("Rootfs copy path was not initialized.")
        if self._allocation is None:
            raise RuntimeError("TAP allocation was not initialized.")
        if self._guest_cid is None:
            raise RuntimeError("guest_cid was not initialized.")
        if self._vsock_uds_path is None:
            raise RuntimeError("vsock uds path was not initialized.")
        if self._log_path is None:
            raise RuntimeError("log path was not initialized.")

        api_client = self._api_client_factory(str(self._api_socket_path))
        preboot = FirePrebootConfig(
            rootfs_path=self._rootfs_copy_path,
            tap_name=self._allocation.tap_name,
            guest_mac="06:00:AC:10:00:02",
            guest_cid=self._guest_cid,
            vsock_uds_path=self._vsock_uds_path,
            log_path=self._log_path,
            network={
                "ip": self._allocation.guest_ip,
                "gateway": self._allocation.tap_ip,
                "netmask": self._allocation.netmask,
                "dns": list(_DEFAULT_FIRE_DNS),
            },
            agent_config=agent_config_payload,
        )
        configure_microvm_preboot(
            api_client=api_client,
            config=self._firecracker_config,
            preboot=preboot,
            credentials=self._credentials,
            llm_config=_extract_llm_config(self._agent_config_template),
        )

    def _connect_vsock_with_retry(self, *, timeout_seconds: float) -> ConnectedSocket:
        if self._vsock_uds_path is None:
            raise RuntimeError("vsock uds path was not initialized.")
        deadline = self._monotonic() + timeout_seconds
        last_error: Exception | None = None
        while self._monotonic() < deadline:
            if not self._vsock_uds_path.exists():
                self._sleep(0.1)
                continue
            conn = self._unix_socket_factory()
            try:
                conn.settimeout(2.0)
                conn.connect(str(self._vsock_uds_path))
                conn.sendall(f"CONNECT {_GUEST_VSOCK_PORT}\n".encode())
                ack = self._recv_line_bytes(conn, timeout_seconds=2.0)
                if not ack.startswith(b"OK "):
                    raise VMBootError(f"Unexpected vsock CONNECT acknowledgment: {ack!r}")
                return conn
            except Exception as exc:
                last_error = exc
                try:
                    conn.close()
                except Exception:
                    pass
                self._sleep(_DEFAULT_VSOCK_RETRY_SECONDS)

        raise VMBootError(
            "Timed out waiting for vsock CONNECT handshake "
            f"on {self._vsock_uds_path}. Last error: {last_error}"
        )

    def _recv_line_bytes(self, conn: ConnectedSocket, *, timeout_seconds: float) -> bytes:
        conn.settimeout(timeout_seconds)
        data = b""
        while b"\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                raise VMBootError("EOF while waiting for vsock handshake line.")
            data += chunk
        line, remainder = data.split(b"\n", 1)
        if remainder:
            self._recv_buffer += remainder.decode("utf-8", errors="strict")
        return line

    def _require_vsock_conn(self) -> ConnectedSocket:
        if self._vsock_conn is None:
            raise RuntimeError("FireSandbox is not connected to guest vsock.")
        return self._vsock_conn

    def _extract_line(self) -> str | None:
        newline_index = self._recv_buffer.find("\n")
        if newline_index < 0:
            return None
        line = self._recv_buffer[: newline_index + 1]
        self._recv_buffer = self._recv_buffer[newline_index + 1 :]
        return line

    def _close_vsock_conn(self) -> None:
        if self._vsock_conn is None:
            return
        try:
            self._vsock_conn.close()
        except Exception:
            pass
        self._vsock_conn = None

    def _graceful_shutdown_process(self) -> None:
        process = self._process
        if process is None:
            return

        self._try_send_ctrl_alt_del()
        if self._wait_for_process_exit(timeout_seconds=5.0):
            return
        try:
            process.terminate()
        except Exception:
            pass
        if self._wait_for_process_exit(timeout_seconds=2.0):
            return
        try:
            process.kill()
        except Exception:
            pass
        self._wait_for_process_exit(timeout_seconds=1.0)

    def _try_send_ctrl_alt_del(self) -> None:
        if self._api_socket_path is None:
            return
        if not self._api_socket_path.exists():
            return
        try:
            api_client = self._api_client_factory(str(self._api_socket_path))
            api_client.put("/actions", {"action_type": "SendCtrlAltDel"})
        except Exception:
            return

    def _wait_for_process_exit(self, *, timeout_seconds: float) -> bool:
        process = self._process
        if process is None:
            return True
        try:
            process.wait(timeout=timeout_seconds)
            return True
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return process.poll() is not None

    def _build_boot_diagnostics(self) -> str:
        parts: list[str] = []
        stderr_text = self._read_process_stderr_best_effort()
        if self._process is not None:
            parts.append(
                f"firecracker stderr: {self._redact_log_text(stderr_text) or '<empty>'}"
            )
        log_tail = self._read_log_tail_best_effort()
        if self._log_path is not None:
            parts.append(f"firecracker log tail: {self._redact_log_text(log_tail) or '<empty>'}")
        if not parts:
            return ""
        return " | " + " | ".join(parts)

    def _read_process_stderr_best_effort(self) -> str:
        process = self._process
        if process is None or process.stderr is None:
            return ""
        try:
            if process.poll() is None:
                return ""
            stderr_value = process.stderr.read()
            if not isinstance(stderr_value, str):
                return ""
            return stderr_value.strip()[-2000:]
        except Exception:
            return ""

    def _read_log_tail_best_effort(self) -> str:
        data = self._read_log_tail_bytes_best_effort(max_bytes=4096)
        if not data:
            return ""
        return data.decode("utf-8", errors="replace").strip()

    def _read_log_tail_bytes_best_effort(self, *, max_bytes: int) -> bytes:
        if max_bytes <= 0:
            return b""
        if self._log_path is None or not self._log_path.exists():
            return b""
        try:
            with self._log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                to_read = min(size, max_bytes)
                if to_read <= 0:
                    return b""
                handle.seek(-to_read, os.SEEK_END)
                return handle.read(to_read)
        except Exception:
            return b""

    def _export_firecracker_log_artifact(self) -> None:
        if not self._firecracker_config.log_export_enabled:
            return
        if self._session_id is None:
            return
        log_tail = self._read_log_tail_bytes_best_effort(
            max_bytes=self._firecracker_config.log_export_max_bytes
        )
        if not log_tail:
            return

        session_outputs_dir = (
            Path.home() / ".strangeclaw" / "sessions" / self._session_id / "outputs" / "system"
        )
        session_outputs_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = session_outputs_dir / "firecracker.log.tail.txt"
        redacted_text = self._redact_log_text(log_tail.decode("utf-8", errors="replace")).strip()
        if not redacted_text:
            return
        artifact_path.write_text(redacted_text + "\n", encoding="utf-8")

    def _redact_log_text(self, text: str) -> str:
        redacted = text
        for pattern, replacement in _LOG_REDACT_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted

    def _register_exit_handlers(self) -> None:
        if self._atexit_registered:
            return
        try:
            atexit.register(self.stop)
            self._atexit_registered = True
        except Exception:
            self._atexit_registered = False

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous = signal.getsignal(sig)
                self._previous_signal_handlers[int(sig)] = previous
                signal.signal(sig, self._handle_exit_signal)
            except Exception:
                continue

    def _unregister_exit_handlers(self) -> None:
        if self._atexit_registered:
            try:
                atexit.unregister(self.stop)
            except Exception:
                pass
            self._atexit_registered = False

        if not self._previous_signal_handlers:
            return
        for sig, handler in list(self._previous_signal_handlers.items()):
            try:
                signal.signal(sig, handler)
            except Exception:
                continue
        self._previous_signal_handlers.clear()

    def _handle_exit_signal(self, signum: int, frame: Any) -> None:
        self.stop()
        previous = self._previous_signal_handlers.get(int(signum))
        if callable(previous) and previous is not self._handle_exit_signal:
            previous(signum, frame)


def _require_file_path(value: Any, *, field_name: str, executable: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FirecrackerConfigError(f"Config field {field_name} must be a non-empty path string.")
    path = Path(value).expanduser()
    if not path.exists():
        raise FirecrackerConfigError(f"Config field {field_name} file not found: {path}")
    if not path.is_file():
        raise FirecrackerConfigError(f"Config field {field_name} must point to a file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise FirecrackerConfigError(f"Config field {field_name} binary is not executable: {path}")
    return path.resolve()


def _require_positive_int(value: Any, field_name: str) -> int:
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise FirecrackerConfigError(f"Config field {field_name} must be an integer.") from exc
    if converted <= 0:
        raise FirecrackerConfigError(f"Config field {field_name} must be greater than zero.")
    return converted


def _require_positive_float(value: Any, field_name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise FirecrackerConfigError(f"Config field {field_name} must be a number.") from exc
    if converted <= 0:
        raise FirecrackerConfigError(f"Config field {field_name} must be greater than zero.")
    return converted


def _validate_host_iface(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FirecrackerConfigError(
            "Config field firecracker.host_iface must be a string or null."
        )
    stripped = value.strip()
    if not stripped:
        return None
    if not _HOST_IFACE_PATTERN.fullmatch(stripped):
        raise FirecrackerConfigError(
            "Config field firecracker.host_iface has invalid format: "
            f"{stripped!r}. Expected 1-15 chars [A-Za-z0-9._-]."
        )
    return stripped


def _parse_log_export(fire_section: Mapping[str, Any]) -> tuple[bool, int]:
    raw = fire_section.get("log_export")
    if raw is None:
        return False, _DEFAULT_LOG_EXPORT_MAX_BYTES
    if not isinstance(raw, Mapping):
        raise FirecrackerConfigError("Config field firecracker.log_export must be a mapping.")

    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise FirecrackerConfigError(
            "Config field firecracker.log_export.enabled must be a boolean."
        )

    max_bytes_raw = raw.get("max_bytes", _DEFAULT_LOG_EXPORT_MAX_BYTES)
    if isinstance(max_bytes_raw, bool):
        raise FirecrackerConfigError(
            "Config field firecracker.log_export.max_bytes must be an integer."
        )
    try:
        max_bytes = int(max_bytes_raw)
    except (TypeError, ValueError) as exc:
        raise FirecrackerConfigError(
            "Config field firecracker.log_export.max_bytes must be an integer."
        ) from exc
    if max_bytes <= 0:
        raise FirecrackerConfigError(
            "Config field firecracker.log_export.max_bytes must be greater than zero."
        )

    return enabled, max_bytes


def _validate_preboot_config(preboot: FirePrebootConfig) -> None:
    if not preboot.rootfs_path.exists() or not preboot.rootfs_path.is_file():
        raise FirecrackerConfigError(f"Rootfs copy not found: {preboot.rootfs_path}")
    if not _HOST_IFACE_PATTERN.fullmatch(preboot.tap_name):
        raise FirecrackerConfigError(
            "tap_name has invalid format: "
            f"{preboot.tap_name!r}. Expected 1-15 chars [A-Za-z0-9._-]."
        )
    if not _MAC_ADDR_PATTERN.fullmatch(preboot.guest_mac):
        raise FirecrackerConfigError(
            "Invalid guest MAC format. Expected six hex bytes, for example "
            "'06:00:AC:10:00:02'."
        )
    if preboot.guest_cid < 3 or preboot.guest_cid > 4294967294:
        raise FirecrackerConfigError(
            "guest_cid must be in range 3..4294967294."
        )
    if not str(preboot.vsock_uds_path):
        raise FirecrackerConfigError("vsock_uds_path must be non-empty.")
    if not str(preboot.log_path):
        raise FirecrackerConfigError("log_path must be non-empty.")
    _validate_network_payload(preboot.network)
    _validate_agent_config_payload(preboot.agent_config)


def _validate_network_payload(network: Mapping[str, Any]) -> None:
    required = ("ip", "gateway", "netmask", "dns")
    missing = [field for field in required if field not in network]
    if missing:
        raise FirecrackerConfigError(
            f"network payload missing required fields: {', '.join(missing)}"
        )
    for field in ("ip", "gateway", "netmask"):
        value = network.get(field)
        if not isinstance(value, str) or not value.strip():
            raise FirecrackerConfigError(f"network.{field} must be a non-empty string.")
    dns = network.get("dns")
    if not isinstance(dns, list) or not dns or not all(isinstance(item, str) for item in dns):
        raise FirecrackerConfigError("network.dns must be a non-empty list of strings.")


def _validate_agent_config_payload(payload: Mapping[str, Any]) -> None:
    tools = payload.get("tools")
    if tools is not None and not isinstance(tools, Mapping):
        raise FirecrackerConfigError("config.tools must be an object when provided.")
    web_search = payload.get("web_search")
    if web_search is not None and not isinstance(web_search, Mapping):
        raise FirecrackerConfigError("config.web_search must be an object when provided.")
    web_fetch = payload.get("web_fetch")
    if web_fetch is not None and not isinstance(web_fetch, Mapping):
        raise FirecrackerConfigError("config.web_fetch must be an object when provided.")
    skills = payload.get("skills")
    if skills is not None and not isinstance(skills, Mapping):
        raise FirecrackerConfigError("config.skills must be an object when provided.")
    context = payload.get("context")
    if context is not None and not isinstance(context, Mapping):
        raise FirecrackerConfigError("config.context must be an object when provided.")
    subagents = payload.get("subagents")
    if subagents is not None and not isinstance(subagents, Mapping):
        raise FirecrackerConfigError("config.subagents must be an object when provided.")
    host_services = payload.get("host_services")
    if host_services is not None:
        if not isinstance(host_services, Mapping):
            raise FirecrackerConfigError("config.host_services must be an object when provided.")
        timeout = host_services.get("llm_timeout_seconds")
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, int):
                raise FirecrackerConfigError(
                    "config.host_services.llm_timeout_seconds must be an integer."
                )
            if timeout <= 0:
                raise FirecrackerConfigError(
                    "config.host_services.llm_timeout_seconds must be greater than zero."
                )

    approval_mode = payload.get("approval_mode")
    if approval_mode is not None and not isinstance(approval_mode, str):
        raise FirecrackerConfigError("config.approval_mode must be a string when provided.")

    max_iterations = payload.get("max_iterations")
    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise FirecrackerConfigError("config.max_iterations must be an integer when provided.")
        if max_iterations <= 0:
            raise FirecrackerConfigError("config.max_iterations must be greater than zero.")


def _coerce_agent_config_template(
    *,
    llm_config: Mapping[str, Any] | None,
    agent_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if agent_config is not None:
        candidate = dict(agent_config)
        sanitized = _sanitize_agent_config_for_mmds(candidate)
        _validate_agent_config_payload(sanitized)
        return candidate

    if llm_config is not None:
        llm_candidate = dict(llm_config)
        return {
            "llm": llm_candidate,
            "tools": {
                "shell": True,
                "web_search": True,
                "web_fetch": True,
                "http_request": True,
                "spawn_subagent": False,
            },
            "web_search": {
                "endpoint": "https://api.search.brave.com/res/v1/web/search",
                "format": "brave",
                "max_results": 10,
            },
            "web_fetch": {"max_response_bytes": 524288},
            "skills": {"directory": "./skills", "max_file_chars": 20000},
            "approval_mode": "review",
            "max_iterations": 50,
            "context": {
                "token_budget": 4000,
                "summary_threshold": 10,
                "max_output_chars": 8000,
            },
            "subagents": {
                "enabled": False,
                "max_children_per_task": 3,
                "max_iterations": 20,
                "timeout_seconds": 600,
                "max_context_chars": 20000,
                "max_result_chars": 20000,
                "max_files_bytes": 10 * 1024 * 1024,
                "journal_events": "summary",
            },
            "host_services": {
                "llm_timeout_seconds": 120,
                "llm_max_request_bytes": 2 * 1024 * 1024,
            },
        }

    raise FirecrackerConfigError("FireSandbox requires either agent_config or llm_config.")


def _sanitize_agent_config_for_mmds(config: Mapping[str, Any]) -> dict[str, Any]:
    tools_raw = config.get("tools")
    if isinstance(tools_raw, Mapping):
        tools = {
            name: bool(tools_raw.get(name, True))
            for name in ("shell", "web_search", "web_fetch", "http_request")
        }
        tools["spawn_subagent"] = bool(tools_raw.get("spawn_subagent", False))
    else:
        tools = {
            "shell": True,
            "web_search": True,
            "web_fetch": True,
            "http_request": True,
            "spawn_subagent": False,
        }

    web_search_raw = config.get("web_search")
    web_search: dict[str, Any]
    if isinstance(web_search_raw, Mapping):
        endpoint_value = web_search_raw.get("endpoint")
        format_value = web_search_raw.get("format")
        max_results_value = web_search_raw.get("max_results", 10)
        if isinstance(max_results_value, bool):
            max_results = 10
        elif isinstance(max_results_value, int):
            max_results = max_results_value if max_results_value > 0 else 10
        elif isinstance(max_results_value, float):
            candidate = int(max_results_value)
            max_results = candidate if candidate > 0 else 10
        elif isinstance(max_results_value, str):
            try:
                candidate = int(max_results_value)
            except ValueError:
                max_results = 10
            else:
                max_results = candidate if candidate > 0 else 10
        else:
            max_results = 10
        web_search = {
            "endpoint": (
                endpoint_value
                if isinstance(endpoint_value, str) and endpoint_value.strip()
                else "https://api.search.brave.com/res/v1/web/search"
            ),
            "format": (
                format_value
                if isinstance(format_value, str) and format_value.strip()
                else "brave"
            ),
            "max_results": max_results,
        }
    else:
        web_search = {
            "endpoint": "https://api.search.brave.com/res/v1/web/search",
            "format": "brave",
            "max_results": 10,
        }

    web_fetch_raw = config.get("web_fetch")
    web_fetch: dict[str, Any]
    if isinstance(web_fetch_raw, Mapping):
        web_fetch = dict(web_fetch_raw)
    else:
        web_fetch = {}
    web_fetch.setdefault("max_response_bytes", 524288)

    skills_raw = config.get("skills")
    skills: dict[str, Any]
    if isinstance(skills_raw, Mapping):
        skills = dict(skills_raw)
    else:
        skills = {}
    skills.setdefault("directory", "./skills")
    skills.setdefault("max_file_chars", 20000)

    approval_mode = config.get("approval_mode", "review")
    if not isinstance(approval_mode, str) or not approval_mode.strip():
        approval_mode = "review"

    max_iterations = 50
    loop_raw = config.get("loop")
    if isinstance(loop_raw, Mapping):
        loop_max = loop_raw.get("max_iterations")
        if isinstance(loop_max, int) and not isinstance(loop_max, bool) and loop_max > 0:
            max_iterations = int(loop_max)
    direct_max = config.get("max_iterations")
    if isinstance(direct_max, int) and not isinstance(direct_max, bool) and direct_max > 0:
        max_iterations = int(direct_max)

    context_raw = config.get("context")
    context: dict[str, Any]
    if isinstance(context_raw, Mapping):
        context = dict(context_raw)
    else:
        context = {}
    context.setdefault("token_budget", 4000)
    context.setdefault("summary_threshold", 10)
    context.setdefault("max_output_chars", 8000)

    host_services_raw = config.get("host_services")
    llm_timeout_seconds = 120
    if isinstance(host_services_raw, Mapping):
        raw_timeout = host_services_raw.get("llm_timeout_seconds", 120)
        if isinstance(raw_timeout, int) and not isinstance(raw_timeout, bool) and raw_timeout > 0:
            llm_timeout_seconds = raw_timeout

    payload = {
        "tools": tools,
        "web_search": web_search,
        "web_fetch": web_fetch,
        "skills": skills,
        "approval_mode": approval_mode,
        "max_iterations": max_iterations,
        "context": context,
        "subagents": _sanitize_subagents_for_mmds(config),
        "host_services": {"llm_timeout_seconds": llm_timeout_seconds},
    }
    _validate_agent_config_payload(payload)
    return payload


def _sanitize_subagents_for_mmds(config: Mapping[str, Any]) -> dict[str, Any]:
    """Produce the non-secret subagent settings delivered to the guest via MMDS."""
    raw = config.get("subagents")
    src: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}

    def _pos_int(key: str, default: int) -> int:
        value = src.get(key, default)
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value if value > 0 else default
        if isinstance(value, float):
            candidate = int(value)
            return candidate if candidate > 0 else default
        if isinstance(value, str):
            try:
                candidate = int(value)
            except ValueError:
                return default
            return candidate if candidate > 0 else default
        return default

    journal_raw = src.get("journal_events", "summary")
    if isinstance(journal_raw, str) and journal_raw.strip().lower() in {"none", "summary", "full"}:
        journal_events = journal_raw.strip().lower()
    else:
        journal_events = "summary"

    return {
        "enabled": bool(src.get("enabled", False)),
        "max_children_per_task": _pos_int("max_children_per_task", 3),
        "max_iterations": _pos_int("max_iterations", 20),
        "timeout_seconds": _pos_int("timeout_seconds", 600),
        "max_context_chars": _pos_int("max_context_chars", 20000),
        "max_result_chars": _pos_int("max_result_chars", 20000),
        "max_files_bytes": _pos_int("max_files_bytes", 10 * 1024 * 1024),
        "journal_events": journal_events,
    }


def _assert_no_secrets(
    mmds_json: str,
    credentials: Mapping[str, Any],
    *,
    llm_config: Mapping[str, Any] | None = None,
) -> None:
    for integration_name, record in credentials.items():
        if not isinstance(record, Mapping):
            continue
        token = record.get("token")
        if not isinstance(token, str) or token == "":
            continue
        if token in mmds_json:
            raise ValueError(f"credential leaked into MMDS: {integration_name}")
    if isinstance(llm_config, Mapping):
        api_key = llm_config.get("api_key")
        if isinstance(api_key, str) and api_key and api_key in mmds_json:
            raise ValueError("LLM API key leaked into MMDS")


def _extract_llm_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    llm = config.get("llm")
    return llm if isinstance(llm, Mapping) else None


def _tap_guest_ips_from_index(session_index: int) -> tuple[str, str]:
    if session_index < 0:
        raise FirecrackerConfigError("session_index must be greater than or equal to zero.")
    effective_index = session_index % _MAX_TAP_SUBNETS
    tap_host_offset = (4 * effective_index) + 1
    guest_host_offset = tap_host_offset + 1
    return _ip_from_host_offset(tap_host_offset), _ip_from_host_offset(guest_host_offset)


def _ip_from_host_offset(host_offset: int) -> str:
    if host_offset < 0 or host_offset > 65535:
        raise FirecrackerConfigError(f"Invalid host offset for TAP /30 allocation: {host_offset}")
    third_octet = host_offset // 256
    fourth_octet = host_offset % 256
    return f"{_TAP_NETWORK_PREFIX}.{third_octet}.{fourth_octet}"


def _parse_default_route_iface_json(output: str) -> str | None:
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None
    for item in parsed:
        if not isinstance(item, Mapping):
            continue
        dev = item.get("dev")
        if isinstance(dev, str) and _HOST_IFACE_PATTERN.fullmatch(dev):
            return dev
    return None


def _parse_default_route_iface_text(output: str) -> str | None:
    match = re.search(r"\bdev\s+([A-Za-z0-9._-]{1,15})\b", output)
    if match is None:
        return None
    return match.group(1)


def _parse_link_name(line: str) -> str | None:
    match = re.match(r"^\d+:\s+([^:]+):", line)
    if match is None:
        return None
    raw_name = match.group(1)
    return raw_name.split("@", 1)[0]


def _validate_iptables_context(allocation: TapNetworkAllocation) -> None:
    if not _HOST_IFACE_PATTERN.fullmatch(allocation.host_iface):
        raise FirecrackerConfigError(
            f"Invalid host interface for iptables rules: {allocation.host_iface!r}"
        )
    if not _HOST_IFACE_PATTERN.fullmatch(allocation.tap_name):
        raise FirecrackerConfigError(
            f"Invalid TAP interface for iptables rules: {allocation.tap_name!r}"
        )
    if not isinstance(allocation.guest_ip, str) or not allocation.guest_ip:
        raise FirecrackerConfigError("Invalid guest_ip for iptables rules.")
