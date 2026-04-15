"""Firecracker sandbox primitives and interface."""

from __future__ import annotations

import fcntl
import http.client
import json
import os
import re
import socket
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol, cast

DEFAULT_BOOT_ARGS = "console=ttyS0 reboot=k panic=1 pci=off init=/sbin/init"
_HOST_IFACE_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,15}$")
_MAC_ADDR_PATTERN = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")
_DEFAULT_TAP_COUNTER_PATH = Path("~/.strangeclaw/tap_counter").expanduser()
_TAP_NETWORK_PREFIX = "172.16"
_TAP_NETMASK = "255.255.255.252"
_TAP_CIDR_SUFFIX = 30
_MAX_TAP_SUBNETS = 16384


class FirecrackerConfigError(ValueError):
    """Raised when firecracker config is invalid."""


class FirecrackerAPIError(RuntimeError):
    """Raised when a Firecracker API call fails."""


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

        return cls(
            binary=binary,
            kernel=kernel,
            rootfs=rootfs,
            vcpu=vcpu,
            mem_mb=mem_mb,
            host_iface=host_iface,
            boot_timeout=boot_timeout,
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
    llm: dict[str, Any]


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
                "llm": preboot.llm,
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
        api_client.put(path, payload)


class FireSandbox:
    """Firecracker sandbox interface."""

    def run(self, task: dict[str, Any]) -> None:
        """Start an agent run for a task."""
        raise NotImplementedError("Backlog task B2.4 implements FireSandbox.")

    def send(self, event: dict[str, Any]) -> None:
        """Send an event to the agent."""
        raise NotImplementedError("Backlog task B2.4 implements FireSandbox.")

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        """Receive an event from the agent."""
        raise NotImplementedError("Backlog task B2.4 implements FireSandbox.")

    def stop(self) -> None:
        """Stop the VM and clean up resources."""
        raise NotImplementedError("Backlog task B2.4 implements FireSandbox.")


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
    _validate_llm_payload(preboot.llm)


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


def _validate_llm_payload(llm: Mapping[str, Any]) -> None:
    required = ("model", "api_key", "max_tokens", "temperature")
    missing = [field for field in required if field not in llm]
    if missing:
        raise FirecrackerConfigError(
            f"llm payload missing required fields: {', '.join(missing)}"
        )
    if not isinstance(llm.get("model"), str) or not llm["model"]:
        raise FirecrackerConfigError("llm.model must be a non-empty string.")
    if not isinstance(llm.get("api_key"), str):
        raise FirecrackerConfigError("llm.api_key must be a string.")
    max_tokens = llm.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise FirecrackerConfigError("llm.max_tokens must be a positive integer.")
    temperature = llm.get("temperature")
    if not isinstance(temperature, (int, float)):
        raise FirecrackerConfigError("llm.temperature must be a number.")


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
