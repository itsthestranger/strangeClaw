"""Firecracker sandbox primitives and interface."""

from __future__ import annotations

import http.client
import json
import os
import re
import socket
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

DEFAULT_BOOT_ARGS = "console=ttyS0 reboot=k panic=1 pci=off init=/sbin/init"
_HOST_IFACE_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,15}$")
_MAC_ADDR_PATTERN = re.compile(r"^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$")


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
    if not preboot.tap_name:
        raise FirecrackerConfigError("tap_name must be non-empty.")
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
