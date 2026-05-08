"""Client for host-side broker calls from inside the agent runtime."""

from __future__ import annotations

import json
import socket
import uuid
from collections.abc import Callable
from typing import Any, Protocol


class HostServiceError(RuntimeError):
    """Raised when a host service request fails or cannot be delivered."""


class HostServiceDispatcher(Protocol):
    """Minimal dispatch API needed by yolo-mode broker calls."""

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]: ...


class _SocketLike(Protocol):
    def settimeout(self, timeout: float | None) -> None: ...

    def connect(self, address: Any) -> None: ...

    def sendall(self, payload: bytes) -> None: ...

    def recv(self, bufsize: int) -> bytes: ...

    def close(self) -> None: ...


class BrokerClient:
    """Mode-aware host service client."""

    def __init__(
        self,
        *,
        mode: str,
        server: HostServiceDispatcher | None = None,
        port: int = 5001,
        host_cid: int | None = None,
        timeout_seconds: float = 30.0,
        socket_family: int | None = None,
        unix_socket_path: str | None = None,
        socket_factory: Callable[[int, int], _SocketLike] | None = None,
    ) -> None:
        if mode not in {"yolo", "fire"}:
            raise ValueError("mode must be 'yolo' or 'fire'.")
        if mode == "yolo" and server is None:
            raise ValueError("server is required when mode='yolo'.")
        if mode == "fire" and port <= 0:
            raise ValueError("port must be greater than zero for fire mode.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        self._mode = mode
        self._server = server
        self._port = port
        self._host_cid = (
            host_cid
            if host_cid is not None
            else int(getattr(socket, "VMADDR_CID_HOST", 2))
        )
        self._timeout_seconds = timeout_seconds
        self._socket_family = socket_family
        self._unix_socket_path = unix_socket_path
        self._socket_factory = socket_factory or socket.socket

        self._conn: _SocketLike | None = None
        self._recv_buffer = ""

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request payload and return the host service payload response."""
        if not isinstance(payload, dict):
            raise HostServiceError("Broker payload must be a JSON object.")

        request_id = uuid.uuid4().hex
        request = {
            "service": "broker",
            "request_id": request_id,
            "payload": payload,
        }

        if self._mode == "yolo":
            if self._server is None:
                raise HostServiceError("Yolo broker client is missing a dispatcher server.")
            response = self._server._dispatch(request)
            return self._validate_response(request_id=request_id, response=response)

        response = self._call_over_socket(request)
        return self._validate_response(request_id=request_id, response=response)

    def close(self) -> None:
        """Close any open fire-mode connection."""
        if self._conn is None:
            return
        try:
            self._conn.close()
        except Exception:
            pass
        self._conn = None
        self._recv_buffer = ""

    def _call_over_socket(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            conn = self._require_connection()
            conn.settimeout(self._timeout_seconds)
            line = json.dumps(request, ensure_ascii=True) + "\n"
            conn.sendall(line.encode("utf-8"))
            response_line = self._recv_line()
            parsed = json.loads(response_line)
        except Exception as exc:
            self.close()
            raise HostServiceError(f"Host service transport failure: {exc}") from exc

        if not isinstance(parsed, dict):
            raise HostServiceError("Host service response must be a JSON object.")
        return parsed

    def _recv_line(self) -> str:
        newline_idx = self._recv_buffer.find("\n")
        if newline_idx >= 0:
            line = self._recv_buffer[:newline_idx]
            self._recv_buffer = self._recv_buffer[newline_idx + 1 :]
            return line

        conn = self._require_connection()
        while True:
            chunk = conn.recv(64 * 1024)
            if not chunk:
                raise HostServiceError("Host service connection closed unexpectedly.")
            self._recv_buffer += chunk.decode("utf-8", errors="strict")
            newline_idx = self._recv_buffer.find("\n")
            if newline_idx >= 0:
                line = self._recv_buffer[:newline_idx]
                self._recv_buffer = self._recv_buffer[newline_idx + 1 :]
                return line

    def _require_connection(self) -> _SocketLike:
        if self._conn is not None:
            return self._conn

        family = self._socket_family
        if family is None:
            family = int(getattr(socket, "AF_VSOCK", 40))
        conn = self._socket_factory(family, socket.SOCK_STREAM)
        conn.settimeout(self._timeout_seconds)
        if family == socket.AF_UNIX:
            if not isinstance(self._unix_socket_path, str) or not self._unix_socket_path:
                raise HostServiceError(
                    "unix_socket_path is required for AF_UNIX fire-mode broker testing."
                )
            conn.connect(self._unix_socket_path)
        else:
            conn.connect((self._host_cid, self._port))
        self._conn = conn
        return conn

    @staticmethod
    def _validate_response(*, request_id: str, response: dict[str, Any]) -> dict[str, Any]:
        returned_id = response.get("request_id")
        if returned_id != request_id:
            raise HostServiceError(
                f"Host service request_id mismatch (expected={request_id}, got={returned_id})."
            )
        success = response.get("success")
        if success is not True:
            error = response.get("error")
            if isinstance(error, str) and error:
                raise HostServiceError(error)
            raise HostServiceError("Host service returned an unknown failure.")
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise HostServiceError("Host service success response is missing payload object.")
        return payload
