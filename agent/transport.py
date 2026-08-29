"""Transport abstractions for host-agent communication."""

from __future__ import annotations

import queue
import socket
import time
from typing import Any, Protocol

from agent.protocol import decode_event, encode_event

# Upper bound on a single newline-delimited frame. A peer that never sends a
# delimiter would otherwise grow the receive buffer without limit, so this is a
# resource guard, not a policy check: it is enforced while bytes are being
# buffered, before any parse happens. It must stay comfortably above the largest
# legitimate frame; the driver is the host LLM proxy, whose own payload cap is
# `host_services.llm_max_request_bytes` (2 MiB by default). This value leaves
# room for that payload plus its JSON event envelope and escaping.
DEFAULT_MAX_FRAME_BYTES = 4 * 1024 * 1024


def derive_max_frame_bytes(llm_max_request_bytes: int) -> int:
    """Frame cap for a caller that never set one explicitly.

    ``max_frame_bytes`` post-dates ``llm_max_request_bytes``, so a config that
    raised the request cap and has no frame key at all must still load. The
    implicit default therefore scales with the request cap instead of
    conflicting with it, while never dropping below the built-in default.

    Lives beside the constant it falls back to so that every implicit cap in the
    project derives the same way and the two cannot drift apart.
    """
    return max(DEFAULT_MAX_FRAME_BYTES, llm_max_request_bytes * 2)


class InProcessTransport:
    """Queue-based in-process transport."""

    def __init__(
        self,
        incoming: queue.Queue[str] | None = None,
        outgoing: queue.Queue[str] | None = None,
    ) -> None:
        shared = incoming if incoming is not None else queue.Queue[str]()
        self._incoming: queue.Queue[str] = shared
        self._outgoing: queue.Queue[str] = outgoing if outgoing is not None else shared
        self._closed = False

    @classmethod
    def pair(cls) -> tuple[InProcessTransport, InProcessTransport]:
        """Create a connected host/agent transport pair."""
        a_to_b: queue.Queue[str] = queue.Queue()
        b_to_a: queue.Queue[str] = queue.Queue()
        return cls(incoming=b_to_a, outgoing=a_to_b), cls(incoming=a_to_b, outgoing=b_to_a)

    def send(self, event: dict[str, Any]) -> None:
        """Send an event."""
        if self._closed:
            raise RuntimeError("Transport is closed.")
        self._outgoing.put(encode_event(event))

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        """Receive an event or None on timeout."""
        if self._closed:
            raise RuntimeError("Transport is closed.")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be >= 0.")

        try:
            if timeout_seconds is None:
                line = self._incoming.get()
            else:
                line = self._incoming.get(timeout=timeout_seconds)
        except queue.Empty:
            return None
        return decode_event(line)

    def close(self) -> None:
        """Close transport resources."""
        self._closed = True


class SocketLike(Protocol):
    """Minimal socket API used by VsockTransport."""

    def setsockopt(self, level: int, option: int, value: int) -> None: ...

    def settimeout(self, timeout: float | None) -> None: ...

    def sendall(self, payload: bytes) -> None: ...

    def recv(self, bufsize: int) -> bytes: ...

    def close(self) -> None: ...


class VsockTransport:
    """Socket-based transport for guest-side host communication."""

    def __init__(
        self,
        guest_port: int,
        *,
        recv_buffer_size: int = 65 * 1024,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        connected_socket: SocketLike | None = None,
        socket_family: int | None = None,
        unix_socket_path: str | None = None,
    ) -> None:
        if guest_port <= 0:
            raise ValueError("guest_port must be greater than zero.")
        if recv_buffer_size <= 0:
            raise ValueError("recv_buffer_size must be greater than zero.")
        if max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be greater than zero.")

        self._recv_buffer_size = recv_buffer_size
        self._max_frame_bytes = max_frame_bytes
        self._closed = False
        self._buffer = ""
        self._failure: str | None = None
        self._listener: socket.socket | None = None

        if connected_socket is not None:
            self._conn = connected_socket
        else:
            self._conn = self._accept_guest_connection(
                guest_port=guest_port,
                socket_family=socket_family,
                unix_socket_path=unix_socket_path,
            )

        try:
            self._conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer_size)
        except OSError:
            # Some socket families/environments may forbid tuning this option.
            pass
        self.send({"type": "agent_ready"})

    def _accept_guest_connection(
        self,
        *,
        guest_port: int,
        socket_family: int | None,
        unix_socket_path: str | None,
    ) -> socket.socket:
        family = socket_family
        if family is None:
            family = int(getattr(socket, "AF_VSOCK", 40))

        listener = socket.socket(family, socket.SOCK_STREAM)
        self._listener = listener

        if family == socket.AF_UNIX:
            if not unix_socket_path:
                raise ValueError("unix_socket_path is required when socket_family=AF_UNIX.")
            listener.bind(unix_socket_path)
        else:
            cid_any = int(getattr(socket, "VMADDR_CID_ANY", 4294967295))
            listener.bind((cid_any, guest_port))

        listener.listen(1)
        conn, _ = listener.accept()
        listener.close()
        self._listener = None
        return conn

    def send(self, event: dict[str, Any]) -> None:
        """Send an event."""
        if self._failure is not None:
            raise RuntimeError(self._failure)
        if self._closed:
            raise RuntimeError("Transport is closed.")
        payload = encode_event(event).encode("utf-8")
        try:
            self._conn.sendall(payload)
        except OSError as exc:
            raise RuntimeError(f"Failed to send event on socket transport: {exc}") from exc

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        """Receive an event or None on timeout."""
        if self._failure is not None:
            raise RuntimeError(self._failure)
        if self._closed:
            raise RuntimeError("Transport is closed.")
        if timeout_seconds is not None and timeout_seconds < 0:
            raise ValueError("timeout_seconds must be >= 0.")

        line = self._extract_line()
        if line is not None:
            return decode_event(line)

        deadline: float | None = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + timeout_seconds

        while True:
            remaining: float | None = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
            self._conn.settimeout(remaining)
            try:
                chunk = self._conn.recv(self._recv_buffer_size)
            except TimeoutError:
                return None
            except OSError as exc:
                raise RuntimeError(f"Failed to receive event on socket transport: {exc}") from exc

            if not chunk:
                pending = len(self._buffer)
                message = "Socket transport peer closed the connection."
                if pending:
                    message += (
                        f" Discarding a truncated final frame of {pending} bytes"
                        " with no trailing newline."
                    )
                raise self._fail(message)

            self._buffer += chunk.decode("utf-8", errors="strict")
            line = self._extract_line()
            if line is not None:
                return decode_event(line)
            # The buffer is instance state that survives timeouts, so an
            # undelimited stream keeps growing across separate receive() calls.
            # Bound it here, where the bytes land. Encoded frames are ASCII
            # JSON, so character count equals byte count for legitimate traffic
            # and under-counts (i.e. trips the cap sooner) for anything else.
            if len(self._buffer) > self._max_frame_bytes:
                raise self._fail(
                    "Socket transport frame exceeded max_frame_bytes "
                    f"({self._max_frame_bytes}) with no newline delimiter."
                )

    def _fail(self, message: str) -> RuntimeError:
        """Reset decoder state, close the stream, and build the fatal error.

        The failed stream is never resumed: every later send/receive raises the
        same error rather than reading more bytes from a peer that has already
        violated the framing contract.
        """
        self._buffer = ""
        self._failure = message
        self.close()
        return RuntimeError(message)

    def close(self) -> None:
        """Close transport resources."""
        self._closed = True
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None
        try:
            self._conn.close()
        except OSError:
            pass

    def _extract_line(self) -> str | None:
        newline_idx = self._buffer.find("\n")
        if newline_idx < 0:
            return None
        line = self._buffer[: newline_idx + 1]
        self._buffer = self._buffer[newline_idx + 1 :]
        return line
