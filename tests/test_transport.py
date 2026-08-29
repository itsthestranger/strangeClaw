"""Tests for in-process transport."""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

import agent.transport as transport_module
from agent.protocol import decode_event, encode_event
from agent.transport import InProcessTransport, VsockTransport


def test_transport_pair_exchanges_events_across_threads() -> None:
    host, agent = InProcessTransport.pair()
    received: list[dict[str, Any]] = []

    def receiver() -> None:
        event = agent.receive(timeout_seconds=2.0)
        if event is not None:
            received.append(event)

    thread = threading.Thread(target=receiver)
    thread.start()
    host.send({"type": "stop"})
    thread.join()

    assert received == [{"type": "stop"}]


def test_transport_receive_without_timeout_returns_event() -> None:
    host, agent = InProcessTransport.pair()
    host.send({"type": "agent_ready"})
    received = agent.receive()
    assert received == {"type": "agent_ready"}


def test_transport_receive_returns_none_on_timeout() -> None:
    transport = InProcessTransport()
    start = time.monotonic()
    event = transport.receive(timeout_seconds=1.0)
    elapsed = time.monotonic() - start

    assert event is None
    assert elapsed >= 0.95


def test_transport_close_blocks_send_and_receive() -> None:
    transport = InProcessTransport()
    transport.close()

    try:
        transport.send({"type": "stop"})
    except RuntimeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("send should raise on closed transport")

    try:
        transport.receive()
    except RuntimeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("receive should raise on closed transport")


def test_vsock_transport_accepts_and_exchanges_events_over_unix_mock(
    monkeypatch: Any,
) -> None:
    accepted_conn = _FakeSocket(
        incoming_chunks=[encode_event({"type": "stop"}).encode("utf-8")],
    )
    listener = _FakeListener(connection=accepted_conn)

    def fake_socket_factory(family: int, sock_type: int) -> _FakeListener:
        assert family == socket.AF_UNIX
        assert sock_type == socket.SOCK_STREAM
        return listener

    monkeypatch.setattr(transport_module.socket, "socket", fake_socket_factory)

    transport = VsockTransport(
        guest_port=5000,
        socket_family=socket.AF_UNIX,
        unix_socket_path="/tmp/test-vsock.sock",
    )
    assert listener.bound_path == "/tmp/test-vsock.sock"
    assert listener.listen_backlog == 1
    assert listener.closed is True
    assert decode_event(accepted_conn.sent[0].decode("utf-8")) == {"type": "agent_ready"}

    received = transport.receive(timeout_seconds=1.0)
    assert received == {"type": "stop"}

    transport.send({"type": "message", "role": "status", "content": "hello"})
    assert decode_event(accepted_conn.sent[1].decode("utf-8")) == {
        "type": "message",
        "role": "status",
        "content": "hello",
    }

    transport.close()


def test_vsock_transport_exchanges_events_with_fake_socket() -> None:
    fake_socket = _FakeSocket(
        incoming_chunks=[encode_event({"type": "stop"}).encode("utf-8")],
    )
    transport = VsockTransport(guest_port=5000, connected_socket=fake_socket)

    assert decode_event(fake_socket.sent[0].decode("utf-8")) == {"type": "agent_ready"}

    received = transport.receive(timeout_seconds=1.0)
    assert received == {"type": "stop"}

    transport.send({"type": "message", "role": "status", "content": "hello"})
    assert decode_event(fake_socket.sent[1].decode("utf-8")) == {
        "type": "message",
        "role": "status",
        "content": "hello",
    }

    transport.close()


def test_vsock_transport_receive_returns_none_on_timeout() -> None:
    fake_socket = _FakeSocket(incoming_chunks=[])
    transport = VsockTransport(guest_port=5000, connected_socket=fake_socket)

    start = time.monotonic()
    event = transport.receive(timeout_seconds=1.0)
    elapsed = time.monotonic() - start

    assert event is None
    assert elapsed >= 0.95
    transport.close()


class _FakeSocket:
    def __init__(self, incoming_chunks: list[bytes]) -> None:
        self._incoming = incoming_chunks[:]
        self._timeout: float | None = None
        self.sent: list[bytes] = []
        self.closed = False

    def setsockopt(self, level: int, option: int, value: int) -> None:
        del level
        del option
        del value

    def settimeout(self, timeout: float | None) -> None:
        self._timeout = timeout

    def sendall(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self, bufsize: int) -> bytes:
        del bufsize
        if self._incoming:
            return self._incoming.pop(0)
        if self._timeout is not None:
            time.sleep(self._timeout)
            raise TimeoutError()
        raise AssertionError("recv called without timeout and without incoming chunks")

    def close(self) -> None:
        self.closed = True


class _FakeListener:
    def __init__(self, connection: _FakeSocket) -> None:
        self._connection = connection
        self.bound_path: str | None = None
        self.listen_backlog: int | None = None
        self.closed = False

    def bind(self, path: str) -> None:
        self.bound_path = path

    def listen(self, backlog: int) -> None:
        self.listen_backlog = backlog

    def accept(self) -> tuple[_FakeSocket, Any]:
        return self._connection, object()

    def close(self) -> None:
        self.closed = True


class _EndlessNoNewlineSocket:
    """A peer that streams bytes forever and never sends a frame delimiter."""

    def __init__(self, chunk: bytes = b"x" * 4096) -> None:
        self._chunk = chunk
        self.recv_calls = 0
        self.closed = False
        self.sent: list[bytes] = []

    def setsockopt(self, level: int, option: int, value: int) -> None:
        del level, option, value

    def settimeout(self, timeout: float | None) -> None:
        del timeout

    def sendall(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv(self, bufsize: int) -> bytes:
        del bufsize
        self.recv_calls += 1
        return self._chunk

    def close(self) -> None:
        self.closed = True


def test_vsock_transport_bounds_stream_without_newline() -> None:
    """A peer that never sends a newline must fail fast, not grow the buffer."""
    cap = 64 * 1024
    fake_socket = _EndlessNoNewlineSocket()
    transport = VsockTransport(
        guest_port=5000,
        connected_socket=fake_socket,
        max_frame_bytes=cap,
    )

    try:
        transport.receive(timeout_seconds=30.0)
    except RuntimeError as exc:
        assert "exceeded max_frame_bytes (65536)" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("receive should raise once the frame cap is exceeded")

    # Bounded: stopped within one chunk of the cap instead of looping forever.
    assert fake_socket.recv_calls <= (cap // 4096) + 1
    # Clean: decoder state reset, connection dropped.
    assert transport._buffer == ""
    assert fake_socket.closed is True

    # The failed stream is never resumed, in either direction.
    for operation in (
        lambda: transport.receive(timeout_seconds=1.0),
        lambda: transport.send({"type": "stop"}),
    ):
        try:
            operation()
        except RuntimeError as exc:
            assert "exceeded max_frame_bytes" in str(exc)
        else:  # pragma: no cover - defensive
            raise AssertionError("failed transport must not resume")


def test_vsock_transport_frame_cap_survives_receive_timeouts() -> None:
    """The cap covers buffer state accumulated across separate receive calls."""
    fake_socket = _EndlessNoNewlineSocket(chunk=b"y" * 3000)
    transport = VsockTransport(
        guest_port=5000,
        connected_socket=fake_socket,
        max_frame_bytes=8192,
    )

    # Two short reads leave a partial frame buffered; it is not cleared on timeout.
    transport._buffer = "z" * 8000
    try:
        transport.receive(timeout_seconds=30.0)
    except RuntimeError as exc:
        assert "exceeded max_frame_bytes" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("receive should raise once the frame cap is exceeded")

    assert fake_socket.recv_calls == 1
    assert transport._buffer == ""


def test_vsock_transport_rejects_non_positive_max_frame_bytes() -> None:
    fake_socket = _FakeSocket(incoming_chunks=[])
    try:
        VsockTransport(guest_port=5000, connected_socket=fake_socket, max_frame_bytes=0)
    except ValueError as exc:
        assert "max_frame_bytes" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("max_frame_bytes must be validated")


def test_vsock_transport_reports_truncated_final_frame_on_eof() -> None:
    fake_socket = _FakeSocket(incoming_chunks=[b'{"type":"sto', b""])
    transport = VsockTransport(guest_port=5000, connected_socket=fake_socket)

    try:
        transport.receive(timeout_seconds=1.0)
    except RuntimeError as exc:
        assert "peer closed the connection" in str(exc)
        assert "truncated final frame of 12 bytes" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("receive should raise on a truncated final frame")

    assert transport._buffer == ""
    assert fake_socket.closed is True


def test_vsock_transport_reports_clean_eof_without_truncation_note() -> None:
    fake_socket = _FakeSocket(incoming_chunks=[b""])
    transport = VsockTransport(guest_port=5000, connected_socket=fake_socket)

    try:
        transport.receive(timeout_seconds=1.0)
    except RuntimeError as exc:
        assert "peer closed the connection" in str(exc)
        assert "truncated final frame" not in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("receive should raise when the peer closes")


def test_vsock_transport_passes_normal_traffic_split_across_reads() -> None:
    """Normal traffic, including frames split across reads, is unaffected."""
    fake_socket = _FakeSocket(
        incoming_chunks=[
            b'{"type":"agent_read',
            b'y"}\n{"type":"st',
            b'op"}\n',
        ]
    )
    transport = VsockTransport(
        guest_port=5000,
        connected_socket=fake_socket,
        max_frame_bytes=64 * 1024,
    )

    assert transport.receive(timeout_seconds=1.0) == {"type": "agent_ready"}
    assert transport.receive(timeout_seconds=1.0) == {"type": "stop"}
    assert transport._buffer == ""
    transport.close()


def test_vsock_transport_accepts_frame_just_under_the_cap() -> None:
    payload = encode_event({"type": "message", "role": "status", "content": "a" * 4000})
    fake_socket = _FakeSocket(incoming_chunks=[payload.encode("utf-8")])
    transport = VsockTransport(
        guest_port=5000,
        connected_socket=fake_socket,
        max_frame_bytes=len(payload),
    )

    received = transport.receive(timeout_seconds=1.0)
    assert received is not None
    assert received["content"] == "a" * 4000
    transport.close()
