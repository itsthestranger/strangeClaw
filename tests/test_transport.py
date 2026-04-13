"""Tests for in-process transport."""

from __future__ import annotations

import threading
import time
from typing import Any

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


def test_vsock_transport_exchanges_events_with_unix_socketpair() -> None:
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
