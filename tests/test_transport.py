"""Tests for in-process transport."""

from __future__ import annotations

import threading
import time
from typing import Any

from agent.transport import InProcessTransport


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
