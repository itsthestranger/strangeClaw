"""Tests for host service dispatch and broker client in-process path."""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from agent.broker_client import BrokerClient, HostServiceError
from sandbox.host_services import HostServiceServer


def test_broker_client_round_trip_echo() -> None:
    server = HostServiceServer()
    server.register("echo", lambda payload: payload)
    client = BrokerClient(server)

    result = client.call("echo", {"x": 1})

    assert result == {"x": 1}


def test_broker_client_unknown_service_raises() -> None:
    server = HostServiceServer()
    client = BrokerClient(server)

    with pytest.raises(HostServiceError, match=r"unknown service: missing"):
        client.call("missing", {"x": 1})


def test_broker_client_handler_exception_raises() -> None:
    def _broken_handler(payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("boom")

    server = HostServiceServer()
    server.register("broken", _broken_handler)
    client = BrokerClient(server)

    with pytest.raises(HostServiceError, match="boom"):
        client.call("broken", {"x": 1})


def test_broker_client_fire_mode_round_trip() -> None:
    server = HostServiceServer()
    server.register("echo", lambda payload: payload)
    inbound: deque[dict[str, Any]] = deque()

    def send_fn(event: dict[str, Any]) -> None:
        inbound.append(server.handle_incoming(event))

    def receive_fn(timeout_seconds: float | None) -> dict[str, Any] | None:
        del timeout_seconds
        if not inbound:
            return None
        return inbound.popleft()

    client = BrokerClient(mode="fire", send_fn=send_fn, receive_fn=receive_fn)
    result = client.call("echo", {"x": 1})

    assert result == {"x": 1}


def test_broker_client_fire_mode_timeout_raises() -> None:
    sent: list[dict[str, Any]] = []

    def send_fn(event: dict[str, Any]) -> None:
        sent.append(event)

    def receive_fn(timeout_seconds: float | None) -> dict[str, Any] | None:
        del timeout_seconds
        return None

    client = BrokerClient(mode="fire", send_fn=send_fn, receive_fn=receive_fn)

    with pytest.raises(HostServiceError, match="broker response timeout"):
        client.call("broker", {"action": "list_integrations"})
    assert sent


def test_broker_client_fire_mode_discards_non_broker_events(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = HostServiceServer()
    server.register("echo", lambda payload: payload)
    queued: deque[dict[str, Any]] = deque()

    def send_fn(event: dict[str, Any]) -> None:
        queued.append({"type": "message", "role": "status", "content": "ignored"})
        queued.append(server.handle_incoming(event))

    def receive_fn(timeout_seconds: float | None) -> dict[str, Any] | None:
        del timeout_seconds
        if not queued:
            return None
        return queued.popleft()

    client = BrokerClient(mode="fire", send_fn=send_fn, receive_fn=receive_fn)
    with caplog.at_level("WARNING"):
        result = client.call("echo", {"x": 1})

    assert result == {"x": 1}
    assert "Discarding non-broker event while waiting for broker response" in caplog.text
