"""Tests for host service dispatch and broker client in-process path."""

from __future__ import annotations

from collections import deque
from typing import Any

import pytest

from agent.broker_client import BrokerClient, HostServiceError
from sandbox.broker import RequestBroker
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


def test_broker_client_yolo_surfaces_broker_internal_error_as_payload() -> None:
    broker = RequestBroker(
        credentials={
            "notion": {
                "auth_type": "bearer",
                "token": "secret-token",
                "allowed_hosts": ["api.notion.com"],
                "allowed_methods": ["GET"],
                "allowed_paths": ["/*"],
                "allowed_schemes": ["https"],
                "protected_headers": ["Authorization"],
                "default_headers": {},
                "max_response_bytes": 4096,
                "rate_limit": None,
            }
        },
        config={},
    )

    def _explode(payload: dict[str, object]) -> dict[str, object]:
        del payload
        raise RuntimeError("boom secret-token")

    broker._handlers["explode"] = _explode  # noqa: SLF001
    server = HostServiceServer()
    server.register("broker", broker.handle)
    client = BrokerClient(server)

    payload = client.call("broker", {"action": "explode"})

    assert payload["success"] is False
    assert payload["error"] == "internal_error"
    detail = str(payload.get("detail", ""))
    assert "secret-token" not in detail
    assert "[REDACTED]" in detail


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


def test_broker_client_fire_surfaces_broker_internal_error_as_payload() -> None:
    broker = RequestBroker(
        credentials={
            "notion": {
                "auth_type": "bearer",
                "token": "secret-token",
                "allowed_hosts": ["api.notion.com"],
                "allowed_methods": ["GET"],
                "allowed_paths": ["/*"],
                "allowed_schemes": ["https"],
                "protected_headers": ["Authorization"],
                "default_headers": {},
                "max_response_bytes": 4096,
                "rate_limit": None,
            }
        },
        config={},
    )

    def _explode(payload: dict[str, object]) -> dict[str, object]:
        del payload
        raise RuntimeError("boom secret-token")

    broker._handlers["explode"] = _explode  # noqa: SLF001
    server = HostServiceServer()
    server.register("broker", broker.handle)
    inbound: deque[dict[str, Any]] = deque()

    def send_fn(event: dict[str, Any]) -> None:
        inbound.append(server.handle_incoming(event))

    def receive_fn(timeout_seconds: float | None) -> dict[str, Any] | None:
        del timeout_seconds
        if not inbound:
            return None
        return inbound.popleft()

    client = BrokerClient(mode="fire", send_fn=send_fn, receive_fn=receive_fn)
    payload = client.call("broker", {"action": "explode"})

    assert payload["success"] is False
    assert payload["error"] == "internal_error"
    detail = str(payload.get("detail", ""))
    assert "secret-token" not in detail
    assert "[REDACTED]" in detail


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
