"""Tests for host service dispatch and broker client in-process path."""

from __future__ import annotations

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
