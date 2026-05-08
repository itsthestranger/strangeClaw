from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from agent.broker_client import BrokerClient, HostServiceError
from sandbox.host_services import HostServiceServer


def test_host_service_server_yolo_dispatches_registered_handler() -> None:
    server = HostServiceServer(mode="yolo")
    server.register("broker", lambda payload: {"echo": payload.get("value")})
    server.start()

    response = server._dispatch(
        {
            "service": "broker",
            "request_id": "req-1",
            "payload": {"value": "ok"},
        }
    )

    assert response == {
        "request_id": "req-1",
        "success": True,
        "payload": {"echo": "ok"},
    }
    server.stop()
    server.stop()


def test_host_service_server_unknown_service_returns_error() -> None:
    server = HostServiceServer(mode="yolo")
    response = server._dispatch(
        {
            "service": "does-not-exist",
            "request_id": "req-1",
            "payload": {},
        }
    )
    assert response["request_id"] == "req-1"
    assert response["success"] is False
    assert "unknown service" in str(response["error"])


def test_broker_client_yolo_roundtrip() -> None:
    server = HostServiceServer(mode="yolo")
    server.register("broker", lambda payload: {"value": payload.get("value", "")})
    client = BrokerClient(mode="yolo", server=server)

    result = client.call({"value": "from-yolo"})
    assert result == {"value": "from-yolo"}


def test_broker_client_fire_roundtrip_over_unix_socket(tmp_path: Path) -> None:
    del tmp_path

    class _RoundTripSocket:
        def __init__(self) -> None:
            self._response_chunks: list[bytes] = []

        def settimeout(self, timeout: float | None) -> None:
            del timeout

        def connect(self, address: object) -> None:
            del address

        def sendall(self, payload: bytes) -> None:
            request = json.loads(payload.decode("utf-8").strip())
            response = {
                "request_id": request["request_id"],
                "success": True,
                "payload": {"echo": request["payload"]["value"]},
            }
            self._response_chunks = [
                (json.dumps(response, ensure_ascii=True) + "\n").encode("utf-8")
            ]

        def recv(self, bufsize: int) -> bytes:
            del bufsize
            if not self._response_chunks:
                return b""
            return self._response_chunks.pop(0)

        def close(self) -> None:
            return None

    mock_socket = _RoundTripSocket()

    client = BrokerClient(
        mode="fire",
        port=5001,
        socket_family=socket.AF_UNIX,
        unix_socket_path="/tmp/mock.sock",
        socket_factory=lambda *_args: mock_socket,
    )
    result = client.call({"value": "from-fire"})
    client.close()

    assert result == {"echo": "from-fire"}


def test_broker_client_raises_on_connection_failure(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.sock"
    client = BrokerClient(
        mode="fire",
        port=5001,
        socket_family=socket.AF_UNIX,
        unix_socket_path=str(missing_path),
        socket_factory=lambda *_args: _FailingSocket(),
    )
    with pytest.raises(HostServiceError, match="transport failure"):
        client.call({"value": "x"})


class _FailingSocket:
    def settimeout(self, timeout: float | None) -> None:
        del timeout

    def connect(self, address: object) -> None:
        del address
        raise OSError("connect failed")

    def sendall(self, payload: bytes) -> None:
        del payload

    def recv(self, bufsize: int) -> bytes:
        del bufsize
        return b""

    def close(self) -> None:
        return None
