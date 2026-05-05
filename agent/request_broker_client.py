"""Broker client abstraction used by agent tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from broker.credentials import CredentialConfigError, load_host_credentials
from broker.request_broker import RequestBroker


class RequestBrokerClient(Protocol):
    """Minimal request-broker client interface."""

    def execute(self, request: Mapping[str, Any]) -> dict[str, Any]: ...


class InProcessRequestBrokerClient:
    """In-process broker client for Yolo-mode parity."""

    def __init__(self, broker: RequestBroker) -> None:
        self._broker = broker

    def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._broker.execute(request)


class UnavailableRequestBrokerClient:
    """Fallback client used when broker initialization is unavailable."""

    def __init__(self, message: str) -> None:
        self._message = message

    def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        integration: str | None = None
        raw_integration = request.get("integration")
        if isinstance(raw_integration, str) and raw_integration.strip():
            integration = raw_integration.strip()
        return {
            "success": False,
            "error_code": "broker_unavailable",
            "message": self._message,
            "integration": integration,
        }


def build_request_broker_client(config: Mapping[str, Any] | None = None) -> RequestBrokerClient:
    """Create the default broker client for tool execution."""
    broker_cfg = config.get("request_broker") if isinstance(config, Mapping) else None
    if isinstance(broker_cfg, Mapping) and broker_cfg.get("enabled") is False:
        return UnavailableRequestBrokerClient("request_broker is disabled by config.")

    try:
        credential_registry = load_host_credentials()
    except CredentialConfigError as exc:
        return UnavailableRequestBrokerClient(
            f"request_broker credentials could not be loaded: {exc}"
        )

    return InProcessRequestBrokerClient(RequestBroker(credential_registry))
