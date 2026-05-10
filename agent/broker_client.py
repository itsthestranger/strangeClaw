"""In-process client for calling host-side services."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sandbox.host_services import HostServiceServer


class HostServiceError(Exception):
    """Raised when a host service call fails."""


class BrokerClient:
    """Client wrapper for host service request/response calls."""

    def __init__(self, server: HostServiceServer) -> None:
        self._server = server

    def call(self, service: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call a host service and return its payload."""
        request = {"request_id": uuid.uuid4().hex, "service": service, "payload": payload}
        response = self._server._dispatch(request)

        if not bool(response.get("success")):
            message = response.get("error")
            raise HostServiceError(str(message))

        response_payload = response.get("payload")
        if not isinstance(response_payload, dict):
            raise HostServiceError("service response payload must be a mapping")
        return response_payload
