"""In-process client for calling host-side services."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sandbox.host_services import HostServiceServer

LOGGER = logging.getLogger(__name__)


class HostServiceError(Exception):
    """Raised when a host service call fails."""


class BrokerClient:
    """Client wrapper for host service request/response calls."""

    def __init__(
        self,
        server: HostServiceServer | None = None,
        *,
        mode: str = "yolo",
        send_fn: Callable[[dict[str, Any]], None] | None = None,
        receive_fn: Callable[[float | None], dict[str, Any] | None] | None = None,
        service_timeouts: dict[str, float] | None = None,
    ) -> None:
        if mode not in {"yolo", "fire"}:
            raise ValueError(f"Unsupported broker client mode: {mode}")
        if mode == "yolo" and server is None:
            raise ValueError("BrokerClient yolo mode requires a HostServiceServer instance.")
        if mode == "fire" and (send_fn is None or receive_fn is None):
            raise ValueError("BrokerClient fire mode requires send_fn and receive_fn.")
        self._mode = mode
        self._server = server
        self._send_fn = send_fn
        self._receive_fn = receive_fn
        self._service_timeouts = dict(service_timeouts or {})

    def call(self, service: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call a host service and return its payload."""
        request_id = uuid.uuid4().hex
        request = {
            "request_id": request_id,
            "service": service,
            "payload": payload,
        }
        response = self._call_yolo(request) if self._mode == "yolo" else self._call_fire(request)

        if not bool(response.get("success")):
            message = response.get("error")
            raise HostServiceError(str(message))

        response_payload = response.get("payload")
        if not isinstance(response_payload, dict):
            raise HostServiceError("service response payload must be a mapping")
        return response_payload

    def _call_yolo(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._server is None:
            raise HostServiceError("Host service server is not configured.")
        return self._server._dispatch(request)

    def _call_fire(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._send_fn is None or self._receive_fn is None:
            raise HostServiceError("Fire-mode broker transport is not configured.")
        self._send_fn(
            {
                "type": "broker_request",
                "request_id": request["request_id"],
                "service": request["service"],
                "payload": request["payload"],
            }
        )
        while True:
            timeout = self._service_timeouts.get(
                str(request.get("service", "")),
                self._service_timeouts.get("_default", 60.0),
            )
            event = self._receive_fn(timeout)
            if event is None:
                raise HostServiceError(f"{request['service']} response timeout")
            if event.get("type") != "broker_response":
                LOGGER.warning(
                    "Discarding non-broker event while waiting for broker response: %s",
                    event.get("type"),
                )
                continue
            if event.get("request_id") != request["request_id"]:
                LOGGER.warning(
                    "Discarding broker_response with mismatched request_id while waiting "
                    "for %s.",
                    request["request_id"],
                )
                continue
            success = bool(event.get("success"))
            if success:
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    raise HostServiceError("broker response payload must be a mapping")
                return {
                    "request_id": str(event.get("request_id", "")),
                    "success": True,
                    "payload": payload,
                }
            return {
                "request_id": str(event.get("request_id", "")),
                "success": False,
                "error": str(event.get("error", "broker call failed")),
            }
