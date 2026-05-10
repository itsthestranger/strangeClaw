"""In-process host services dispatch for broker-backed tools."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class HostServiceServer:
    """Registry and dispatcher for host-side services."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def register(self, service: str, handler: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        """Register a service handler by name."""
        self._handlers[service] = handler

    def start(self) -> None:
        """Start the service server (no-op for in-process mode)."""
        return None

    def stop(self) -> None:
        """Stop the service server (no-op for in-process mode)."""
        return None

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a request to a registered host service handler."""
        request_id = str(request.get("request_id", ""))
        service = request.get("service")
        payload = request.get("payload", {})

        if not isinstance(service, str):
            return {
                "request_id": request_id,
                "success": False,
                "error": f"unknown service: {service}",
            }

        handler = self._handlers.get(service)
        if handler is None:
            return {
                "request_id": request_id,
                "success": False,
                "error": f"unknown service: {service}",
            }

        try:
            result = handler(payload if isinstance(payload, dict) else {})
        except Exception as exc:
            return {"request_id": request_id, "success": False, "error": str(exc)}

        return {"request_id": request_id, "success": True, "payload": result}
