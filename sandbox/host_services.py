"""Host services transport for sandbox-to-host request/response calls."""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

HostServiceHandler = Callable[[dict[str, Any]], dict[str, Any]]
_DEFAULT_SOCKET_BACKLOG = 8
_DEFAULT_ACCEPT_TIMEOUT_SECONDS = 0.2
_DEFAULT_RECV_BYTES = 64 * 1024


class HostServiceServer:
    """Host-side service dispatcher for Yolo (in-process) and Fire (AF_UNIX)."""

    def __init__(
        self,
        *,
        mode: str,
        fire_uds_path: str | Path | None = None,
    ) -> None:
        if mode not in {"yolo", "fire"}:
            raise ValueError("mode must be 'yolo' or 'fire'.")
        self._mode = mode
        self._fire_uds_path = Path(fire_uds_path) if fire_uds_path is not None else None
        if self._mode == "fire" and self._fire_uds_path is None:
            raise ValueError("fire_uds_path is required when mode='fire'.")

        self._handlers: dict[str, HostServiceHandler] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._listener: socket.socket | None = None
        self._connections: set[socket.socket] = set()
        self._lock = threading.Lock()

    def register(self, service: str, handler: HostServiceHandler) -> None:
        """Register a handler for a service name."""
        name = service.strip()
        if not name:
            raise ValueError("service must be a non-empty string.")
        self._handlers[name] = handler

    def start(self) -> None:
        """Start the service server for the configured mode."""
        if self._mode == "yolo":
            return
        if self._thread is not None and self._thread.is_alive():
            return
        if self._fire_uds_path is None:
            raise RuntimeError("fire_uds_path is not set.")

        self._stop_event.clear()
        self._fire_uds_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._fire_uds_path.unlink()
        except FileNotFoundError:
            pass

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self._fire_uds_path))
        listener.listen(_DEFAULT_SOCKET_BACKLOG)
        listener.settimeout(_DEFAULT_ACCEPT_TIMEOUT_SECONDS)
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop server resources. Safe to call repeatedly."""
        self._stop_event.set()
        try:
            if self._listener is not None:
                self._listener.close()
        except Exception:
            pass
        self._listener = None

        with self._lock:
            connections = list(self._connections)
            self._connections.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._thread = None

        if self._mode == "fire" and self._fire_uds_path is not None:
            try:
                self._fire_uds_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _serve(self) -> None:
        listener = self._listener
        if listener is None:
            return
        while not self._stop_event.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            thread = threading.Thread(target=self._serve_connection, args=(conn,), daemon=True)
            thread.start()

    def _serve_connection(self, conn: socket.socket) -> None:
        with self._lock:
            self._connections.add(conn)
        buffer = ""
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = conn.recv(_DEFAULT_RECV_BYTES)
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="strict")
                while True:
                    newline_idx = buffer.find("\n")
                    if newline_idx < 0:
                        break
                    line = buffer[:newline_idx]
                    buffer = buffer[newline_idx + 1 :]
                    response = self._handle_line(line)
                    try:
                        payload = json.dumps(response, ensure_ascii=True) + "\n"
                        conn.sendall(payload.encode("utf-8"))
                    except OSError:
                        return
        finally:
            with self._lock:
                self._connections.discard(conn)
            try:
                conn.close()
            except Exception:
                pass

    def _handle_line(self, line: str) -> dict[str, Any]:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            return {
                "request_id": "",
                "success": False,
                "error": "Invalid JSON request.",
            }
        if not isinstance(request, dict):
            return {
                "request_id": "",
                "success": False,
                "error": "Request must be a JSON object.",
            }
        return self._dispatch(request)

    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one request object and build a normalized response envelope."""
        request_id = request.get("request_id")
        if not isinstance(request_id, str):
            request_id = ""

        service = request.get("service")
        if not isinstance(service, str) or not service.strip():
            return {
                "request_id": request_id,
                "success": False,
                "error": "Missing or invalid service field.",
            }
        payload = request.get("payload")
        if not isinstance(payload, dict):
            return {
                "request_id": request_id,
                "success": False,
                "error": "Missing or invalid payload field.",
            }

        handler = self._handlers.get(service)
        if handler is None:
            return {
                "request_id": request_id,
                "success": False,
                "error": f"unknown service: {service}",
            }
        try:
            handler_result = handler(payload)
        except Exception as exc:
            return {
                "request_id": request_id,
                "success": False,
                "error": str(exc) or "service handler failed",
            }
        if not isinstance(handler_result, dict):
            return {
                "request_id": request_id,
                "success": False,
                "error": "service handler must return a JSON object payload.",
            }
        return {
            "request_id": request_id,
            "success": True,
            "payload": handler_result,
        }
