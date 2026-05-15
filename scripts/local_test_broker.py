#!/usr/bin/env python3
"""Helpers for broker-backed local manual tool checks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent.broker_client import BrokerClient
from agent.tools import Tools
from host_secrets import load_secrets
from sandbox.broker import RequestBroker
from sandbox.host_services import HostServiceServer


@contextmanager
def broker_backed_tools(
    config: dict[str, Any],
    *,
    secrets_path: str | None = None,
) -> Iterator[Tools]:
    """Yield a Tools instance wired to an in-process host broker."""
    resolved_secrets: Path | None = None
    if secrets_path is not None:
        resolved_secrets = Path(secrets_path).expanduser()
    credentials = load_secrets(resolved_secrets)
    broker = RequestBroker(credentials=credentials, config=config)
    server = HostServiceServer()
    server.register("broker", broker.handle)
    server.start()
    try:
        client = BrokerClient(server)
        yield Tools(config=config, broker=client)
    finally:
        server.stop()
