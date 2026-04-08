"""Yolo sandbox implementation."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from agent.agent import Agent, AgentError, LLMRuntime
from agent.transport import InProcessTransport


class YoloSandbox:
    """In-process sandbox interface."""

    def __init__(
        self,
        *,
        skills_dir: str = "./skills",
        max_iterations: int = 50,
        output_dir: str = "/output",
        token_budget: int = 4000,
        summary_threshold: int = 10,
        llm_factory: Callable[[dict[str, Any]], LLMRuntime] | None = None,
    ) -> None:
        self._skills_dir = skills_dir
        self._max_iterations = max_iterations
        self._output_dir = output_dir
        self._token_budget = token_budget
        self._summary_threshold = summary_threshold
        self._llm_factory = llm_factory

        self._host_transport: InProcessTransport | None = None
        self._agent_transport: InProcessTransport | None = None
        self._agent_thread: threading.Thread | None = None
        self._thread_error: Exception | None = None

    def run(self, task: dict[str, Any]) -> None:
        """Start an agent run for a task."""
        if self._agent_thread is not None and self._agent_thread.is_alive():
            raise RuntimeError("YoloSandbox is already running.")

        host_transport, agent_transport = InProcessTransport.pair()
        self._host_transport = host_transport
        self._agent_transport = agent_transport
        self._thread_error = None

        agent = Agent(
            transport=agent_transport,
            skills_dir=self._skills_dir,
            max_iterations=self._max_iterations,
            output_dir=self._output_dir,
            token_budget=self._token_budget,
            summary_threshold=self._summary_threshold,
            llm_factory=self._llm_factory,
        )
        self._agent_thread = threading.Thread(target=self._run_agent, args=(agent,), daemon=True)
        self._agent_thread.start()
        self.send(task)

    def send(self, event: dict[str, Any]) -> None:
        """Send an event to the agent."""
        transport = self._require_host_transport()
        transport.send(event)

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
        """Receive an event from the agent."""
        transport = self._require_host_transport()
        return transport.receive(timeout_seconds=timeout_seconds)

    def stop(self) -> None:
        """Stop the agent run."""
        if self._host_transport is None:
            return

        try:
            self._host_transport.send({"type": "stop"})
        except Exception:
            pass

        if self._agent_thread is not None:
            self._agent_thread.join(timeout=2.0)

        if self._host_transport is not None:
            self._host_transport.close()
        if self._agent_transport is not None:
            self._agent_transport.close()

        self._host_transport = None
        self._agent_transport = None
        self._agent_thread = None

    def _run_agent(self, agent: Agent) -> None:
        try:
            agent.run()
        except AgentError as exc:
            if "Received stop while waiting for user reply." in str(exc):
                return
            self._thread_error = exc
        except Exception as exc:  # pragma: no cover - defensive
            self._thread_error = exc

    def _require_host_transport(self) -> InProcessTransport:
        if self._host_transport is None:
            raise RuntimeError("YoloSandbox is not running.")
        if self._thread_error is not None:
            raise RuntimeError(f"Agent thread failed: {self._thread_error}") from self._thread_error
        return self._host_transport
