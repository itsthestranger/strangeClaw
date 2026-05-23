"""Yolo sandbox implementation."""

from __future__ import annotations

import threading
from typing import Any

from agent.agent import Agent, AgentError
from agent.broker_client import BrokerClient
from agent.llm_types import LLMRuntime
from agent.transport import InProcessTransport
from host_secrets import load_secrets
from sandbox.broker import RequestBroker
from sandbox.host_services import HostServiceServer


class YoloSandbox:
    """In-process sandbox interface."""

    def __init__(
        self,
        *,
        agent_config: dict[str, Any] | None = None,
        skills_dir: str = "./skills",
        max_iterations: int = 50,
        output_dir: str = "/output",
        token_budget: int = 4000,
        summary_threshold: int = 10,
        llm_runtime: LLMRuntime | None = None,
    ) -> None:
        self._agent_config = dict(agent_config) if isinstance(agent_config, dict) else None
        if self._agent_config is not None:
            skills_cfg = self._agent_config.get("skills")
            if isinstance(skills_cfg, dict) and isinstance(skills_cfg.get("directory"), str):
                skills_dir = str(skills_cfg["directory"])
            loop_cfg = self._agent_config.get("loop")
            if isinstance(loop_cfg, dict):
                max_iterations = int(loop_cfg.get("max_iterations", max_iterations))
            context_cfg = self._agent_config.get("context")
            if isinstance(context_cfg, dict):
                token_budget = int(context_cfg.get("token_budget", token_budget))
                summary_threshold = int(context_cfg.get("summary_threshold", summary_threshold))

        self._skills_dir = skills_dir
        self._max_iterations = max_iterations
        self._output_dir = output_dir
        self._token_budget = token_budget
        self._summary_threshold = summary_threshold
        self._llm_runtime = llm_runtime

        self._host_transport: InProcessTransport | None = None
        self._agent_transport: InProcessTransport | None = None
        self._agent_thread: threading.Thread | None = None
        self._thread_error: Exception | None = None
        self._host_service_server: HostServiceServer | None = None
        self._started = False

    def run(self, task: dict[str, Any]) -> None:
        """Compatibility wrapper: start the sandbox and send one task."""
        self.start()
        self.send_task(task)

    def start(self, session_id: str | None = None) -> None:
        """Start the sandbox lifecycle.

        Yolo has no persistent isolation boundary. The per-task runtime is
        created by send_task(), but this shim lets coordinators use the same
        lifecycle interface for all sandbox types.
        """
        del session_id
        self._started = True

    def is_running(self) -> bool:
        """Return whether the Yolo sandbox lifecycle is active and healthy."""
        if not self._started:
            return False
        if self._thread_error is not None:
            return False
        return True

    def send_task(self, task: dict[str, Any]) -> None:
        """Start a fresh in-process agent runtime for one task."""
        if not self._started:
            self.start()
        self._cleanup_inactive_runtime()
        if self._agent_thread is not None and self._agent_thread.is_alive():
            self._agent_thread.join(timeout=0.2)
        if self._agent_thread is not None and self._agent_thread.is_alive():
            raise RuntimeError("YoloSandbox is already running.")

        host_transport, agent_transport = InProcessTransport.pair()
        self._host_transport = host_transport
        self._agent_transport = agent_transport
        self._thread_error = None
        self._host_service_server = None

        credentials = load_secrets()
        request_broker = RequestBroker(credentials=credentials, config=self._agent_config or {})
        host_services = HostServiceServer()
        host_services.register("broker", request_broker.handle)
        host_services.start()
        self._host_service_server = host_services
        broker_client = BrokerClient(host_services)

        agent = Agent(
            transport=agent_transport,
            skills_dir=self._skills_dir,
            agent_config=self._agent_config,
            max_iterations=self._max_iterations,
            output_dir=self._output_dir,
            token_budget=self._token_budget,
            summary_threshold=self._summary_threshold,
            llm_runtime=self._llm_runtime,
            broker=broker_client,
        )
        self._agent_thread = threading.Thread(target=self._run_agent, args=(agent,), daemon=True)
        self._agent_thread.start()
        task_event = dict(task)
        task_event.pop("llm", None)
        self.send(task_event)

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
        if self._host_transport is not None:
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
        if self._host_service_server is not None:
            self._host_service_server.stop()
        self._host_service_server = None
        self._started = False

    def _cleanup_inactive_runtime(self) -> None:
        if self._agent_thread is not None and self._agent_thread.is_alive():
            return
        if self._host_transport is not None:
            self._host_transport.close()
        if self._agent_transport is not None:
            self._agent_transport.close()
        self._host_transport = None
        self._agent_transport = None
        self._agent_thread = None
        if self._host_service_server is not None:
            self._host_service_server.stop()
        self._host_service_server = None

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
