"""Sequential in-sandbox subagent runner.

A `SubagentRunner` executes one child `Agent` to completion in the parent's own
thread and returns a single bounded result envelope. The child shares the
parent's `LLMRuntime` and `BrokerClient` (host-service plane) but runs over its
own private in-process transport (event/control plane), so child events never
reach adapters.

Because the child runs synchronously in the calling thread, it is always fully
finished before `run()` returns: there is no separate thread to join and no way
for a child to outlive the call and interleave a request on the shared broker.
The time budget is enforced by the child itself, which compares a monotonic
deadline (`task_timeout_seconds`) at each iteration boundary; on timeout it
returns without emitting a `done`, and the runner maps a missing `done` to a
`timeout` status.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from agent.agent import MAX_ITERATIONS_REPLY, Agent
from agent.broker_client import BrokerClient
from agent.llm_types import LLMRuntime
from agent.transport import InProcessTransport

_KNOWN_CAPABILITY_TOOLS = ("shell", "web_search", "web_fetch", "http_request")
_EVENTS_SUMMARY_CAP = 50


class SubagentRunner:
    """Runs one child agent at a time inside the parent's sandbox/session."""

    def __init__(
        self,
        *,
        llm_runtime: LLMRuntime,
        broker: BrokerClient | None,
        skills_dir: str,
        base_config: dict[str, Any],
        parent_enabled_tools: list[str],
        parent_session_id: str,
        output_root: str,
        limits: dict[str, Any],
    ) -> None:
        self._llm_runtime = llm_runtime
        self._broker = broker
        self._skills_dir = skills_dir
        self._base_config = base_config
        self._parent_enabled_tools = set(parent_enabled_tools)
        self._parent_session_id = parent_session_id
        self._output_root = Path(output_root)
        self._limits = limits

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run one child agent and return a bounded result envelope."""
        child_id = uuid.uuid4().hex[:12]
        child_output_dir = self._output_root / child_id
        child_output_dir.mkdir(parents=True, exist_ok=True)

        max_iterations = int(
            request.get("max_iterations") or self._limits.get("max_iterations", 20)
        )
        timeout_seconds = float(
            request.get("timeout_seconds") or self._limits.get("timeout_seconds", 600)
        )
        max_files_bytes = int(self._limits.get("max_files_bytes") or 10 * 1024 * 1024)

        host_transport, agent_transport = InProcessTransport.pair()
        child = Agent(
            transport=agent_transport,
            skills_dir=self._skills_dir,
            agent_config=self._build_child_config(request, max_iterations),
            max_iterations=max_iterations,
            output_dir=str(child_output_dir),
            max_output_total_bytes=max_files_bytes,
            llm_runtime=self._llm_runtime,
            broker=self._broker,
            clarify_enabled=False,
            task_timeout_seconds=timeout_seconds,
            subagent_runner=None,
        )

        task_event = {
            "type": "task",
            "text": self._compose_goal(request),
            "session_id": self._child_session_id(child_id),
            "approval_mode": "auto",
        }

        started = time.monotonic()
        try:
            host_transport.send(task_event)
            child.run()
            events = self._drain(host_transport)
        finally:
            host_transport.close()
            agent_transport.close()
        duration = time.monotonic() - started

        return self._build_envelope(child_id=child_id, events=events, duration=duration)

    def _build_child_config(self, request: dict[str, Any], max_iterations: int) -> dict[str, Any]:
        allowed = {
            tool
            for tool in request.get("allowed_tools", [])
            if tool in self._parent_enabled_tools
        }
        child_tools = {name: (name in allowed) for name in _KNOWN_CAPABILITY_TOOLS}
        child_tools["spawn_subagent"] = False

        config = dict(self._base_config)
        config["tools"] = child_tools
        config["loop"] = {"max_iterations": max_iterations}
        # Defense in depth: children never recurse, even if base config enabled it.
        subagents_raw = config.get("subagents")
        subagents = dict(subagents_raw) if isinstance(subagents_raw, dict) else {}
        subagents["enabled"] = False
        config["subagents"] = subagents
        return config

    def _compose_goal(self, request: dict[str, Any]) -> str:
        parts = [str(request.get("task", "")).strip()]
        context = str(request.get("context", "")).strip()
        if context:
            parts.append(f"Context:\n{context}")
        expected = str(request.get("expected_output", "")).strip()
        if expected:
            parts.append(f"Expected output:\n{expected}")
        referenced = request.get("referenced_skills") or []
        if isinstance(referenced, list) and referenced:
            parts.append("Suggested skills: " + ", ".join(str(name) for name in referenced))
        return "\n\n".join(part for part in parts if part)

    def _child_session_id(self, child_id: str) -> str:
        base = self._parent_session_id or "session"
        return f"{base}.sub.{child_id}"

    @staticmethod
    def _drain(host_transport: InProcessTransport) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            event = host_transport.receive(timeout_seconds=0)
            if event is None:
                break
            events.append(event)
        return events

    def _build_envelope(
        self,
        *,
        child_id: str,
        events: list[dict[str, Any]],
        duration: float,
    ) -> dict[str, Any]:
        done: dict[str, Any] | None = None
        for event in events:
            if event.get("type") == "done":
                done = event
        mode = str(self._limits.get("journal_events", "summary")).lower()
        if mode not in {"none", "summary", "full"}:
            mode = "summary"
        base: dict[str, Any] = {
            "child_id": child_id,
            "duration_seconds": round(duration, 3),
            "files": self._collect_child_files(child_id, done),
            "events_summary": self._summarize_events(events, mode),
        }

        if done is None:
            # The child loop returns without a done event only when it reaches its
            # time budget at an iteration boundary (Agent._run_task).
            return {
                **base,
                "success": False,
                "status": "timeout",
                "reply": "Subagent exceeded its time budget before finishing.",
                "state_summary": "",
            }
        if bool(done.get("success")):
            return {
                **base,
                "success": True,
                "status": "completed",
                "reply": str(done.get("reply", "")),
                "state_summary": self._state_summary(done),
            }
        status = "max_iterations" if done.get("reply") == MAX_ITERATIONS_REPLY else "child_failed"
        return {
            **base,
            "success": False,
            "status": status,
            "reply": str(done.get("reply", "")),
            "state_summary": self._state_summary(done),
        }

    @staticmethod
    def _state_summary(done: dict[str, Any]) -> str:
        state = done.get("state")
        if isinstance(state, dict):
            summary = state.get("summary")
            if isinstance(summary, str):
                return summary
        return ""

    @staticmethod
    def _summarize_events(events: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
        """Bounded child-event summary for the observation and journal.

        ``none`` omits per-event detail; ``summary`` lists tool calls and exit
        codes; ``full`` lists every captured event's type/role/tool (still without
        payloads, so it stays size-bounded). All modes are capped.
        """
        if mode == "none":
            return []
        summary: list[dict[str, Any]] = []
        for event in events:
            event_type = event.get("type")
            if mode == "summary":
                if event_type != "action":
                    continue
                result = event.get("result")
                exit_code = result.get("exit_code") if isinstance(result, dict) else None
                summary.append({"tool": event.get("tool"), "exit_code": exit_code})
                continue
            entry: dict[str, Any] = {"type": event_type}
            if event_type == "action":
                entry["tool"] = event.get("tool")
                result = event.get("result")
                if isinstance(result, dict):
                    entry["exit_code"] = result.get("exit_code")
            elif event_type == "message":
                entry["role"] = event.get("role")
            summary.append(entry)
        return summary[-_EVENTS_SUMMARY_CAP:]

    @staticmethod
    def _collect_child_files(
        child_id: str,
        done: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if done is None:
            return []
        raw_files = done.get("files")
        if not isinstance(raw_files, list):
            return []
        files: list[dict[str, Any]] = []
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            relative = str(item.get("path", ""))
            files.append(
                {
                    "path": f"subagents/{child_id}/{relative}",
                    "size_bytes": item.get("size_bytes", 0),
                    "content_b64": item.get("content_b64", ""),
                }
            )
        return files
