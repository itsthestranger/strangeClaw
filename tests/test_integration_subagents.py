"""Integration coverage for sequential subagents in Yolo and simulated Fire mode (C10.8).

These tests are deterministic: a scripted LLM serves the parent and child in
sequence, HTTP is mocked with ``responses``, and Fire mode is simulated in-process
(no real Firecracker) by wiring a fire-mode ``BrokerClient`` and ``LLMProxyRuntime``
to an in-process ``HostServiceServer`` through a synchronous bridge.
"""

from __future__ import annotations

import base64
import json
import queue
import shlex
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest
import responses

import agent.subagents as subagents_module
import session
from adapters.session_persistence import persist_done_event
from agent.agent import Agent
from agent.broker_client import BrokerClient
from agent.llm_proxy import LLMProxyRuntime
from agent.llm_types import LLMResponse, ToolCall
from agent.transport import InProcessTransport
from sandbox.broker import RequestBroker
from sandbox.host_services import HostServiceServer
from sandbox.llm_service import LLMService
from sandbox.yolo import YoloSandbox


class ScriptedLLM:
    """Deterministic LLM shared by parent and child, in call order."""

    def __init__(
        self,
        responses_list: list[LLMResponse],
        *,
        slow_call_index: int | None = None,
        slow_seconds: float = 0.0,
    ) -> None:
        self._responses = responses_list
        self.calls = 0
        self._slow_call_index = slow_call_index
        self._slow_seconds = slow_seconds

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self._slow_call_index is not None and self.calls == self._slow_call_index:
            time.sleep(self._slow_seconds)
        self.calls += 1
        if not self._responses:
            raise AssertionError("ScriptedLLM exhausted responses.")
        return self._responses.pop(0)

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        return 1


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def _plan(step: str = "step") -> LLMResponse:
    return LLMResponse(
        text=json.dumps({"goal": "g", "steps": [step], "referenced_skills": []}), action=None
    )


def _spawn(task: str, allowed_tools: list[str]) -> LLMResponse:
    return LLMResponse(
        text="",
        action=ToolCall(tool="spawn_subagent", args={"task": task, "allowed_tools": allowed_tools}),
    )


def _tool(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(text="", action=ToolCall(tool=name, args=args))


def _http(integration: str, method: str, url: str, body: str | None = None) -> LLMResponse:
    return _tool(
        "http_request",
        {"integration": integration, "method": method, "url": url, "headers": {}, "body": body},
    )


def _done(reply: str) -> LLMResponse:
    return _tool("agent_done", {"reply": reply})


def _task() -> dict[str, Any]:
    return {"type": "task", "text": "parent goal", "session_id": "sess-1", "approval_mode": "auto"}


def _unwrap(text: str) -> dict[str, Any]:
    prefix, suffix = "--- BEGIN DATA ---\n", "\n--- END DATA ---"
    assert text.startswith(prefix) and text.endswith(suffix)
    parsed = json.loads(text[len(prefix) : -len(suffix)])
    assert isinstance(parsed, dict)
    return parsed


def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )


def _subagent_config() -> dict[str, Any]:
    return {
        "llm": {"model": "fake/model", "api_key": "fake-key"},
        "tools": {
            "shell": True,
            "web_search": True,
            "web_fetch": True,
            "http_request": True,
            "spawn_subagent": True,
        },
        "subagents": {
            "enabled": True,
            "max_children_per_task": 3,
            "max_iterations": 20,
            "timeout_seconds": 600,
            "max_context_chars": 20000,
            "max_result_chars": 20000,
            "max_files_bytes": 10 * 1024 * 1024,
            "journal_events": "summary",
        },
    }


def _notion_credentials(token: str) -> dict[str, Any]:
    return {
        "notion": {
            "auth_type": "bearer",
            "token": token,
            "allowed_hosts": ["api.notion.com"],
            "allowed_methods": ["POST"],
            "allowed_paths": ["/v1/*"],
            "allowed_schemes": ["https"],
            "protected_headers": ["Authorization"],
            "default_headers": {"Notion-Version": "2022-06-28"},
            "max_response_bytes": 4096,
            "rate_limit": None,
        }
    }


def _yolo_sandbox(
    llm: ScriptedLLM, tmp_path: Path, config: dict[str, Any] | None = None
) -> YoloSandbox:
    return YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=config or _subagent_config(),
        output_dir=str(tmp_path / "output"),
    )


def _collect_yolo(sandbox: YoloSandbox) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while True:
        event = sandbox.receive(timeout_seconds=3.0)
        assert event is not None, "Timed out waiting for sandbox event."
        events.append(event)
        if event["type"] == "done":
            return events


# --------------------------------------------------------------------------- Yolo


def test_yolo_subagent_delegation_and_no_event_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: {})
    llm = ScriptedLLM(
        [
            _plan(),  # parent plan
            _spawn("research the thing", ["shell"]),  # parent delegates
            _plan(),  # child plan
            _tool("shell", {"command": "echo found"}),  # child works
            _done("child: found it"),  # child finishes
            _done("parent: summarized"),  # parent finishes
        ]
    )
    sandbox = _yolo_sandbox(llm, tmp_path)
    sandbox.run(_task())
    try:
        events = _collect_yolo(sandbox)
    finally:
        sandbox.stop()

    actions = [e for e in events if e["type"] == "action"]
    plans = [e for e in events if e["type"] == "message" and e.get("role") == "plan"]
    # Only the parent's plan and the single spawn_subagent action are visible.
    assert len(plans) == 1
    assert [a["tool"] for a in actions] == ["spawn_subagent"]
    assert events[-1]["reply"] == "parent: summarized"
    envelope = _unwrap(actions[0]["result"]["stdout"])
    assert envelope["status"] == "completed"
    assert envelope["reply"] == "child: found it"


def test_yolo_subagent_denied_tool_is_observed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: {})
    llm = ScriptedLLM(
        [
            _plan(),
            _spawn("research", ["shell"]),  # child only gets shell
            _plan(),
            _tool("web_search", {"query": "x"}),  # not delegated -> denied
            _done("child carried on"),
            _done("parent done"),
        ]
    )
    sandbox = _yolo_sandbox(llm, tmp_path)
    sandbox.run(_task())
    try:
        events = _collect_yolo(sandbox)
    finally:
        sandbox.stop()

    spawn_action = next(e for e in events if e.get("tool") == "spawn_subagent")
    envelope = _unwrap(spawn_action["result"]["stdout"])
    assert envelope["status"] == "completed"
    denied = [(e["tool"], e["exit_code"]) for e in envelope["events_summary"]]
    assert ("web_search", 1) in denied


def test_yolo_subagent_clarify_does_not_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: {})
    llm = ScriptedLLM(
        [
            _plan(),
            _spawn("research", ["shell"]),
            _plan(),
            _tool("agent_clarify", {"question": "which one?"}),  # child cannot ask
            _done("child decided itself"),
            _done("parent done"),
        ]
    )
    sandbox = _yolo_sandbox(llm, tmp_path)
    sandbox.run(_task())
    try:
        events = _collect_yolo(sandbox)
    finally:
        sandbox.stop()

    # No clarification ever reaches the adapter; the parent completes normally.
    assert not [e for e in events if e["type"] == "message" and e.get("role") == "clarification"]
    assert events[-1]["reply"] == "parent done"


def test_yolo_subagent_timeout_then_parent_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: {})
    # Child planning (call index 2) sleeps past the per-call timeout budget.
    llm = ScriptedLLM(
        [
            _plan(),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="spawn_subagent",
                    args={"task": "slow", "allowed_tools": ["shell"], "timeout_seconds": 1},
                ),
            ),
            _plan(),  # child plan (slow)
            _done("parent recovered"),
        ],
        slow_call_index=2,
        slow_seconds=1.2,
    )
    sandbox = _yolo_sandbox(llm, tmp_path)
    sandbox.run(_task())
    try:
        events = _collect_yolo(sandbox)
    finally:
        sandbox.stop()

    spawn_action = next(e for e in events if e.get("tool") == "spawn_subagent")
    envelope = _unwrap(spawn_action["result"]["stdout"])
    assert envelope["status"] == "timeout"
    # The parent resumes cleanly after the (fully joined) child timed out.
    assert events[-1]["reply"] == "parent recovered"


@responses.activate
def test_yolo_subagent_credential_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _public_dns(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    token = "yolo-notion-secret"
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: _notion_credentials(token))
    responses.add(responses.POST, "https://api.notion.com/v1/pages", json={"ok": True}, status=201)

    llm = ScriptedLLM(
        [
            _plan(),
            _spawn("create a page", ["http_request"]),
            _plan(),
            _http("notion", "POST", "https://api.notion.com/v1/pages", '{"title":"x"}'),
            _done("child created the page"),
            _done("parent done"),
        ]
    )
    sandbox = _yolo_sandbox(llm, tmp_path)
    sandbox.run(_task())
    try:
        events = _collect_yolo(sandbox)
    finally:
        sandbox.stop()

    # The broker injected the credential host-side...
    assert responses.calls[0].request.headers["Authorization"] == f"Bearer {token}"
    # ...but it never appears in any adapter-facing event...
    assert token not in json.dumps(events)
    # ...nor in persisted parent state.
    persist_done_event(session_id="sess-1", done_event=events[-1])
    loaded = session.load(tmp_path / ".strangeclaw" / "sessions" / "sess-1")
    assert loaded is not None
    assert token not in json.dumps(loaded)


# --------------------------------------------------------------------------- Fire (simulated)


class _FireHostBridge:
    """Synchronous in-process stand-in for the FireSandbox vsock dispatch loop."""

    def __init__(self, server: HostServiceServer) -> None:
        self._server = server
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()

    def send(self, event: dict[str, Any]) -> None:
        if event.get("type") == "broker_request":
            self._responses.put(self._server.handle_incoming(event))

    def receive(self, timeout: float | None) -> dict[str, Any] | None:
        try:
            return self._responses.get(timeout=timeout if timeout and timeout > 0 else 0.05)
        except queue.Empty:
            return None


def _fire_guest_config() -> dict[str, Any]:
    config = _subagent_config()
    config.pop("llm")  # the Fire guest holds no LLM config
    config["host_services"] = {"llm_timeout_seconds": 120, "llm_max_request_bytes": 2 * 1024 * 1024}
    config["web_search"] = {
        "endpoint": "https://api.search.brave.com/res/v1/web/search",
        "format": "brave",
        "max_results": 10,
    }
    config["web_fetch"] = {"max_response_bytes": 524288}
    return config


def _fire_host_config(llm_api_key: str) -> dict[str, Any]:
    return {
        "llm": {"model": "fake/model", "api_key": llm_api_key},
        "host_services": {"llm_timeout_seconds": 120, "llm_max_request_bytes": 2 * 1024 * 1024},
        "web_search": {
            "endpoint": "https://api.search.brave.com/res/v1/web/search",
            "format": "brave",
            "max_results": 10,
        },
        "web_fetch": {"max_response_bytes": 524288},
        "broker": {
            "public_policy": {
                "enabled": True,
                "allowed_methods": ["GET"],
                "max_response_bytes": 524288,
            }
        },
    }


def _build_fire_agent(
    llm: ScriptedLLM,
    *,
    output_dir: Path,
    credentials: dict[str, Any],
    host_config: dict[str, Any],
) -> tuple[Agent, InProcessTransport]:
    server = HostServiceServer()
    server.register("broker", RequestBroker(credentials=credentials, config=host_config).handle)
    server.register("llm", LLMService(host_config, llm_client=llm).handle)
    server.start()
    bridge = _FireHostBridge(server)
    client = BrokerClient(
        mode="fire",
        send_fn=bridge.send,
        receive_fn=bridge.receive,
        service_timeouts={"_default": 5.0, "broker": 5.0, "llm": 5.0},
    )
    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        agent_config=_fire_guest_config(),
        output_dir=str(output_dir),
        llm_runtime=LLMProxyRuntime(client),
        broker=client,
    )
    return agent, host_transport


def _run_fire(agent: Agent, host_transport: InProcessTransport) -> list[dict[str, Any]]:
    thread = threading.Thread(target=agent.run, daemon=True)
    thread.start()
    host_transport.send(_task())
    events: list[dict[str, Any]] = []
    while True:
        event = host_transport.receive(timeout_seconds=5.0)
        assert event is not None, "Timed out waiting for Fire guest event."
        events.append(event)
        if event["type"] == "done":
            break
    thread.join(timeout=2.0)
    return events


@responses.activate
def test_fire_subagent_credential_isolation_no_broker_event_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _public_dns(monkeypatch)
    token = "fire-notion-secret"
    llm_key = "FIRE-LLM-API-KEY"
    responses.add(responses.POST, "https://api.notion.com/v1/pages", json={"ok": True}, status=201)

    llm = ScriptedLLM(
        [
            _plan(),
            _spawn("create a page", ["http_request"]),
            _plan(),
            _http("notion", "POST", "https://api.notion.com/v1/pages", '{"title":"x"}'),
            _done("child created the page"),
            _done("parent done"),
        ]
    )
    agent, host_transport = _build_fire_agent(
        llm,
        output_dir=tmp_path / "output",
        credentials=_notion_credentials(token),
        host_config=_fire_host_config(llm_key),
    )
    events = _run_fire(agent, host_transport)

    # Broker injected the credential through the host-service (proxy) path.
    assert responses.calls[0].request.headers["Authorization"] == f"Bearer {token}"
    # broker_request/broker_response rode the bridge, never the guest event stream.
    assert all(e["type"] not in {"broker_request", "broker_response"} for e in events)
    actions = [e for e in events if e["type"] == "action"]
    assert [a["tool"] for a in actions] == ["spawn_subagent"]
    rendered = json.dumps(events)
    assert token not in rendered
    assert llm_key not in rendered


def test_fire_subagent_reads_session_file_and_exports_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    child_id = "0" * 12
    monkeypatch.setattr(
        subagents_module.uuid,
        "uuid4",
        lambda: uuid.UUID("00000000-0000-4000-8000-000000000000"),
    )
    earlier = tmp_path / "earlier.txt"
    earlier.write_text("session-data", encoding="utf-8")
    output_dir = tmp_path / "output"
    child_out = output_dir / "subagents" / child_id / "result.txt"

    llm = ScriptedLLM(
        [
            _plan(),
            _spawn("process the earlier file", ["shell"]),
            _plan(),
            _tool(
                "shell",
                {"command": f"cat {shlex.quote(str(earlier))} > {shlex.quote(str(child_out))}"},
            ),
            _done("child processed the file"),
            _done("parent done"),
        ]
    )
    agent, host_transport = _build_fire_agent(
        llm, output_dir=output_dir, credentials={}, host_config=_fire_host_config("k")
    )
    events = _run_fire(agent, host_transport)

    spawn_action = next(e for e in events if e.get("tool") == "spawn_subagent")
    envelope = _unwrap(spawn_action["result"]["stdout"])
    assert envelope["status"] == "completed"
    assert f"subagents/{child_id}/result.txt" in [f["path"] for f in envelope["files"]]

    # The parent's own /output export also collects the child artifact, and it
    # contains the earlier session file's contents (proving shared-FS access).
    done = events[-1]
    artifact = next(f for f in done["files"] if f["path"] == f"subagents/{child_id}/result.txt")
    assert base64.b64decode(artifact["content_b64"]).decode("utf-8") == "session-data"
