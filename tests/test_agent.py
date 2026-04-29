"""Tests for agent core loop."""

from __future__ import annotations

import base64
import json
import shlex
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import agent.agent as agent_module
from agent.agent import Agent
from agent.llm import LLMResponse, ToolCall
from agent.transport import InProcessTransport


class ScriptedLLM:
    """Deterministic fake LLM for agent-loop tests."""

    def __init__(
        self,
        responses: list[LLMResponse],
        *,
        count_tokens_func: Callable[[list[dict[str, Any]]], int] | None = None,
    ) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []
        self._count_tokens_func = count_tokens_func

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "action_schema": action_schema})
        if not self._responses:
            raise AssertionError("ScriptedLLM exhausted responses.")
        return self._responses.pop(0)

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        if self._count_tokens_func is not None:
            return int(self._count_tokens_func(messages))
        del messages
        return 1


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def _task_event(approval_mode: str = "auto") -> dict[str, Any]:
    return {
        "type": "task",
        "text": "check Python version and write hello world script",
        "session_id": "sess-1",
        "approval_mode": approval_mode,
        "llm": {"model": "fake/model", "api_key": "fake-key"},
    }


def _collect_until_done(
    host_transport: InProcessTransport,
    *,
    review_replies: list[dict[str, Any]] | None = None,
    clarification_reply: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    replies = review_replies[:] if review_replies else []

    while True:
        event = host_transport.receive(timeout_seconds=2.0)
        assert event is not None, "Timed out waiting for agent event."
        events.append(event)

        if event["type"] == "message" and event["role"] == "plan" and replies:
            host_transport.send(replies.pop(0))
            continue
        if event["type"] == "message" and event["role"] == "clarification" and clarification_reply:
            host_transport.send(clarification_reply)
            clarification_reply = None
            continue
        if event["type"] == "done":
            return events


def test_agent_completes_multi_step_task_end_to_end(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    hello_path = tmp_path / "hello.py"
    quoted_hello = shlex.quote(str(hello_path))
    quoted_output = shlex.quote(str(output_dir / "artifact.txt"))

    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(
                text=(
                    '{"goal":"check Python version and write script",'
                    '"steps":["check version","write file"]}'
                ),
                action=None,
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell", args={"command": "python3 --version"}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell",
                    args={"command": f'printf \'print("hello")\\n\' > {quoted_hello}'},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell",
                    args={"command": f"printf artifact > {quoted_output}"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="__agent__.done",
                    args={"reply": "Completed successfully."},
                ),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=10,
        output_dir=str(output_dir),
        llm_factory=lambda _: scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    assert events[0]["type"] == "message"
    assert events[0]["role"] == "plan"
    action_events = [event for event in events if event["type"] == "action"]
    assert len(action_events) == 3
    assert action_events[0]["tool"] == "shell"
    assert hello_path.read_text(encoding="utf-8") == 'print("hello")\n'

    done_event = events[-1]
    assert done_event["type"] == "done"
    assert done_event["success"] is True
    assert done_event["reply"] == "Completed successfully."

    files = done_event["files"]
    assert isinstance(files, list)
    artifact_entry = next(item for item in files if item["path"] == "artifact.txt")
    decoded = base64.b64decode(artifact_entry["content_b64"]).decode("utf-8")
    assert decoded == "artifact"


def test_agent_plan_rejection_replans_in_review_mode() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["old plan"]}', action=None, usage=None),
            LLMResponse(text='{"steps":["updated plan"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="__agent__.done",
                    args={"reply": "Done after replan."},
                ),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        llm_factory=lambda _: scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event(approval_mode="review"))

    events = _collect_until_done(
        host_transport,
        review_replies=[
            {"type": "user_reply", "text": "Please update the plan.", "approved": False},
            {"type": "user_reply", "text": "Approved.", "approved": True},
        ],
    )
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    plan_events = [
        event for event in events if event["type"] == "message" and event["role"] == "plan"
    ]
    assert len(plan_events) == 2
    done_event = events[-1]
    assert done_event["type"] == "done"
    assert done_event["reply"] == "Done after replan."


def test_agent_emits_clarification_when_max_iterations_reached() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["single"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell", args={"command": "printf one-step"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=1,
        llm_factory=lambda _: scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    clarification_events = [
        event for event in events if event["type"] == "message" and event["role"] == "clarification"
    ]
    assert clarification_events
    done_event = events[-1]
    assert done_event["type"] == "done"
    assert done_event["success"] is False
    assert "iteration limit" in done_event["reply"].lower()


def test_build_execution_prompt_drops_oldest_history_when_over_budget() -> None:
    def count_tokens(messages: list[dict[str, Any]]) -> int:
        payload = json.loads(messages[1]["content"])
        recent_count = len(payload["recent_history"])
        summary_len = len(payload["history_summary"])
        return 100 + (recent_count * 100) + summary_len

    scripted_llm = ScriptedLLM(responses=[], count_tokens_func=count_tokens)
    host_transport, agent_transport = InProcessTransport.pair()
    del host_transport
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        token_budget=220,
        summary_threshold=50,
        llm_factory=lambda _: scripted_llm,
    )
    agent._llm = scripted_llm

    history = [
        {"type": "action", "idx": 1},
        {"type": "action", "idx": 2},
        {"type": "action", "idx": 3},
    ]
    messages = agent.build_execution_prompt(
        goal="g",
        plan={"steps": []},
        history=history,
        activated_skills={},
    )
    payload = json.loads(messages[1]["content"])
    assert payload["recent_history"] == [{"type": "action", "idx": 3}]
    assert payload["enabled_tools"] == ["http_request", "shell", "web_fetch", "web_search"]
    assert isinstance(payload["tool_schemas"], list)
    assert payload["activated_skills"] == {}
    assert payload["output_instruction"] == "Place any files for the user in /output/."


def test_agent_done_reports_error_when_output_limit_exceeded(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "too-big.txt").write_bytes(b"0123456789abcdef")

    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["single"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="__agent__.done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        output_dir=str(output_dir),
        max_output_total_bytes=8,
        llm_factory=lambda _: scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    done_event = events[-1]
    assert done_event["type"] == "done"
    assert done_event["success"] is False
    assert "Output export error" in done_event["reply"]
    assert "exceeds output limit" in done_event["reply"]
    assert done_event["files"] == []


def test_recent_history_triggers_summary_and_done_state_persists_it() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["s1","s2"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell", args={"command": "printf step1"}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell", args={"command": "printf step2"}),
                usage=None,
            ),
            LLMResponse(text="summarized previous observations", action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="__agent__.done",
                    args={"reply": "done with summary"},
                ),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        summary_threshold=1,
        llm_factory=lambda _: scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    done_event = events[-1]
    assert done_event["type"] == "done"
    assert done_event["state"]["summary"] == "summarized previous observations"


def test_agent_handles_invalid_decision_output_without_crashing() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["attempt"]}', action=None, usage=None),
            LLMResponse(text="not json", action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="__agent__.done", args={"reply": "recovered"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        llm_factory=lambda _: scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    error_actions = [
        event
        for event in events
        if event["type"] == "action" and event["tool"] == "__agent__.decision_error"
    ]
    assert error_actions
    assert error_actions[0]["result"]["exit_code"] == 1
    assert "Decision parse error" in error_actions[0]["result"]["stderr"]
    assert events[-1]["type"] == "done"
    assert events[-1]["success"] is True
    assert events[-1]["reply"] == "recovered"


def test_agent_uses_agent_config_file_when_task_llm_is_disallowed(tmp_path: Path) -> None:
    llm_from_file = {"model": "file/model", "api_key": "file-key"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"llm": llm_from_file}), encoding="utf-8")

    captured: dict[str, Any] = {}
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["single"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="__agent__.done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )

    def llm_factory(config: dict[str, Any]) -> ScriptedLLM:
        captured["config"] = config
        return scripted_llm

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        agent_config_path=str(config_path),
        allow_task_llm=False,
        llm_factory=llm_factory,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()

    task = _task_event()
    task["llm"] = {"model": "task/model", "api_key": "task-key"}
    host_transport.send(task)

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    assert captured["config"] == llm_from_file
    assert events[-1]["type"] == "done"
    assert events[-1]["success"] is True
    event_dump = json.dumps(events, ensure_ascii=True)
    assert "file-key" not in event_dump
    assert "/run/strangeclaw/config.json" not in event_dump


def test_agent_stage3_read_skill_file_control_action() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(
                text='{"goal":"g","steps":["read"],"referenced_skills":["shell"]}',
                action=None,
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="__agent__.read_skill_file",
                    args={"skill": "shell", "path": "SKILL.md"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="__agent__.done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        llm_factory=lambda _: scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    read_events = [
        event
        for event in events
        if event.get("type") == "action"
        and event.get("tool") == "__agent__.read_skill_file"
    ]
    assert read_events
    assert read_events[0]["result"]["exit_code"] == 0
    assert "shell" in read_events[0]["result"]["stdout"].lower()


def test_agent_replans_when_plan_references_unknown_skill() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(
                text='{"goal":"g","steps":["bad"],"referenced_skills":["missing-skill"]}',
                action=None,
                usage=None,
            ),
            LLMResponse(
                text='{"goal":"g","steps":["good"],"referenced_skills":[]}',
                action=None,
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="__agent__.done", args={"reply": "ok"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        llm_factory=lambda _: scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    plan_events = [
        event for event in events if event.get("type") == "message" and event.get("role") == "plan"
    ]
    assert len(plan_events) == 2
    status_events = [
        event
        for event in events
        if event.get("type") == "message" and event.get("role") == "status"
    ]
    assert any("Unknown referenced skill" in str(event.get("content")) for event in status_events)


def test_agent_main_requires_vsock_port() -> None:
    with pytest.raises(ValueError, match="--vsock-port"):
        agent_module.main([])


def test_agent_main_vsock_entrypoint_wires_transport_and_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    class FakeVsockTransport:
        def __init__(self, *, guest_port: int) -> None:
            captured["guest_port"] = guest_port
            captured["closed"] = False

        def close(self) -> None:
            captured["closed"] = True

        def send(self, event: dict[str, Any]) -> None:
            del event

        def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None:
            del timeout_seconds
            return None

    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            captured["agent_kwargs"] = kwargs

        def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr(agent_module, "VsockTransport", FakeVsockTransport)
    monkeypatch.setattr(agent_module, "Agent", FakeAgent)

    agent_module.main(
        [
            "--vsock-port",
            "5000",
            "--skills-dir",
            str(tmp_path / "skills"),
            "--agent-config-path",
            str(tmp_path / "config.json"),
        ]
    )

    assert captured["guest_port"] == 5000
    assert captured["ran"] is True
    assert captured["closed"] is True

    kwargs = captured["agent_kwargs"]
    assert kwargs["allow_task_llm"] is False
    assert kwargs["skills_dir"] == str(tmp_path / "skills")
    assert kwargs["agent_config_path"] == str(tmp_path / "config.json")
