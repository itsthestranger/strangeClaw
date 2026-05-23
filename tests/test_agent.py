"""Tests for agent core loop."""

from __future__ import annotations

import base64
import json
import re
import shlex
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import agent.agent as agent_module
from agent.agent import Agent
from agent.broker_client import BrokerClient
from agent.llm import LLMClient
from agent.llm_types import LLMResponse, ToolCall
from agent.transport import InProcessTransport
from sandbox.host_services import HostServiceServer


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
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
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


def _build_temp_skill(skills_root: Path, *, name: str = "demo") -> None:
    skill_dir = skills_root / name
    references_dir = skill_dir / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            "description: Demo skill for tests\n"
            "---\n\n"
            "Use this skill in tests.\n"
        ),
        encoding="utf-8",
    )
    (references_dir / "notes.md").write_text("skill-reference-content\n", encoding="utf-8")


def _task_event(approval_mode: str = "auto") -> dict[str, Any]:
    return {
        "type": "task",
        "text": "check Python version and write hello world script",
        "session_id": "sess-1",
        "approval_mode": approval_mode,
        "llm": {"model": "fake/model", "api_key": "fake-key"},
    }


def _agent_config() -> dict[str, Any]:
    return {"llm": {"model": "fake/model", "api_key": "fake-key"}}


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


def test_agent_constructs_llm_client_once_and_reuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_configs: list[dict[str, Any]] = []
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["first"]}', action=None, usage=None),
            LLMResponse(text="", action=ToolCall(tool="agent_done", args={"reply": "one"})),
            LLMResponse(text='{"steps":["second"]}', action=None, usage=None),
            LLMResponse(text="", action=ToolCall(tool="agent_done", args={"reply": "two"})),
        ]
    )

    def fake_from_config(config: dict[str, Any]) -> ScriptedLLM:
        created_configs.append(dict(config))
        return scripted_llm

    monkeypatch.setattr(agent_module.LLMClient, "from_config", staticmethod(fake_from_config))

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        agent_config=_agent_config(),
    )

    worker = threading.Thread(target=agent.run_forever)
    worker.start()
    host_transport.send({**_task_event(), "text": "first"})
    first = _collect_until_done(host_transport)
    host_transport.send({**_task_event(), "text": "second"})
    second = _collect_until_done(host_transport)
    host_transport.send({"type": "stop"})
    worker.join(timeout=2.0)

    assert created_configs == [_agent_config()["llm"]]
    assert first[-1]["reply"] == "one"
    assert second[-1]["reply"] == "two"


def test_agent_uses_supplied_llm_runtime_without_constructing_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module.LLMClient,
        "from_config",
        staticmethod(lambda config: (_ for _ in ()).throw(AssertionError(config))),
    )
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["single"]}', action=None, usage=None),
            LLMResponse(text="", action=ToolCall(tool="agent_done", args={"reply": "done"})),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        agent_config=_agent_config(),
        llm_runtime=scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())
    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)

    assert events[-1]["type"] == "done"
    assert events[-1]["reply"] == "done"


def test_agent_loads_config_file_when_llm_runtime_is_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_module.LLMClient,
        "from_config",
        staticmethod(lambda config: (_ for _ in ()).throw(AssertionError(config))),
    )
    skills_root = tmp_path / "skills"
    _build_temp_skill(skills_root, name="runtime-config-skill")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "llm": {"model": "file/model", "api_key": "file-key"},
                "tools": {
                    "shell": False,
                    "web_search": False,
                    "web_fetch": False,
                    "http_request": False,
                },
                "skills": {
                    "directory": str(skills_root),
                    "max_file_chars": 20000,
                },
                "loop": {"max_iterations": 3},
                "context": {
                    "token_budget": 4000,
                    "summary_threshold": 10,
                    "max_output_chars": 8000,
                },
            }
        ),
        encoding="utf-8",
    )
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["single"]}', action=None),
            LLMResponse(text="", action=ToolCall(tool="agent_done", args={"reply": "done"})),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(tmp_path / "unused-skills"),
        agent_config_path=str(config_path),
        llm_runtime=scripted_llm,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())
    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)

    assert events[-1]["reply"] == "done"
    planning_payload = scripted_llm.calls[0]["messages"][1]["content"]
    assert "runtime-config-skill" in planning_payload
    execution_payload = json.loads(scripted_llm.calls[1]["messages"][1]["content"])
    assert execution_payload["enabled_tools"] == []



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
                action=ToolCall(tool="agent_done",
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
        llm_runtime=scripted_llm,
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


def test_agent_run_forever_processes_sequential_tasks_with_clean_state(
    tmp_path: Path,
) -> None:
    shared_file = tmp_path / "persistent.txt"
    quoted_shared_file = shlex.quote(str(shared_file))
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["write file"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="shell",
                    args={"command": f"printf hello > {quoted_shared_file}"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "first done"}),
                usage=None,
            ),
            LLMResponse(text='{"steps":["read file"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="shell",
                    args={"command": f"cat {quoted_shared_file}"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "second done"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        llm_runtime=scripted_llm,
    )

    worker = threading.Thread(target=agent.run_forever)
    worker.start()
    host_transport.send({**_task_event(), "text": "write shared file"})
    first_events = _collect_until_done(host_transport)
    assert worker.is_alive()
    assert first_events[-1]["reply"] == "first done"

    host_transport.send({**_task_event(), "text": "read shared file"})
    second_events = _collect_until_done(host_transport)
    host_transport.send({"type": "stop"})
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    second_shell = next(
        event
        for event in second_events
        if event.get("type") == "action" and event["tool"] == "shell"
    )
    assert second_shell["result"]["stdout"] == "hello"
    assert second_events[-1]["reply"] == "second done"

    second_task_first_execution = scripted_llm.calls[4]
    payload = json.loads(second_task_first_execution["messages"][1]["content"])
    assert payload["recent_history"] == []
    assert payload["history_summary"] == ""


def test_agent_run_forever_continues_after_max_iteration_done() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["too short"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell", args={"command": "printf first"}),
                usage=None,
            ),
            LLMResponse(text='{"steps":["recover"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "recovered"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=1,
        llm_runtime=scripted_llm,
    )

    worker = threading.Thread(target=agent.run_forever)
    worker.start()
    host_transport.send({**_task_event(), "text": "hit max iteration"})
    first_events = _collect_until_done(host_transport)
    assert first_events[-1]["success"] is False
    assert "iteration limit" in first_events[-1]["reply"].lower()
    assert worker.is_alive()

    host_transport.send({**_task_event(), "text": "second task"})
    second_events = _collect_until_done(host_transport)
    host_transport.send({"type": "stop"})
    worker.join(timeout=2.0)

    assert not worker.is_alive()
    assert second_events[-1]["success"] is True
    assert second_events[-1]["reply"] == "recovered"


def test_agent_run_forever_stop_during_idle_exits() -> None:
    scripted_llm = ScriptedLLM(responses=[])
    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        llm_runtime=scripted_llm,
    )

    worker = threading.Thread(target=agent.run_forever)
    worker.start()
    host_transport.send({"type": "stop"})
    worker.join(timeout=2.0)

    assert not worker.is_alive()


def test_agent_plan_rejection_replans_in_review_mode() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["old plan"]}', action=None, usage=None),
            LLMResponse(text='{"steps":["updated plan"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done",
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
        llm_runtime=scripted_llm,
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
        llm_runtime=scripted_llm,
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
        llm_runtime=scripted_llm,
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
    assert "Configured integrations: none" in messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    assert payload["recent_history"] == [{"type": "action", "idx": 3}]
    assert payload["enabled_tools"] == ["http_request", "shell", "web_fetch", "web_search"]
    assert isinstance(payload["tool_schemas"], list)
    tool_names = [entry.get("name") for entry in payload["tool_schemas"] if isinstance(entry, dict)]
    safe_name_pattern = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
    assert "agent_done" in tool_names
    assert "agent_clarify" in tool_names
    assert "agent_replan" in tool_names
    assert "agent_read_skill_file" in tool_names
    assert all(
        isinstance(name, str) and safe_name_pattern.fullmatch(name) for name in tool_names
    )
    assert payload["activated_skills"] == {}
    assert payload["output_instruction"] == "Place any files for the user in /output/."


def test_agent_loads_integrations_via_broker_when_task_starts() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["single"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )
    calls: list[dict[str, Any]] = []

    def _broker_handler(payload: dict[str, Any]) -> dict[str, Any]:
        calls.append(dict(payload))
        return {"integrations": ["github", "notion"]}

    server = HostServiceServer()
    server.register("broker", _broker_handler)
    broker = BrokerClient(server)
    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        llm_runtime=scripted_llm,
        broker=broker,
    )

    assert calls == []
    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())
    _collect_until_done(host_transport)
    worker.join(timeout=2.0)

    assert calls == [{"action": "list_integrations"}]
    assert agent._integrations == ["github", "notion"]


def test_execution_prompt_includes_configured_integrations() -> None:
    scripted_llm = ScriptedLLM(responses=[])
    server = HostServiceServer()
    server.register("broker", lambda payload: {"integrations": ["notion"]})
    broker = BrokerClient(server)
    host_transport, agent_transport = InProcessTransport.pair()
    del host_transport
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        llm_runtime=scripted_llm,
        broker=broker,
    )
    agent._llm = scripted_llm
    agent._integrations = agent._load_integrations()

    messages = agent.build_execution_prompt(
        goal="g",
        plan={"steps": []},
        history=[],
        activated_skills={},
    )

    assert "Configured integrations: notion" in messages[0]["content"]


def test_agent_done_reports_error_when_output_limit_exceeded(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "too-big.txt").write_bytes(b"0123456789abcdef")

    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["single"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "done"}),
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
        llm_runtime=scripted_llm,
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
                action=ToolCall(tool="agent_done",
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
        llm_runtime=scripted_llm,
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
                action=ToolCall(tool="agent_done", args={"reply": "recovered"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        llm_runtime=scripted_llm,
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
        if event["type"] == "action" and event["tool"] == "agent_decision_error"
    ]
    assert error_actions
    assert error_actions[0]["result"]["exit_code"] == 1
    assert "Decision parse error" in error_actions[0]["result"]["stderr"]
    assert events[-1]["type"] == "done"
    assert events[-1]["success"] is True
    assert events[-1]["reply"] == "recovered"


@pytest.mark.parametrize(
    ("bad_tool", "bad_args", "error_fragment"),
    [
        ("agent_done", {}, "requires args.reply"),
        ("agent_clarify", {"question": 123}, "args.question must be a string"),
        ("agent_replan", {"feedback": 123}, "args.feedback must be a string"),
        ("agent_read_skill_file", {"skill": "shell"}, "requires args.skill and args.path"),
    ],
)
def test_agent_malformed_control_call_emits_action_error_and_recovers(
    bad_tool: str,
    bad_args: dict[str, Any],
    error_fragment: str,
) -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["attempt malformed control"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool=bad_tool, args=bad_args),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "recovered"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        llm_runtime=scripted_llm,
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
        if event.get("type") == "action"
        and event.get("tool") == bad_tool
        and event.get("result", {}).get("exit_code") == 1
    ]
    assert error_actions
    assert error_fragment in str(error_actions[0]["result"]["stderr"])
    assert events[-1]["type"] == "done"
    assert events[-1]["success"] is True
    assert events[-1]["reply"] == "recovered"

    # The malformed-control action error must be observed for model self-correction.
    third_call_payload = json.loads(scripted_llm.calls[2]["messages"][1]["content"])
    recent_history = third_call_payload["recent_history"]
    assert any(
        isinstance(item, dict)
        and item.get("type") == "action"
        and item.get("tool") == bad_tool
        and item.get("result", {}).get("exit_code") == 1
        for item in recent_history
    )


def test_agent_uses_agent_config_file_and_ignores_task_llm(tmp_path: Path) -> None:
    llm_from_file = {"model": "file/model", "api_key": "file-key"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"llm": llm_from_file}), encoding="utf-8")

    captured: dict[str, Any] = {}
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["single"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )

    def fake_from_config(config: dict[str, Any]) -> ScriptedLLM:
        captured["config"] = config
        return scripted_llm

    host_transport, agent_transport = InProcessTransport.pair()
    original_from_config = agent_module.LLMClient.from_config
    agent_module.LLMClient.from_config = fake_from_config  # type: ignore[assignment]
    try:
        agent = Agent(
            transport=agent_transport,
            skills_dir=str(_skills_root()),
            agent_config_path=str(config_path),
        )
    finally:
        agent_module.LLMClient.from_config = original_from_config  # type: ignore[assignment]

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
                text='{"goal":"g","steps":["read"],"referenced_skills":[]}',
                action=None,
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="agent_read_skill_file",
                    args={"skill": "shell", "path": "SKILL.md"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        llm_runtime=scripted_llm,
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
        and event.get("tool") == "agent_read_skill_file"
    ]
    assert read_events
    assert read_events[0]["result"]["exit_code"] == 1
    assert "not activated" in read_events[0]["result"]["stderr"].lower()


def test_agent_stage3_read_skill_file_allows_activated_skill(tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    _build_temp_skill(skills_root, name="demo")

    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(
                text='{"goal":"g","steps":["read"],"referenced_skills":["demo"]}',
                action=None,
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="agent_read_skill_file",
                    args={"skill": "demo", "path": "references/notes.md"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(skills_root),
        max_iterations=5,
        llm_runtime=scripted_llm,
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
        and event.get("tool") == "agent_read_skill_file"
    ]
    assert read_events
    assert read_events[0]["result"]["exit_code"] == 0
    assert "skill-reference-content" in read_events[0]["result"]["stdout"]


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
                action=ToolCall(tool="agent_done", args={"reply": "ok"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=5,
        llm_runtime=scripted_llm,
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


def test_agent_native_read_skill_file_success_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    _build_temp_skill(skills_root, name="demo")

    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if "tools" not in kwargs:
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"goal":"g","steps":["read"],'
                            '"referenced_skills":["demo"]}'
                        }
                    }
                ]
            }
        tool_defs = kwargs.get("tools", [])
        tool_names = {
            item.get("function", {}).get("name")
            for item in tool_defs
            if isinstance(item, dict)
        }
        if "agent_done" not in tool_names or "agent_read_skill_file" not in tool_names:
            return {"choices": [{"message": {"content": "", "tool_calls": []}}]}
        if len([call for call in calls if "tools" in call]) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "agent_read_skill_file",
                                        "arguments": (
                                            '{"skill":"demo",'
                                            '"path":"references/notes.md"}'
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "agent_done",
                                    "arguments": '{"reply":"done"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    monkeypatch.setattr("agent.llm.litellm.token_counter", lambda **_: 1)

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(skills_root),
        llm_runtime=LLMClient.from_config(
            {
                **_agent_config()["llm"],
                "structured_output": "native",
                "native_probe": False,
                "native_tool_choice": "required",
            }
        ),
        max_iterations=5,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    read_actions = [
        event
        for event in events
        if event.get("type") == "action"
        and event.get("tool") == "agent_read_skill_file"
    ]
    assert read_actions
    assert read_actions[0]["result"]["exit_code"] == 0
    assert "skill-reference-content" in read_actions[0]["result"]["stdout"]


@pytest.mark.parametrize(
    ("plan_payload", "control_arguments", "expected_error_fragment"),
    [
        (
            '{"goal":"g","steps":["read"],"referenced_skills":["demo"]}',
            '{"skill":"demo"}',
            "requires args.skill and args.path",
        ),
        (
            '{"goal":"g","steps":["read"],"referenced_skills":[]}',
            '{"skill":"demo","path":"references/notes.md"}',
            "not activated",
        ),
    ],
)
def test_agent_native_read_skill_file_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plan_payload: str,
    control_arguments: str,
    expected_error_fragment: str,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    _build_temp_skill(skills_root, name="demo")

    calls: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        if "tools" not in kwargs:
            return {"choices": [{"message": {"content": plan_payload}}]}
        if len([call for call in calls if "tools" in call]) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "agent_read_skill_file",
                                        "arguments": control_arguments,
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "agent_done",
                                    "arguments": '{"reply":"done"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    monkeypatch.setattr("agent.llm.litellm.token_counter", lambda **_: 1)

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(skills_root),
        llm_runtime=LLMClient.from_config(
            {
                **_agent_config()["llm"],
                "structured_output": "native",
                "native_probe": False,
                "native_tool_choice": "required",
            }
        ),
        max_iterations=5,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    read_actions = [
        event
        for event in events
        if event.get("type") == "action"
        and event.get("tool") == "agent_read_skill_file"
    ]
    assert read_actions
    assert read_actions[0]["result"]["exit_code"] == 1
    assert expected_error_fragment in str(read_actions[0]["result"]["stderr"])
    assert events[-1]["type"] == "done"
    assert events[-1]["success"] is True


def test_agent_prompt_read_skill_file_success_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    _build_temp_skill(skills_root, name="demo")

    execution_turn = 0

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        nonlocal execution_turn
        messages = kwargs.get("messages")
        first_content = ""
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            content = messages[0].get("content")
            if isinstance(content, str):
                first_content = content
        if first_content.startswith("Return exactly one structured action"):
            execution_turn += 1
            if execution_turn == 1:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"tool":"agent_read_skill_file","args":{"skill":"demo",'
                                    '"path":"references/notes.md"}}'
                                )
                            }
                        }
                    ]
                }
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"tool":"agent_done","args":{"reply":"done"}}'
                        }
                    }
                ]
            }

        return {
            "choices": [
                {
                    "message": {
                        "content": '{"goal":"g","steps":["read"],"referenced_skills":["demo"]}'
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    monkeypatch.setattr("agent.llm.litellm.token_counter", lambda **_: 1)

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(skills_root),
        llm_runtime=LLMClient.from_config(
            {
                **_agent_config()["llm"],
                "structured_output": "prompt",
            }
        ),
        max_iterations=5,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    read_actions = [
        event
        for event in events
        if event.get("type") == "action"
        and event.get("tool") == "agent_read_skill_file"
    ]
    assert read_actions
    assert read_actions[0]["result"]["exit_code"] == 0
    assert "skill-reference-content" in read_actions[0]["result"]["stdout"]


@pytest.mark.parametrize(
    ("plan_payload", "control_payload", "expected_error_fragment"),
    [
        (
            '{"goal":"g","steps":["read"],"referenced_skills":["demo"]}',
            '{"tool":"agent_read_skill_file","args":{"skill":"demo"}}',
            "requires args.skill and args.path",
        ),
        (
            '{"goal":"g","steps":["read"],"referenced_skills":[]}',
            (
                '{"tool":"agent_read_skill_file","args":{"skill":"demo",'
                '"path":"references/notes.md"}}'
            ),
            "not activated",
        ),
    ],
)
def test_agent_prompt_read_skill_file_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plan_payload: str,
    control_payload: str,
    expected_error_fragment: str,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    _build_temp_skill(skills_root, name="demo")

    execution_turn = 0

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        nonlocal execution_turn
        messages = kwargs.get("messages")
        first_content = ""
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            content = messages[0].get("content")
            if isinstance(content, str):
                first_content = content
        if first_content.startswith("Return exactly one structured action"):
            execution_turn += 1
            if execution_turn == 1:
                return {"choices": [{"message": {"content": control_payload}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"tool":"agent_done","args":{"reply":"done"}}'
                        }
                    }
                ]
            }
        return {"choices": [{"message": {"content": plan_payload}}]}

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    monkeypatch.setattr("agent.llm.litellm.token_counter", lambda **_: 1)

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(skills_root),
        llm_runtime=LLMClient.from_config(
            {
                **_agent_config()["llm"],
                "structured_output": "prompt",
            }
        ),
        max_iterations=5,
    )

    worker = threading.Thread(target=agent.run)
    worker.start()
    host_transport.send(_task_event())

    events = _collect_until_done(host_transport)
    worker.join(timeout=2.0)
    assert not worker.is_alive()

    read_actions = [
        event
        for event in events
        if event.get("type") == "action"
        and event.get("tool") == "agent_read_skill_file"
    ]
    assert read_actions
    assert read_actions[0]["result"]["exit_code"] == 1
    assert expected_error_fragment in str(read_actions[0]["result"]["stderr"])
    assert events[-1]["type"] == "done"
    assert events[-1]["success"] is True


def test_agent_prompt_recovers_after_decision_error(monkeypatch: pytest.MonkeyPatch) -> None:
    execution_turn = 0
    execution_payloads: list[dict[str, Any]] = []

    def fake_completion(**kwargs: Any) -> dict[str, Any]:
        nonlocal execution_turn
        messages = kwargs.get("messages")
        first_content = ""
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            content = messages[0].get("content")
            if isinstance(content, str):
                first_content = content
        if first_content.startswith("Return exactly one structured action"):
            execution_turn += 1
            if isinstance(messages, list) and messages and isinstance(messages[-1], dict):
                user_payload = messages[-1].get("content")
                if isinstance(user_payload, str):
                    execution_payloads.append(json.loads(user_payload))
            if execution_turn == 1:
                return {"choices": [{"message": {"content": "I will proceed now."}}]}
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"tool":"agent_done","args":{"reply":"recovered"}}'
                        }
                    }
                ]
            }
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"goal":"g","steps":["attempt"],"referenced_skills":[]}'
                    }
                }
            ]
        }

    monkeypatch.setattr("agent.llm.litellm.completion", fake_completion)
    monkeypatch.setattr("agent.llm.litellm.token_counter", lambda **_: 1)

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        llm_runtime=LLMClient.from_config(
            {
                **_agent_config()["llm"],
                "structured_output": "prompt",
            }
        ),
        max_iterations=5,
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
        if event.get("type") == "action"
        and event.get("tool") == "agent_decision_error"
    ]
    assert error_actions
    assert error_actions[0]["result"]["exit_code"] == 1
    assert "Decision parse error" in str(error_actions[0]["result"]["stderr"])
    assert events[-1]["type"] == "done"
    assert events[-1]["success"] is True
    assert events[-1]["reply"] == "recovered"

    assert len(execution_payloads) == 2
    second_turn_history = execution_payloads[1]["recent_history"]
    assert any(
        isinstance(item, dict)
        and item.get("type") == "action"
        and item.get("tool") == "agent_decision_error"
        and item.get("result", {}).get("exit_code") == 1
        for item in second_turn_history
    )


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

        def run_forever(self) -> None:
            captured["ran_forever"] = True

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
    assert captured["ran_forever"] is True
    assert captured["closed"] is True

    kwargs = captured["agent_kwargs"]
    assert kwargs["skills_dir"] == str(tmp_path / "skills")
    assert kwargs["agent_config_path"] == str(tmp_path / "config.json")
    assert "llm_runtime" not in kwargs
