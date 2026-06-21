"""Tests for the spawn_subagent action surface, validation, and result contract (C10.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.agent import Agent
from agent.llm_types import LLMResponse, ToolCall
from agent.transport import InProcessTransport


class _NullLLM:
    """LLM that must never be called: these tests drive dispatch directly."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        raise AssertionError("LLM should not be called in these tests.")

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        return 1


class _FakeRunner:
    """Records the request it receives and returns a canned envelope."""

    def __init__(self, envelope: dict[str, Any]) -> None:
        self.envelope = envelope
        self.requests: list[dict[str, Any]] = []

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return self.envelope


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def _config(*, gates_on: bool = True, **subagent_overrides: Any) -> dict[str, Any]:
    subagents: dict[str, Any] = {
        "enabled": gates_on,
        "max_children_per_task": 3,
        "max_iterations": 20,
        "timeout_seconds": 600,
        "max_context_chars": 20000,
        "max_result_chars": 20000,
        "max_files_bytes": 10 * 1024 * 1024,
        "journal_events": "summary",
    }
    subagents.update(subagent_overrides)
    return {
        "llm": {"model": "fake/model", "api_key": "fake-key"},
        "tools": {
            "shell": True,
            "web_search": True,
            "web_fetch": False,
            "http_request": False,
            "spawn_subagent": gates_on,
        },
        "subagents": subagents,
    }


def _make_agent(
    config: dict[str, Any],
    *,
    runner: _FakeRunner | None = None,
) -> Agent:
    _host, agent_transport = InProcessTransport.pair()
    return Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        agent_config=config,
        llm_runtime=_NullLLM(),
        subagent_runner=runner,
    )


def _surface_names(agent: Agent) -> list[str]:
    return [schema["name"] for schema in agent._execution_action_surface]


def _parse_wrapped(stdout: str) -> dict[str, Any]:
    assert stdout.startswith("--- BEGIN DATA ---\n")
    body = stdout.split("--- BEGIN DATA ---\n", 1)[1].rsplit("\n--- END DATA ---", 1)[0]
    parsed = json.loads(body)
    assert isinstance(parsed, dict)
    return parsed


def _dispatch_spawn(agent: Agent, args: dict[str, Any]) -> dict[str, Any]:
    outcome = agent._handle_spawn_subagent(
        decision=ToolCall(tool="spawn_subagent", args=args),
        current_plan={"goal": "g", "steps": [], "referenced_skills": []},
    )
    observation = outcome["observation"]
    assert observation["tool"] == "spawn_subagent"
    return _parse_wrapped(observation["result"]["stdout"])


def test_schema_present_only_when_both_gates_enabled() -> None:
    assert "spawn_subagent" in _surface_names(_make_agent(_config(gates_on=True)))
    assert "spawn_subagent" not in _surface_names(_make_agent(_config(gates_on=False)))


def test_schema_absent_when_only_tool_toggle_enabled() -> None:
    config = _config(gates_on=True)
    config["subagents"]["enabled"] = False
    assert "spawn_subagent" not in _surface_names(_make_agent(config))


def test_disabled_call_returns_deterministic_envelope() -> None:
    agent = _make_agent(_config(gates_on=False))
    envelope = _dispatch_spawn(agent, {"task": "do something"})
    assert envelope["success"] is False
    assert envelope["status"] == "disabled"


def test_missing_task_returns_invalid_request() -> None:
    agent = _make_agent(_config(), runner=_FakeRunner({"success": True, "status": "completed"}))
    envelope = _dispatch_spawn(agent, {"context": "no task here"})
    assert envelope["status"] == "invalid_request"
    assert "task" in envelope["reason"]


def test_allowed_tools_must_be_subset_of_parent() -> None:
    agent = _make_agent(_config(), runner=_FakeRunner({"success": True, "status": "completed"}))
    envelope = _dispatch_spawn(agent, {"task": "t", "allowed_tools": ["http_request"]})
    assert envelope["status"] == "invalid_request"
    assert "http_request" in envelope["reason"]


def test_allowed_tools_cannot_include_spawn_subagent() -> None:
    agent = _make_agent(_config(), runner=_FakeRunner({"success": True, "status": "completed"}))
    envelope = _dispatch_spawn(agent, {"task": "t", "allowed_tools": ["spawn_subagent"]})
    assert envelope["status"] == "invalid_request"
    assert "recursion" in envelope["reason"]


def test_runner_unavailable_returns_child_failed() -> None:
    agent = _make_agent(_config(), runner=None)
    envelope = _dispatch_spawn(agent, {"task": "t"})
    assert envelope["status"] == "child_failed"


def test_valid_request_delegates_to_runner_with_normalized_fields() -> None:
    runner = _FakeRunner({"success": True, "status": "completed", "reply": "ok"})
    agent = _make_agent(_config(max_iterations=10, timeout_seconds=100), runner=runner)

    envelope = _dispatch_spawn(
        agent,
        {
            "task": "  investigate  ",
            "allowed_tools": ["web_search", "web_search"],
            "max_iterations": 999,
            "timeout_seconds": 5,
        },
    )

    assert envelope["success"] is True
    assert envelope["status"] == "completed"
    assert len(runner.requests) == 1
    request = runner.requests[0]
    assert request["task"] == "investigate"
    assert request["allowed_tools"] == ["web_search"]
    assert request["max_iterations"] == 10  # clamped to the config cap
    assert request["timeout_seconds"] == 5  # under the cap, preserved


def test_allowed_tools_defaults_to_parent_enabled_subset() -> None:
    runner = _FakeRunner({"success": True, "status": "completed"})
    agent = _make_agent(_config(), runner=runner)

    _dispatch_spawn(agent, {"task": "t"})

    assert runner.requests[0]["allowed_tools"] == ["shell", "web_search"]


def test_context_is_truncated_to_max_context_chars() -> None:
    runner = _FakeRunner({"success": True, "status": "completed"})
    agent = _make_agent(_config(max_context_chars=10), runner=runner)

    _dispatch_spawn(agent, {"task": "t", "context": "x" * 100})

    assert "truncated" in runner.requests[0]["context"]
    assert runner.requests[0]["context"].startswith("x" * 10)


def test_result_envelope_is_bounded_and_files_reduced_to_metadata() -> None:
    runner = _FakeRunner(
        {
            "success": True,
            "status": "completed",
            "reply": "R" * 100,
            "state_summary": "S" * 100,
            "files": [
                {"path": "subagents/c1/out.txt", "size_bytes": 12, "content_b64": "QUJD"},
            ],
            "events_summary": [{"i": i} for i in range(120)],
        }
    )
    agent = _make_agent(_config(max_result_chars=20), runner=runner)

    envelope = _dispatch_spawn(agent, {"task": "t"})

    assert envelope["reply"].startswith("R" * 20)
    assert "truncated" in envelope["reply"]
    assert "truncated" in envelope["state_summary"]
    assert envelope["files"] == [{"path": "subagents/c1/out.txt", "size_bytes": 12}]
    assert "content_b64" not in envelope["files"][0]
    assert len(envelope["events_summary"]) == 50


def test_runner_exception_becomes_child_failed_observation() -> None:
    class _BoomRunner:
        def run(self, request: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("boom")

    _host, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        agent_config=_config(),
        llm_runtime=_NullLLM(),
        subagent_runner=_BoomRunner(),
    )

    envelope = _dispatch_spawn(agent, {"task": "t"})
    assert envelope["status"] == "child_failed"
    assert "boom" in envelope["reason"]
