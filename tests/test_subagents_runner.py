"""Tests for the sequential SubagentRunner (C10.4)."""

from __future__ import annotations

import json
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

import agent.subagents as subagents_module
from agent.agent import Agent
from agent.llm_types import LLMResponse, ToolCall
from agent.subagents import SubagentRunner
from agent.transport import InProcessTransport


class _ScriptedLLM:
    """Deterministic fake LLM shared by parent and child in a test."""

    def __init__(self, responses: list[LLMResponse], *, sleep_first: float = 0.0) -> None:
        self._responses = responses
        self._sleep_first = sleep_first
        self.calls = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self.calls == 0 and self._sleep_first:
            time.sleep(self._sleep_first)
        self.calls += 1
        if not self._responses:
            raise AssertionError("ScriptedLLM exhausted responses.")
        return self._responses.pop(0)

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        return 1


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def _plan(steps: str = "do the thing") -> LLMResponse:
    return LLMResponse(
        text=json.dumps({"goal": "g", "steps": [steps], "referenced_skills": []}),
        action=None,
    )


def _make_runner(
    llm: _ScriptedLLM,
    tmp_path: Path,
    *,
    parent_enabled_tools: list[str] | None = None,
    limits: dict[str, Any] | None = None,
) -> SubagentRunner:
    return SubagentRunner(
        llm_runtime=llm,
        broker=None,
        skills_dir=str(_skills_root()),
        base_config={},
        parent_enabled_tools=parent_enabled_tools or ["shell", "web_search"],
        parent_session_id="sess-1",
        output_root=str(tmp_path / "output" / "subagents"),
        limits=limits or {"max_iterations": 20, "timeout_seconds": 600},
    )


def _request(**overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "task": "investigate",
        "context": "",
        "expected_output": "",
        "allowed_tools": ["shell"],
        "referenced_skills": [],
        "max_iterations": 5,
        "timeout_seconds": 30,
    }
    request.update(overrides)
    return request


def test_runner_completed_returns_child_report(tmp_path: Path) -> None:
    done = ToolCall(tool="agent_done", args={"reply": "child report"})
    llm = _ScriptedLLM([_plan(), LLMResponse(text="", action=done)])
    runner = _make_runner(llm, tmp_path)

    envelope = runner.run(_request())

    assert envelope["success"] is True
    assert envelope["status"] == "completed"
    assert envelope["reply"] == "child report"
    assert isinstance(envelope["events_summary"], list)


def test_runner_captures_tool_actions_and_hits_max_iterations(tmp_path: Path) -> None:
    llm = _ScriptedLLM(
        [
            _plan(),
            LLMResponse(text="", action=ToolCall(tool="shell", args={"command": "echo one"})),
            LLMResponse(text="", action=ToolCall(tool="shell", args={"command": "echo two"})),
        ]
    )
    runner = _make_runner(llm, tmp_path)

    envelope = runner.run(_request(max_iterations=2))

    assert envelope["success"] is False
    assert envelope["status"] == "max_iterations"
    summary = envelope["events_summary"]
    assert [entry["tool"] for entry in summary] == ["shell", "shell"]
    assert all(entry["exit_code"] == 0 for entry in summary)


def test_runner_timeout_when_deadline_passes(tmp_path: Path) -> None:
    # Planning sleeps past the budget; the deadline trips at the first loop boundary.
    llm = _ScriptedLLM([_plan()], sleep_first=0.1)
    runner = _make_runner(llm, tmp_path)

    envelope = runner.run(_request(timeout_seconds=0.02, max_iterations=5))

    assert envelope["success"] is False
    assert envelope["status"] == "timeout"


def test_build_child_config_restricts_tools_and_disables_recursion(tmp_path: Path) -> None:
    runner = _make_runner(
        _ScriptedLLM([]),
        tmp_path,
        parent_enabled_tools=["shell", "web_search"],
    )

    config = runner._build_child_config(
        {"allowed_tools": ["web_search", "http_request"]}, max_iterations=7
    )

    # http_request was not parent-enabled, so it is dropped; spawn_subagent is off.
    assert config["tools"] == {
        "shell": False,
        "web_search": True,
        "web_fetch": False,
        "http_request": False,
        "spawn_subagent": False,
    }
    assert config["subagents"]["enabled"] is False
    assert config["loop"] == {"max_iterations": 7}


def test_compose_goal_includes_context_and_expected_output(tmp_path: Path) -> None:
    runner = _make_runner(_ScriptedLLM([]), tmp_path)

    goal = runner._compose_goal(
        {
            "task": "do x",
            "context": "background info",
            "expected_output": "a summary",
            "referenced_skills": ["web-research"],
        }
    )

    assert "do x" in goal
    assert "background info" in goal
    assert "a summary" in goal
    assert "web-research" in goal


def _fix_child_id(monkeypatch: pytest.MonkeyPatch) -> str:
    fixed = uuid.UUID("00000000-0000-4000-8000-0000000abcde")
    monkeypatch.setattr(subagents_module.uuid, "uuid4", lambda: fixed)
    return fixed.hex[:12]


def _shell(command: str) -> LLMResponse:
    return LLMResponse(text="", action=ToolCall(tool="shell", args={"command": command}))


def _done(reply: str = "done") -> LLMResponse:
    return LLMResponse(text="", action=ToolCall(tool="agent_done", args={"reply": reply}))


def test_runner_returns_child_files_with_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_id = _fix_child_id(monkeypatch)
    out_file = tmp_path / "output" / "subagents" / child_id / "report.txt"
    llm = _ScriptedLLM(
        [_plan(), _shell(f"printf hello > {shlex.quote(str(out_file))}"), _done()]
    )
    runner = _make_runner(llm, tmp_path)

    envelope = runner.run(_request(max_iterations=5))

    assert envelope["status"] == "completed"
    assert [f["path"] for f in envelope["files"]] == [f"subagents/{child_id}/report.txt"]


def test_runner_excludes_files_outside_child_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fix_child_id(monkeypatch)
    outside = tmp_path / "outside.txt"
    llm = _ScriptedLLM(
        [_plan(), _shell(f"printf leak > {shlex.quote(str(outside))}"), _done()]
    )
    runner = _make_runner(llm, tmp_path)

    envelope = runner.run(_request(max_iterations=5))

    assert envelope["status"] == "completed"
    assert envelope["files"] == []


def test_runner_skips_symlinked_child_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_id = _fix_child_id(monkeypatch)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    link = tmp_path / "output" / "subagents" / child_id / "link.txt"
    llm = _ScriptedLLM(
        [
            _plan(),
            _shell(f"ln -s {shlex.quote(str(secret))} {shlex.quote(str(link))}"),
            _done(),
        ]
    )
    runner = _make_runner(llm, tmp_path)

    envelope = runner.run(_request(max_iterations=5))

    # A symlink that escapes the child dir is skipped, never exported.
    assert envelope["files"] == []


def test_runner_oversized_child_output_is_bounded_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_id = _fix_child_id(monkeypatch)
    big = tmp_path / "output" / "subagents" / child_id / "big.bin"
    llm = _ScriptedLLM(
        [_plan(), _shell(f"printf 0123456789abcdef > {shlex.quote(str(big))}"), _done()]
    )
    runner = _make_runner(
        llm,
        tmp_path,
        limits={"max_iterations": 5, "timeout_seconds": 600, "max_files_bytes": 8},
    )

    envelope = runner.run(_request(max_iterations=5))

    assert envelope["success"] is False
    assert envelope["status"] == "child_failed"
    assert "exceeds output limit" in envelope["reply"]


def test_child_clarify_degrades_and_child_still_completes(tmp_path: Path) -> None:
    # A child that tries to clarify must not block on a (nonexistent) user reply;
    # the stray clarify degrades to a decision error and the child carries on.
    clarify = LLMResponse(text="", action=ToolCall(tool="agent_clarify", args={"question": "?"}))
    llm = _ScriptedLLM([_plan(), clarify, _done("finished without asking")])
    runner = _make_runner(llm, tmp_path)

    envelope = runner.run(_request(max_iterations=5))

    assert envelope["status"] == "completed"
    assert envelope["reply"] == "finished without asking"
    clarify_entries = [e for e in envelope["events_summary"] if e["tool"] == "agent_clarify"]
    assert clarify_entries and clarify_entries[0]["exit_code"] == 1


def test_child_output_instruction_points_at_child_dir(tmp_path: Path) -> None:
    _host, agent_transport = InProcessTransport.pair()
    child_dir = tmp_path / "output" / "subagents" / "c1"
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        agent_config={"llm": {"model": "f", "api_key": "k"}},
        output_dir=str(child_dir),
        llm_runtime=_ScriptedLLM([]),
    )

    messages = agent.build_execution_prompt(
        goal="g", plan={"steps": []}, history=[], activated_skills={}
    )
    payload = json.loads(messages[1]["content"])

    assert str(child_dir) in payload["output_instruction"]


def test_collect_child_files_prefixes_paths() -> None:
    done = {"files": [{"path": "out.txt", "size_bytes": 3, "content_b64": "QUJD"}]}
    files = SubagentRunner._collect_child_files("abc123", done)
    assert files == [
        {"path": "subagents/abc123/out.txt", "size_bytes": 3, "content_b64": "QUJD"}
    ]


def _parent_config() -> dict[str, Any]:
    return {
        "llm": {"model": "fake/model", "api_key": "fake-key"},
        "tools": {
            "shell": True,
            "web_search": False,
            "web_fetch": False,
            "http_request": False,
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


def test_parent_does_not_leak_child_events(tmp_path: Path) -> None:
    # One shared scripted LLM serves, in order: parent plan, parent spawn decision,
    # child plan, child done, parent done.
    llm = _ScriptedLLM(
        [
            _plan(),
            LLMResponse(
                text="",
                action=ToolCall(tool="spawn_subagent", args={"task": "sub", "allowed_tools": []}),
            ),
            _plan(),
            LLMResponse(text="", action=ToolCall(tool="agent_done", args={"reply": "child done"})),
            LLMResponse(text="", action=ToolCall(tool="agent_done", args={"reply": "parent done"})),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        agent_config=_parent_config(),
        output_dir=str(tmp_path / "output"),
        llm_runtime=llm,
    )

    thread = threading.Thread(target=agent.run, daemon=True)
    thread.start()
    host_transport.send(
        {"type": "task", "text": "parent goal", "session_id": "sess-1", "approval_mode": "auto"}
    )

    events: list[dict[str, Any]] = []
    while True:
        event = host_transport.receive(timeout_seconds=3.0)
        assert event is not None, "Timed out waiting for parent events."
        events.append(event)
        if event["type"] == "done":
            break
    thread.join(timeout=2.0)

    actions = [e for e in events if e["type"] == "action"]
    plans = [e for e in events if e["type"] == "message" and e.get("role") == "plan"]
    dones = [e for e in events if e["type"] == "done"]

    # The child's own plan message and done never surface; only the parent's do.
    assert len(plans) == 1
    assert [a["tool"] for a in actions] == ["spawn_subagent"]
    assert len(dones) == 1
    assert dones[0]["reply"] == "parent done"
    # The child's result is surfaced once, as the spawn_subagent observation.
    assert "child done" in actions[0]["result"]["stdout"]
