"""Integration tests for Yolo mode with deterministic mock LLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent.agent import EXECUTION_ACTION_SCHEMA
from agent.llm import LLMResponse, ToolCall
from sandbox.yolo import YoloSandbox


class ScriptedLLM:
    """Deterministic LLM fixture for integration tests."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

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
        del messages
        return 1


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def _task(approval_mode: str = "auto", state: dict[str, Any] | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "task",
        "text": "integration task",
        "session_id": "sess-1",
        "approval_mode": approval_mode,
        "llm": {"model": "fake/model", "api_key": "fake-key"},
    }
    if state is not None:
        event["state"] = state
    return event


def _collect_until_done(
    sandbox: YoloSandbox,
    *,
    review_replies: list[dict[str, Any]] | None = None,
    clarification_replies: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    plan_replies = review_replies[:] if review_replies else []
    clarify_replies = clarification_replies[:] if clarification_replies else []

    while True:
        event = sandbox.receive(timeout_seconds=2.0)
        assert event is not None, "Timed out waiting for sandbox event."
        events.append(event)

        if event["type"] == "message" and event.get("role") == "plan" and plan_replies:
            sandbox.send({"type": "user_reply", **plan_replies.pop(0)})
        if event["type"] == "message" and event.get("role") == "clarification" and clarify_replies:
            sandbox.send({"type": "user_reply", **clarify_replies.pop(0)})
        if event["type"] == "done":
            return events


def test_yolo_integration_success_path() -> None:
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["check version"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(skill="shell", action="run", args={"command": "python3 --version"}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(skill="__agent__", action="done", args={"reply": "Done."}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(skills_dir=str(_skills_root()), llm_factory=lambda _: llm)
    sandbox.run(_task())
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    assert events[0]["type"] == "message"
    assert events[0]["role"] == "plan"
    assert any(event["type"] == "action" for event in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["success"] is True


def test_yolo_integration_plan_rejection_and_replan() -> None:
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["first"]}', action=None, usage=None),
            LLMResponse(text='{"steps":["replanned"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(skill="__agent__", action="done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(skills_dir=str(_skills_root()), llm_factory=lambda _: llm)
    sandbox.run(_task(approval_mode="review"))
    try:
        events = _collect_until_done(
            sandbox,
            review_replies=[
                {"approved": False, "text": "try another plan"},
                {"approved": True, "text": "looks good"},
            ],
        )
    finally:
        sandbox.stop()

    plan_events = [
        event for event in events if event["type"] == "message" and event["role"] == "plan"
    ]
    assert len(plan_events) == 2


def test_yolo_integration_clarification_round_trip() -> None:
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["clarify"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    skill="__agent__",
                    action="clarify",
                    args={"question": "Which environment?"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(skill="__agent__", action="done", args={"reply": "clarified"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(skills_dir=str(_skills_root()), llm_factory=lambda _: llm)
    sandbox.run(_task())
    try:
        events = _collect_until_done(
            sandbox,
            clarification_replies=[{"approved": True, "text": "Use dev"}],
        )
    finally:
        sandbox.stop()

    assert any(
        event["type"] == "message" and event.get("role") == "clarification" for event in events
    )
    assert events[-1]["type"] == "done"


def test_yolo_integration_invalid_tool_call_is_observed_next_turn() -> None:
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["attempt invalid shell"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(skill="shell", action="run", args={}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(skill="__agent__", action="done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(skills_dir=str(_skills_root()), llm_factory=lambda _: llm)
    sandbox.run(_task())
    try:
        _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    execution_call = llm.calls[2]
    payload = json.loads(execution_call["messages"][1]["content"])
    history = payload["recent_history"]
    assert history
    assert history[-1]["result"]["exit_code"] == 1
    assert "Invalid args" in history[-1]["result"]["stderr"]


def test_yolo_integration_max_iteration_guard() -> None:
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["loop"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(skill="shell", action="run", args={"command": "printf loop"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        max_iterations=1,
        llm_factory=lambda _: llm,
    )
    sandbox.run(_task())
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    assert any(
        event["type"] == "message" and event.get("role") == "clarification" for event in events
    )
    assert events[-1]["type"] == "done"
    assert events[-1]["success"] is False


def test_yolo_integration_resume_from_saved_state_skips_replanning() -> None:
    llm = ScriptedLLM(
        responses=[
            LLMResponse(
                text="",
                action=ToolCall(skill="__agent__", action="done", args={"reply": "resumed"}),
                usage=None,
            )
        ]
    )
    resume_state = {
        "goal": "integration task",
        "plan": {"steps": ["already planned"]},
        "history": [{"type": "action", "skill": "shell", "action": "run", "args": {}}],
        "summary": "prior summary",
    }
    sandbox = YoloSandbox(skills_dir=str(_skills_root()), llm_factory=lambda _: llm)
    sandbox.run(_task(state=resume_state))
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    assert all(not (event["type"] == "message" and event.get("role") == "plan") for event in events)
    assert llm.calls[0]["action_schema"] == EXECUTION_ACTION_SCHEMA
