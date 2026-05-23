"""Tests for YoloSandbox."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agent.llm_types import LLMResponse, ToolCall
from sandbox.yolo import YoloSandbox


class ScriptedLLM:
    """Deterministic LLM used for sandbox tests."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        del messages
        del action_schema
        if not self._responses:
            raise AssertionError("ScriptedLLM exhausted responses.")
        return self._responses.pop(0)

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        del messages
        return 1


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def _task_event(*, approval_mode: str) -> dict[str, Any]:
    return {
        "type": "task",
        "text": "say hello",
        "session_id": "sess-1",
        "approval_mode": approval_mode,
    }


def test_yolo_sandbox_runs_agent_and_exchanges_events() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["do it"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="agent_done",
                    args={"reply": "hello"},
                ),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=scripted_llm,
        agent_config={"llm": {"model": "fake/model", "api_key": "fake-key"}},
    )
    sandbox.run(_task_event(approval_mode="auto"))

    first = sandbox.receive(timeout_seconds=2.0)
    assert first is not None
    assert first["type"] == "message"
    assert first["role"] == "plan"

    done = sandbox.receive(timeout_seconds=2.0)
    assert done is not None
    assert done["type"] == "done"
    assert done["reply"] == "hello"

    sandbox.stop()


def test_yolo_sandbox_receive_returns_none_on_timeout() -> None:
    scripted_llm = ScriptedLLM(
        responses=[LLMResponse(text='{"steps":["wait"]}', action=None, usage=None)]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=scripted_llm,
        agent_config={"llm": {"model": "fake/model", "api_key": "fake-key"}},
    )
    sandbox.run(_task_event(approval_mode="review"))

    plan = sandbox.receive(timeout_seconds=2.0)
    assert plan is not None
    assert plan["type"] == "message"
    assert plan["role"] == "plan"

    timeout_event = sandbox.receive(timeout_seconds=0.2)
    assert timeout_event is None

    sandbox.stop()


def test_yolo_sandbox_lifecycle_send_task_starts_fresh_runtime() -> None:
    scripted_llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["first"]}', action=None, usage=None),
            LLMResponse(text="", action=ToolCall(tool="agent_done", args={"reply": "one"})),
            LLMResponse(text='{"steps":["second"]}', action=None, usage=None),
            LLMResponse(text="", action=ToolCall(tool="agent_done", args={"reply": "two"})),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=scripted_llm,
        agent_config={"llm": {"model": "fake/model", "api_key": "fake-key"}},
    )

    assert sandbox.is_running() is False
    sandbox.start()
    assert sandbox.is_running() is True

    sandbox.send_task(_task_event(approval_mode="auto"))
    assert sandbox.receive(timeout_seconds=2.0)["role"] == "plan"  # type: ignore[index]
    done_one = sandbox.receive(timeout_seconds=2.0)
    assert done_one is not None
    assert done_one["reply"] == "one"
    assert sandbox.is_running() is True

    sandbox.send_task({**_task_event(approval_mode="auto"), "text": "say bye"})
    assert sandbox.receive(timeout_seconds=2.0)["role"] == "plan"  # type: ignore[index]
    done_two = sandbox.receive(timeout_seconds=2.0)
    assert done_two is not None
    assert done_two["reply"] == "two"

    sandbox.stop()
    sandbox.stop()
    assert sandbox.is_running() is False


def test_yolo_sandbox_stop_after_start_without_task_marks_not_running() -> None:
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=ScriptedLLM([]),
        agent_config={"llm": {"model": "fake/model", "api_key": "fake-key"}},
    )

    assert sandbox.is_running() is False
    sandbox.start()
    assert sandbox.is_running() is True

    sandbox.stop()
    assert sandbox.is_running() is False
