"""Tests for agent core loop."""

from __future__ import annotations

import base64
import shlex
import threading
from pathlib import Path
from typing import Any

from agent.agent import Agent
from agent.llm import LLMResponse, ToolCall
from agent.transport import InProcessTransport


class ScriptedLLM:
    """Deterministic fake LLM for agent-loop tests."""

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
                action=ToolCall(skill="shell", action="run", args={"command": "python3 --version"}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    skill="shell",
                    action="run",
                    args={"command": f'printf \'print("hello")\\n\' > {quoted_hello}'},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    skill="shell",
                    action="run",
                    args={"command": f"printf artifact > {quoted_output}"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    skill="__agent__",
                    action="done",
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
        llm_factory=lambda _: scripted_llm,  # type: ignore[arg-type]
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
    assert action_events[0]["skill"] == "shell"
    assert action_events[0]["action"] == "run"
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
                action=ToolCall(
                    skill="__agent__",
                    action="done",
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
        llm_factory=lambda _: scripted_llm,  # type: ignore[arg-type]
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
                action=ToolCall(skill="shell", action="run", args={"command": "printf one-step"}),
                usage=None,
            ),
        ]
    )

    host_transport, agent_transport = InProcessTransport.pair()
    agent = Agent(
        transport=agent_transport,
        skills_dir=str(_skills_root()),
        max_iterations=1,
        llm_factory=lambda _: scripted_llm,  # type: ignore[arg-type]
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
