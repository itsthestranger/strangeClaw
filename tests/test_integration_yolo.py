"""Integration tests for Yolo mode with deterministic mock LLM."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests
import responses

import agent.agent as agent_module
import session
from adapters.session_persistence import persist_done_event
from agent.llm_types import LLMResponse, ToolCall
from agent.tools import Tools
from coordinator import Coordinator
from sandbox.yolo import YoloSandbox


class ScriptedLLM:
    """Deterministic LLM fixture for integration tests."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

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
            "description: Demo skill for integration tests\n"
            "---\n\n"
            "Use this skill in integration tests.\n"
        ),
        encoding="utf-8",
    )
    (references_dir / "notes.md").write_text("skill-reference-content\n", encoding="utf-8")


def _agent_config() -> dict[str, Any]:
    return {"llm": {"model": "fake/model", "api_key": "fake-key"}}


def _task(approval_mode: str = "auto", state: dict[str, Any] | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "task",
        "text": "integration task",
        "session_id": "sess-1",
        "approval_mode": approval_mode,
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


def _unwrap_data(text: str) -> dict[str, Any]:
    prefix = "--- BEGIN DATA ---\n"
    suffix = "\n--- END DATA ---"
    assert text.startswith(prefix)
    assert text.endswith(suffix)
    payload = text[len(prefix) : -len(suffix)]
    loaded = json.loads(payload)
    assert isinstance(loaded, dict)
    return loaded


def _public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port: [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 0),
            )
        ],
    )


def _wait_until(predicate: Callable[[], bool], timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_yolo_integration_success_path() -> None:
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["check version"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell", args={"command": "python3 --version"}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "Done."}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )
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
                action=ToolCall(tool="agent_done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )
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
                action=ToolCall(tool="agent_clarify",
                    args={"question": "Which environment?"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "clarified"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )
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
                action=ToolCall(tool="shell", args={}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "done"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )
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
    assert "must be a non-empty string" in history[-1]["result"]["stderr"]


def test_yolo_integration_max_iteration_guard() -> None:
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["loop"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell", args={"command": "printf loop"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        max_iterations=1,
        llm_runtime=llm,
        agent_config=_agent_config(),
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
                action=ToolCall(tool="agent_done", args={"reply": "resumed"}),
                usage=None,
            )
        ]
    )
    resume_state = {
        "goal": "integration task",
        "plan": {"steps": ["already planned"]},
        "history": [{"type": "action", "tool": "shell", "args": {}}],
        "summary": "prior summary",
    }
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )
    sandbox.run(_task(state=resume_state))
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    assert all(not (event["type"] == "message" and event.get("role") == "plan") for event in events)
    action_schema = llm.calls[0]["action_schema"]
    assert isinstance(action_schema, list)
    tool_names = [entry.get("name") for entry in action_schema if isinstance(entry, dict)]
    assert all(isinstance(name, str) and "." not in name for name in tool_names)
    expected_names = set(Tools({}).list_enabled())
    expected_names.update(
        {"agent_done", "agent_clarify", "agent_replan", "agent_read_skill_file"}
    )
    assert set(tool_names) == expected_names


def test_yolo_integration_autonomous_replan_read_and_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    _build_temp_skill(skills_root, name="demo")

    llm = ScriptedLLM(
        responses=[
            LLMResponse(
                text='{"goal":"integration task","steps":["inspect"],"referenced_skills":[]}',
                action=None,
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="agent_replan",
                    args={"feedback": "Need skill-backed plan"},
                ),
                usage=None,
            ),
            LLMResponse(
                text=(
                    '{"goal":"integration task","steps":["read reference","finish"],'
                    '"referenced_skills":["demo"]}'
                ),
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
                action=ToolCall(tool="shell", args={"command": "printf integrated"}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "autonomous done"}),
                usage=None,
            ),
        ]
    )
    captured_llm_config: dict[str, Any] = {}
    def fake_from_config(config: dict[str, Any]) -> ScriptedLLM:
        captured_llm_config.update(config)
        return llm

    monkeypatch.setattr(agent_module, "_build_default_llm_runtime", fake_from_config)
    sandbox = YoloSandbox(
        skills_dir=str(skills_root),
        agent_config=_agent_config(),
    )
    task = _task(approval_mode="auto")
    try:
        sandbox.run(task)
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    # Yolo must use construction-time config.
    assert captured_llm_config == _agent_config()["llm"]

    plan_events = [
        event for event in events if event["type"] == "message" and event.get("role") == "plan"
    ]
    assert len(plan_events) == 2
    assert plan_events[1]["content"]["referenced_skills"] == ["demo"]

    read_actions = [
        event
        for event in events
        if event["type"] == "action" and event.get("tool") == "agent_read_skill_file"
    ]
    assert read_actions
    assert read_actions[0]["result"]["exit_code"] == 0
    assert "skill-reference-content" in read_actions[0]["result"]["stdout"]

    done_event = events[-1]
    assert done_event["type"] == "done"
    assert done_event["success"] is True
    assert done_event["reply"] == "autonomous done"


def test_yolo_integration_missing_notion_credentials_denial_no_retry(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: {})
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["create notion page"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="http_request",
                    args={
                        "integration": "notion",
                        "method": "POST",
                        "url": "https://api.notion.com/v1/pages",
                        "headers": {},
                        "body": "{\"parent\":{\"data_source_id\":\"abc\"}}",
                    },
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="agent_done",
                    args={"reply": "Notion integration is not configured; cannot continue."},
                ),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )
    sandbox.run(_task())
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    http_actions = [
        event
        for event in events
        if event["type"] == "action" and event["tool"] == "http_request"
    ]
    assert len(http_actions) == 1
    denial_payload = _unwrap_data(http_actions[0]["result"]["stdout"])
    assert denial_payload["error"] == "policy_denied"
    assert denial_payload["integration"] == "notion"
    assert denial_payload["requested_method"] == "POST"
    assert denial_payload["requested_url"] == "https://api.notion.com/v1/pages"
    assert "not found in secrets.yaml" in denial_payload["reason"]

    done_event = events[-1]
    assert done_event["type"] == "done"
    assert "not configured" in done_event["reply"].lower()


@responses.activate
def test_yolo_integration_http_request_valid_integration_injects_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "notion-valid-token"
    monkeypatch.setattr(
        "sandbox.yolo.load_secrets",
        lambda: {
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
        },
    )
    responses.add(
        responses.POST,
        "https://api.notion.com/v1/pages",
        json={"ok": True},
        status=201,
        headers={"Content-Type": "application/json"},
    )
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["call notion"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="http_request",
                    args={
                        "integration": "notion",
                        "method": "POST",
                        "url": "https://api.notion.com/v1/pages",
                        "headers": {},
                        "body": '{"title":"Test"}',
                    },
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "created"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )

    sandbox.run(_task())
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    assert responses.calls
    outbound_headers = responses.calls[0].request.headers
    assert "Authorization" in outbound_headers
    assert outbound_headers["Authorization"] == f"Bearer {token}"

    action = next(event for event in events if event.get("tool") == "http_request")
    payload = _unwrap_data(action["result"]["stdout"])
    assert payload["success"] is True
    assert payload["status_code"] == 201
    rendered_action = json.dumps(action, ensure_ascii=True, sort_keys=True)
    assert token not in rendered_action


@responses.activate
def test_yolo_integration_http_request_public_get_no_auth_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: {})
    responses.add(
        responses.GET,
        "https://public.example.com/data",
        body="public body",
        status=200,
        headers={"Content-Type": "text/plain"},
    )
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["public get"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="http_request",
                    args={
                        "method": "GET",
                        "url": "https://public.example.com/data",
                        "headers": {},
                        "body": None,
                    },
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "fetched"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )

    sandbox.run(_task())
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    outbound_headers = responses.calls[0].request.headers
    assert "Authorization" not in outbound_headers
    payload = _unwrap_data(
        next(event for event in events if event.get("tool") == "http_request")["result"][
            "stdout"
        ]
    )
    assert payload["success"] is True
    assert payload["body"] == "public body"


@responses.activate
def test_yolo_integration_web_fetch_returns_raw_html_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _public_dns(monkeypatch)
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: {})
    html = "<html><head><title>Example</title></head><body><p>Hello</p></body></html>"
    responses.add(
        responses.GET,
        "https://public.example.com/page",
        body=html,
        status=200,
        headers={"Content-Type": "text/html"},
    )
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["fetch page"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="web_fetch",
                    args={"url": "https://public.example.com/page"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "fetched"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )

    sandbox.run(_task())
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    payload = _unwrap_data(
        next(event for event in events if event.get("tool") == "web_fetch")["result"][
            "stdout"
        ]
    )
    assert payload["success"] is True
    assert payload["status_code"] == 200
    assert payload["body"] == html
    assert payload["truncated"] is False
    assert set(payload) == {"success", "status_code", "headers", "body", "truncated"}


def test_yolo_integration_web_fetch_rfc1918_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: {})
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["fetch internal"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="web_fetch",
                    args={"url": "http://127.0.0.1/private"},
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="agent_done",
                    args={"reply": "cannot fetch private URL"},
                ),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )

    sandbox.run(_task())
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    payload = _unwrap_data(
        next(event for event in events if event.get("tool") == "web_fetch")["result"][
            "stdout"
        ]
    )
    assert payload["success"] is False
    assert payload["error"] == "policy_denied"
    assert payload["integration"] == "_web_fetch"
    assert "blocked address" in payload["reason"]


def test_yolo_integration_web_search_missing_integration_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sandbox.yolo.load_secrets", lambda: {})
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["search"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="web_search", args={"query": "strangeclaw"}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="agent_done",
                    args={"reply": "search is not configured"},
                ),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )

    sandbox.run(_task())
    try:
        events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    payload = _unwrap_data(
        next(event for event in events if event.get("tool") == "web_search")["result"][
            "stdout"
        ]
    )
    assert payload["success"] is False
    assert payload["error"] == "policy_denied"
    assert payload["integration"] == "_web_search"
    assert "web_search integration not configured" in payload["reason"]


def test_yolo_integration_broker_redaction_propagates_to_events_and_persistence(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "sandbox.yolo.load_secrets",
        lambda: {
            "notion": {
                "name": "notion",
                "auth_type": "bearer",
                "token": "notion-secret-token",
                "allowed_hosts": ["api.notion.com"],
                "allowed_methods": ["POST"],
                "allowed_paths": ["/v1/*"],
                "protected_headers": ["Authorization"],
                "default_headers": {},
                "max_response_bytes": 4096,
                "rate_limit": None,
            }
        },
    )
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["call notion"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="http_request",
                    args={
                        "integration": "notion",
                        "method": "POST",
                        "url": "https://api.notion.com/v1/pages",
                        "headers": {},
                        "body": '{"title":"x"}',
                    },
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "finished"}),
                usage=None,
            ),
        ]
    )
    sandbox = YoloSandbox(
        skills_dir=str(_skills_root()),
        llm_runtime=llm,
        agent_config=_agent_config(),
    )

    class _EchoResponse:
        status_code = 200
        headers = {"Content-Type": "application/json"}
        encoding = "utf-8"

        def __init__(self, body: str) -> None:
            self._body = body

        def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
            _ = chunk_size
            return [self._body.encode("utf-8")]

    def _fake_request(*args: object, **kwargs: object) -> _EchoResponse:
        _ = args
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        authorization = str(headers.get("Authorization", ""))
        return _EchoResponse(f'{{"echo":"{authorization}"}}')

    sandbox.run(
        {
            "type": "task",
            "text": "integration task",
            "session_id": "sess-redaction",
            "approval_mode": "auto",
        }
    )
    try:
        with patch.object(requests.Session, "request", side_effect=_fake_request):
            events = _collect_until_done(sandbox)
    finally:
        sandbox.stop()

    action_events = [
        event
        for event in events
        if event["type"] == "action" and event["tool"] == "http_request"
    ]
    assert len(action_events) == 1
    action_event = action_events[0]
    action_payload = _unwrap_data(action_event["result"]["stdout"])
    rendered_action = json.dumps(action_payload, ensure_ascii=True, sort_keys=True)
    assert "notion-secret-token" not in rendered_action
    assert "[REDACTED]" in rendered_action

    done_event = events[-1]
    assert done_event["type"] == "done"
    rendered_state = json.dumps(done_event["state"], ensure_ascii=True, sort_keys=True)
    assert "notion-secret-token" not in rendered_state

    persist_done_event(session_id="sess-redaction", done_event=done_event)
    session_dir = session.create("sess-redaction")
    state_on_disk = session.load(session_dir)
    assert isinstance(state_on_disk, dict)
    rendered_persisted = json.dumps(state_on_disk, ensure_ascii=True, sort_keys=True)
    assert "notion-secret-token" not in rendered_persisted


def test_yolo_integration_credential_isolation_in_journal_and_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    integration_token = "notion-journal-secret"
    llm_api_key = "llm-journal-secret"
    monkeypatch.setattr(
        "sandbox.yolo.load_secrets",
        lambda: {
            "notion": {
                "auth_type": "bearer",
                "token": integration_token,
                "allowed_hosts": ["api.notion.com"],
                "allowed_methods": ["POST"],
                "allowed_paths": ["/v1/*"],
                "allowed_schemes": ["https"],
                "protected_headers": ["Authorization"],
                "default_headers": {},
                "max_response_bytes": 4096,
                "rate_limit": None,
            }
        },
    )
    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["call notion"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(
                    tool="http_request",
                    args={
                        "integration": "notion",
                        "method": "POST",
                        "url": "https://api.notion.com/v1/pages",
                        "headers": {},
                        "body": '{"title":"journal"}',
                    },
                ),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "finished"}),
                usage=None,
            ),
        ]
    )

    class _EchoResponse:
        status_code = 200
        headers = {"Content-Type": "application/json"}
        encoding = "utf-8"

        def __init__(self, body: str) -> None:
            self._body = body

        def iter_content(self, chunk_size: int = 8192) -> list[bytes]:
            del chunk_size
            return [self._body.encode("utf-8")]

    def _fake_request(*args: object, **kwargs: object) -> _EchoResponse:
        del args
        headers = kwargs.get("headers")
        assert isinstance(headers, dict)
        authorization = str(headers.get("Authorization", ""))
        return _EchoResponse(f'{{"echo":"{authorization}"}}')

    agent_config: dict[str, Any] = {
        "llm": {"model": "fake/model", "api_key": llm_api_key},
        "session_journal": {"enabled": True, "max_bytes": 100_000},
    }
    coordinator = Coordinator(
        sandbox_factory=lambda: YoloSandbox(
            skills_dir=str(_skills_root()),
            llm_runtime=llm,
            agent_config=agent_config,
        ),
        approval_mode="auto",
        llm_config=agent_config["llm"],
        session_journal={"enabled": True, "max_bytes": 100_000},
        session_idle_timeout_seconds=0,
    )
    events: list[dict[str, Any]] = []

    with caplog.at_level("DEBUG"):
        with patch.object(requests.Session, "request", side_effect=_fake_request):
            status = coordinator.start_task(
                session_id="sess-journal",
                text="integration task",
                sink=events.append,
            )
            assert status == "started"
            assert _wait_until(
                lambda: bool(events) and events[-1].get("type") == "done",
                timeout=3.0,
            )
    coordinator.stop_all()

    rendered_events = json.dumps(events, ensure_ascii=True, sort_keys=True)
    assert integration_token not in rendered_events
    assert llm_api_key not in rendered_events
    assert "[REDACTED]" in rendered_events

    session_dir = session.create("sess-journal")
    state_on_disk = session.load(session_dir)
    assert isinstance(state_on_disk, dict)
    journal_text = (session_dir / "events.jsonl").read_text(encoding="utf-8")
    rendered_state = json.dumps(state_on_disk, ensure_ascii=True, sort_keys=True)
    captured = capsys.readouterr()

    for surface in (
        rendered_state,
        journal_text,
        caplog.text,
        captured.out,
        captured.err,
    ):
        assert integration_token not in surface
        assert llm_api_key not in surface
