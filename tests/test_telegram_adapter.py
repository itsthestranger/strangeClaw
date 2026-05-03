"""Tests for TelegramAdapter core behavior."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from adapters.telegram import (
    MARKDOWN_V2_PARSE_MODE,
    TelegramAdapter,
    TelegramLimits,
)
from agent.llm import LLMResponse, ToolCall
from coordinator import Coordinator
from sandbox.yolo import YoloSandbox


class FakeBot:
    """Async bot stub for adapter tests."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.documents: list[dict[str, Any]] = []
        self.chat_actions: list[dict[str, Any]] = []

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_markup: Any = None,
        parse_mode: str | None = None,
    ) -> None:
        self.messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "reply_markup": reply_markup,
                "parse_mode": parse_mode,
            }
        )

    async def send_document(self, *, chat_id: int, document: Any, caption: str = "") -> None:
        self.documents.append(
            {
                "chat_id": chat_id,
                "caption": caption,
                "name": getattr(document, "name", ""),
            }
        )

    async def send_chat_action(self, *, chat_id: int, action: str) -> None:
        self.chat_actions.append({"chat_id": chat_id, "action": action})


class FakeApplication:
    """Application wrapper exposing bot attribute."""

    def __init__(self, bot: FakeBot) -> None:
        self.bot = bot


class FakeCoordinator:
    """Coordinator stub for adapter tests."""

    def __init__(self, *, start_status: str = "started") -> None:
        self.start_status = start_status
        self.pending: dict[str, str] = {}
        self.started_tasks: list[dict[str, Any]] = []
        self.submitted_replies: list[dict[str, Any]] = []

    def pending_role(self, *, session_id: str) -> str | None:
        return self.pending.get(session_id)

    def submit_reply(self, *, session_id: str, approved: bool, text: str) -> bool:
        role = self.pending.get(session_id)
        if role is None:
            return False
        self.pending.pop(session_id, None)
        self.submitted_replies.append(
            {"session_id": session_id, "approved": approved, "text": text}
        )
        return True

    def start_task(self, *, session_id: str, text: str, sink: Any) -> str:
        self.started_tasks.append({"session_id": session_id, "text": text, "sink": sink})
        return self.start_status


class ScriptedLLM:
    """Deterministic LLM fixture for Telegram integration tests."""

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


def _update(chat_id: int, text: str) -> Any:
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=SimpleNamespace(text=text),
    )


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def test_get_task_uses_chat_id_session_and_llm() -> None:
    adapter = TelegramAdapter(
        sandbox_factory=lambda: object(),
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        token="token",
    )
    task = adapter.get_task(_update(42, "  build a tool  "))

    assert task == {
        "type": "task",
        "text": "build a tool",
        "session_id": "42",
        "approval_mode": "review",
    }


def test_get_task_applies_session_id_prefix_when_configured() -> None:
    adapter = TelegramAdapter(
        sandbox_factory=lambda: object(),
        approval_mode="review",
        llm_config={"model": "x", "api_key": "k"},
        token="token",
        session_id_prefix="telegram-",
    )
    task = adapter.get_task(_update(42, "task"))
    assert task["session_id"] == "telegram-42"


def test_chunk_text_respects_limit() -> None:
    text = "line1\nline2\nline3"
    chunks = TelegramAdapter._chunk_text(text, limit=6)

    assert chunks == ["line1", "line2", "line3"]


def test_chunk_markdown_text_prefers_code_fence_boundaries() -> None:
    text = "*Action:* shell\n```text\nline1\nline2\n```\nend"
    chunks = TelegramAdapter._chunk_markdown_text(text, limit=24)
    assert len(chunks) >= 2
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks[:-1])


def test_show_done_sends_text_and_document() -> None:
    adapter = TelegramAdapter(sandbox_factory=lambda: object(), token="token")
    bot = FakeBot()
    adapter._application = FakeApplication(bot)  # noqa: SLF001

    payload = {
        "type": "done",
        "success": True,
        "reply": "finished",
        "files": [
            {
                "path": "nested/out.txt",
                "content_b64": base64.b64encode(b"artifact").decode("ascii"),
            }
        ],
    }

    asyncio.run(adapter.show(payload, chat_id=7))

    assert bot.messages == [
        {
            "chat_id": 7,
            "text": "*Success:* finished",
            "reply_markup": None,
            "parse_mode": MARKDOWN_V2_PARSE_MODE,
        }
    ]
    assert bot.documents == [{"chat_id": 7, "caption": "Output: nested/out.txt", "name": "out.txt"}]


def test_show_plan_review_uses_text_approval_prompt_without_buttons() -> None:
    adapter = TelegramAdapter(
        sandbox_factory=lambda: object(),
        token="token",
        approval_mode="review",
    )
    bot = FakeBot()
    adapter._application = FakeApplication(bot)  # noqa: SLF001

    asyncio.run(
        adapter.show(
            {"type": "message", "role": "plan", "content": {"steps": ["one"]}},
            chat_id=7,
        )
    )

    assert len(bot.messages) == 1
    assert bot.messages[0]["reply_markup"] is None
    assert "Reply with" in bot.messages[0]["text"]


def test_send_done_files_reports_upload_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    adapter = TelegramAdapter(
        sandbox_factory=lambda: object(),
        token="token",
        limits=TelegramLimits(
            max_active_sessions=8,
            max_output_total_bytes=5,
            max_output_file_bytes=5,
        ),
    )
    bot = FakeBot()
    adapter._application = FakeApplication(bot)  # noqa: SLF001

    files = [
        {
            "path": "out.bin",
            "size_bytes": 10,
            "content_b64": base64.b64encode(b"0123456789").decode("ascii"),
        }
    ]
    asyncio.run(adapter._send_done_files(3, files))  # noqa: SLF001

    assert bot.documents == []
    assert len(bot.messages) == 1
    assert "not sent" in bot.messages[0]["text"]
    assert bot.messages[0]["parse_mode"] == MARKDOWN_V2_PARSE_MODE


def test_on_text_message_starts_task_via_coordinator() -> None:
    coordinator = FakeCoordinator(start_status="started")
    adapter = TelegramAdapter(
        sandbox_factory=lambda: object(),
        coordinator=coordinator,  # type: ignore[arg-type]
        token="token",
    )

    async def scenario() -> None:
        await adapter._on_text_message(_update(42, "run task"), None)  # noqa: SLF001

    asyncio.run(scenario())

    assert len(coordinator.started_tasks) == 1
    assert coordinator.started_tasks[0]["session_id"] == "42"
    assert coordinator.started_tasks[0]["text"] == "run task"


def test_on_text_message_submits_pending_plan_feedback() -> None:
    coordinator = FakeCoordinator(start_status="started")
    coordinator.pending["42"] = "plan"
    adapter = TelegramAdapter(
        sandbox_factory=lambda: object(),
        coordinator=coordinator,  # type: ignore[arg-type]
        token="token",
    )

    async def scenario() -> None:
        await adapter._on_text_message(_update(42, "needs changes"), None)  # noqa: SLF001

    asyncio.run(scenario())

    assert coordinator.submitted_replies == [
        {"session_id": "42", "approved": False, "text": "needs changes"}
    ]
    assert coordinator.started_tasks == []


def test_on_text_message_reports_capacity() -> None:
    coordinator = FakeCoordinator(start_status="capacity")
    adapter = TelegramAdapter(
        sandbox_factory=lambda: object(),
        coordinator=coordinator,  # type: ignore[arg-type]
        token="token",
    )
    bot = FakeBot()
    adapter._application = FakeApplication(bot)  # noqa: SLF001

    async def scenario() -> None:
        await adapter._on_text_message(_update(2, "new task"), None)  # noqa: SLF001

    asyncio.run(scenario())

    assert len(bot.messages) == 1
    assert "at capacity" in bot.messages[0]["text"]


def test_on_text_message_submits_pending_plan_approval() -> None:
    coordinator = FakeCoordinator(start_status="started")
    coordinator.pending["11"] = "plan"
    adapter = TelegramAdapter(
        sandbox_factory=lambda: object(),
        coordinator=coordinator,  # type: ignore[arg-type]
        token="token",
    )

    async def scenario() -> None:
        await adapter._on_text_message(_update(11, "approve"), None)  # noqa: SLF001

    asyncio.run(scenario())
    assert coordinator.submitted_replies == [{"session_id": "11", "approved": True, "text": ""}]


def test_show_status_long_output_is_chunked_for_telegram_limit() -> None:
    adapter = TelegramAdapter(sandbox_factory=lambda: object(), token="token")
    bot = FakeBot()
    adapter._application = FakeApplication(bot)  # noqa: SLF001

    long_status = "x" * 9000
    asyncio.run(
        adapter.show(
            {"type": "message", "role": "status", "content": long_status},
            chat_id=5,
        )
    )

    assert len(bot.messages) >= 3
    assert all(len(message["text"]) <= 4096 for message in bot.messages)
    assert all(message["parse_mode"] == MARKDOWN_V2_PARSE_MODE for message in bot.messages)


def test_telegram_integration_task_plan_approve_execute_done(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["check shell"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(tool="shell", args={"command": "printf hi"}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(tool="agent_done", args={"reply": "complete"}),
                usage=None,
            ),
        ]
    )
    adapter = TelegramAdapter(
        sandbox_factory=lambda: YoloSandbox(
            skills_dir=str(_skills_root()),
            llm_factory=lambda _: llm,
            agent_config={"llm": {"model": "fake/model", "api_key": "fake-key"}},
        ),
        coordinator=Coordinator(
            sandbox_factory=lambda: YoloSandbox(
                skills_dir=str(_skills_root()),
                llm_factory=lambda _: llm,
                agent_config={"llm": {"model": "fake/model", "api_key": "fake-key"}},
            ),
            approval_mode="review",
            llm_config={"model": "fake/model", "api_key": "fake-key"},
            max_active_sessions=8,
        ),
        token="token",
        approval_mode="review",
        llm_config={"model": "fake/model", "api_key": "fake-key"},
    )
    bot = FakeBot()
    adapter._application = FakeApplication(bot)  # noqa: SLF001

    async def scenario() -> None:
        await adapter._on_text_message(_update(77, "run task"), None)  # noqa: SLF001
        for _ in range(400):
            if any(message["text"].startswith("*Plan*") for message in bot.messages):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Timed out waiting for plan approval prompt.")

        await adapter._on_text_message(_update(77, "approve"), None)  # noqa: SLF001

        for _ in range(400):
            if any(message["text"].startswith("*Success:* complete") for message in bot.messages):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Timed out waiting for Telegram session completion.")

    asyncio.run(scenario())

    text_payloads = [message["text"] for message in bot.messages]
    assert any(payload.startswith("*Plan*") for payload in text_payloads)
    assert any(payload.startswith("*Action:*") for payload in text_payloads)
    assert any(payload.startswith("*Success:* complete") for payload in text_payloads)
