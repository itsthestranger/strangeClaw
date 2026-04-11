"""Tests for TelegramAdapter core behavior."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import adapters.telegram as telegram_module
from adapters.telegram import (
    MARKDOWN_V2_PARSE_MODE,
    PLAN_APPROVE_CALLBACK,
    ChatSession,
    TelegramAdapter,
    TelegramLimits,
)
from agent.llm import LLMResponse, ToolCall
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


class FakeCallbackQuery:
    """Callback query test double."""

    def __init__(self, *, chat_id: int, data: str) -> None:
        self.message = SimpleNamespace(chat=SimpleNamespace(id=chat_id))
        self.data = data
        self.answers: list[dict[str, str]] = []

    async def answer(self, text: str = "") -> None:
        self.answers.append({"text": text})


class ScriptedLLM:
    """Deterministic LLM fixture for Telegram integration tests."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = responses

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | None = None,
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


def _callback_update(chat_id: int, data: str) -> Any:
    return SimpleNamespace(callback_query=FakeCallbackQuery(chat_id=chat_id, data=data))


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
        "llm": {"model": "x", "api_key": "k"},
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


def test_plan_feedback_text_resolves_pending_reply() -> None:
    adapter = TelegramAdapter(sandbox_factory=lambda: object(), token="token")
    session = ChatSession(chat_id=1, session_id="1", sandbox=object())
    adapter._sessions[1] = session  # noqa: SLF001

    async def scenario() -> dict[str, Any]:
        waiter = asyncio.create_task(adapter.get_reply("plan", chat_id=1))
        await asyncio.sleep(0)
        assert adapter._resolve_text_reply(session, "needs changes")  # noqa: SLF001
        return await waiter

    reply = asyncio.run(scenario())
    assert reply == {"approved": False, "text": "needs changes"}


def test_chunk_markdown_text_prefers_code_fence_boundaries() -> None:
    text = "*Action:* shell\n```text\nline1\nline2\n```\nend"
    chunks = TelegramAdapter._chunk_markdown_text(text, limit=24)
    assert len(chunks) >= 2
    assert all(chunk.count("```") % 2 == 0 for chunk in chunks[:-1])


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


def test_run_chat_session_reports_failure_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingSandbox:
        def __init__(self) -> None:
            self.stop_calls = 0

        def run(self, task: dict[str, Any]) -> None:
            del task

        def receive(self, timeout: float) -> dict[str, Any] | None:
            del timeout
            raise RuntimeError("sandbox crashed")

        def stop(self) -> None:
            self.stop_calls += 1

    sandbox = FailingSandbox()
    adapter = TelegramAdapter(sandbox_factory=lambda: sandbox, token="token")
    bot = FakeBot()
    adapter._application = FakeApplication(bot)  # noqa: SLF001
    session = ChatSession(chat_id=9, session_id="9", sandbox=sandbox)
    adapter._sessions[9] = session  # noqa: SLF001

    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(telegram_module.asyncio, "to_thread", fake_to_thread)

    asyncio.run(adapter._run_chat_session(session, {"approval_mode": "review"}))  # noqa: SLF001

    assert sandbox.stop_calls == 1
    assert 9 not in adapter._sessions  # noqa: SLF001
    assert len(bot.messages) == 1
    assert bot.messages[0]["parse_mode"] == MARKDOWN_V2_PARSE_MODE
    assert "Task Failed" in bot.messages[0]["text"]


def test_on_text_message_includes_follow_up_state(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = TelegramAdapter(sandbox_factory=lambda: object(), token="token")
    adapter._latest_state_by_chat[42] = {  # noqa: SLF001
        "goal": "g",
        "plan": {"steps": ["old"]},
        "history": [{"type": "action"}],
    }
    captured_task: dict[str, Any] = {}

    async def fake_run_chat_session(
        session: ChatSession,
        task_event: dict[str, Any],
    ) -> None:
        del session
        captured_task.update(task_event)

    monkeypatch.setattr(adapter, "_run_chat_session", fake_run_chat_session)

    async def scenario() -> None:
        await adapter._on_text_message(_update(42, "follow-up"), None)  # noqa: SLF001
        await asyncio.sleep(0)

    asyncio.run(scenario())

    assert captured_task["text"] == "follow-up"
    assert captured_task["state"] == {"goal": "g", "history": [{"type": "action"}]}
    assert "plan" not in captured_task["state"]


def test_on_text_message_respects_max_active_sessions() -> None:
    adapter = TelegramAdapter(
        sandbox_factory=lambda: object(),
        token="token",
        limits=TelegramLimits(max_active_sessions=1),
    )
    bot = FakeBot()
    adapter._application = FakeApplication(bot)  # noqa: SLF001

    class RunningTask:
        @staticmethod
        def done() -> bool:
            return False

    adapter._sessions[1] = ChatSession(  # noqa: SLF001
        chat_id=1,
        session_id="1",
        sandbox=object(),
        runner=cast(Any, RunningTask()),
    )

    async def scenario() -> None:
        await adapter._on_text_message(_update(2, "new task"), None)  # noqa: SLF001

    asyncio.run(scenario())

    assert len(bot.messages) == 1
    assert "at capacity" in bot.messages[0]["text"]
    assert 2 not in adapter._sessions  # noqa: SLF001


def test_plan_callback_approve_resolves_pending_reply() -> None:
    adapter = TelegramAdapter(sandbox_factory=lambda: object(), token="token")
    session = ChatSession(chat_id=11, session_id="11", sandbox=object())
    adapter._sessions[11] = session  # noqa: SLF001

    async def scenario() -> tuple[dict[str, Any], FakeCallbackQuery]:
        waiter = asyncio.create_task(adapter.get_reply("plan", chat_id=11))
        await asyncio.sleep(0)
        callback_update = _callback_update(11, PLAN_APPROVE_CALLBACK)
        callback_query = cast(FakeCallbackQuery, callback_update.callback_query)
        await adapter._on_plan_callback(callback_update, None)  # noqa: SLF001
        return await waiter, callback_query

    reply, callback_query = asyncio.run(scenario())
    assert reply == {"approved": True, "text": ""}
    assert callback_query.answers == [{"text": "Plan approved."}]


def test_concurrent_pending_replies_do_not_cross_contaminate() -> None:
    adapter = TelegramAdapter(sandbox_factory=lambda: object(), token="token")
    adapter._sessions[1] = ChatSession(chat_id=1, session_id="1", sandbox=object())  # noqa: SLF001
    adapter._sessions[2] = ChatSession(chat_id=2, session_id="2", sandbox=object())  # noqa: SLF001

    async def scenario() -> tuple[dict[str, Any], dict[str, Any]]:
        waiter_one = asyncio.create_task(adapter.get_reply("clarification", chat_id=1))
        waiter_two = asyncio.create_task(adapter.get_reply("clarification", chat_id=2))
        await asyncio.sleep(0)

        await adapter._on_text_message(_update(1, "alpha"), None)  # noqa: SLF001
        reply_one = await asyncio.wait_for(waiter_one, timeout=1.0)
        assert waiter_two.done() is False

        await adapter._on_text_message(_update(2, "beta"), None)  # noqa: SLF001
        reply_two = await asyncio.wait_for(waiter_two, timeout=1.0)
        return reply_one, reply_two

    reply_one, reply_two = asyncio.run(scenario())
    assert reply_one == {"approved": True, "text": "alpha"}
    assert reply_two == {"approved": True, "text": "beta"}


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
    async def fake_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr(telegram_module.asyncio, "to_thread", fake_to_thread)

    llm = ScriptedLLM(
        responses=[
            LLMResponse(text='{"steps":["check shell"]}', action=None, usage=None),
            LLMResponse(
                text="",
                action=ToolCall(skill="shell", action="run", args={"command": "printf hi"}),
                usage=None,
            ),
            LLMResponse(
                text="",
                action=ToolCall(skill="__agent__", action="done", args={"reply": "complete"}),
                usage=None,
            ),
        ]
    )
    adapter = TelegramAdapter(
        sandbox_factory=lambda: YoloSandbox(
            skills_dir=str(_skills_root()),
            llm_factory=lambda _: llm,
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
            session = adapter._sessions.get(77)  # noqa: SLF001
            if (
                session is not None
                and session.pending_reply is not None
                and session.pending_reply.role == "plan"
            ):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Timed out waiting for plan approval prompt.")

        callback_update = _callback_update(77, PLAN_APPROVE_CALLBACK)
        await adapter._on_plan_callback(callback_update, None)  # noqa: SLF001

        for _ in range(400):
            if 77 not in adapter._sessions:  # noqa: SLF001
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("Timed out waiting for Telegram session completion.")

    asyncio.run(scenario())

    text_payloads = [message["text"] for message in bot.messages]
    assert any(payload.startswith("*Plan*") for payload in text_payloads)
    assert any(payload.startswith("*Action:*") for payload in text_payloads)
    assert any(payload.startswith("*Success:* complete") for payload in text_payloads)
