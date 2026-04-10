"""Tests for TelegramAdapter core behavior."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any

from adapters.telegram import ChatSession, TelegramAdapter


class FakeBot:
    """Async bot stub for adapter tests."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.documents: list[dict[str, Any]] = []

    async def send_message(self, *, chat_id: int, text: str, reply_markup: Any = None) -> None:
        self.messages.append({"chat_id": chat_id, "text": text, "reply_markup": reply_markup})

    async def send_document(self, *, chat_id: int, document: Any, caption: str = "") -> None:
        self.documents.append(
            {
                "chat_id": chat_id,
                "caption": caption,
                "name": getattr(document, "name", ""),
            }
        )


class FakeApplication:
    """Application wrapper exposing bot attribute."""

    def __init__(self, bot: FakeBot) -> None:
        self.bot = bot


def _update(chat_id: int, text: str) -> Any:
    return SimpleNamespace(
        effective_chat=SimpleNamespace(id=chat_id),
        effective_message=SimpleNamespace(text=text),
    )


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

    assert bot.messages == [{"chat_id": 7, "text": "Success: finished", "reply_markup": None}]
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
