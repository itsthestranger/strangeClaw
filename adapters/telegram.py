"""Telegram adapter implementation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

TELEGRAM_IMPORT_ERROR: Exception | None = None
TelegramApplication: Any = None
TelegramCallbackQueryHandler: Any = None
TelegramInlineKeyboardButton: Any = None
TelegramInlineKeyboardMarkup: Any = None
TelegramMessageHandler: Any = None
telegram_filters: Any = None

try:
    from telegram import InlineKeyboardButton as TelegramInlineKeyboardButton
    from telegram import InlineKeyboardMarkup as TelegramInlineKeyboardMarkup
    from telegram.ext import Application as TelegramApplication
    from telegram.ext import CallbackQueryHandler as TelegramCallbackQueryHandler
    from telegram.ext import MessageHandler as TelegramMessageHandler
    from telegram.ext import filters as telegram_filters
except ImportError as exc:  # pragma: no cover - exercised in environments without telegram deps
    TELEGRAM_IMPORT_ERROR = exc


MAX_TELEGRAM_MESSAGE_CHARS = 4096
PLAN_APPROVE_CALLBACK = "sc:plan:approve"
PLAN_REJECT_CALLBACK = "sc:plan:reject"


@dataclass
class PendingReply:
    """A waiting user reply for an active chat session."""

    role: str
    future: asyncio.Future[dict[str, Any]]


@dataclass
class ChatSession:
    """Runtime state for one chat."""

    chat_id: int
    session_id: str
    sandbox: Any
    runner: asyncio.Task[None] | None = None
    pending_reply: PendingReply | None = None


class TelegramAdapter:
    """Telegram interaction contract."""

    def __init__(
        self,
        *,
        sandbox_factory: Callable[[], Any],
        approval_mode: str = "review",
        llm_config: dict[str, Any] | None = None,
        token: str,
        allowed_chat_ids: list[int] | None = None,
    ) -> None:
        self._sandbox_factory = sandbox_factory
        self._approval_mode = approval_mode
        self._llm_config = llm_config
        self._token = token
        self._allowed_chat_ids = set(allowed_chat_ids or [])
        self._sessions: dict[int, ChatSession] = {}
        self._application: Any | None = None

    def get_task(self, update: Any | None = None) -> dict[str, Any]:
        """Build a task payload from an incoming Telegram update."""
        if update is None:
            raise ValueError("update is required.")

        message = getattr(update, "effective_message", None)
        text = getattr(message, "text", None)
        chat = getattr(update, "effective_chat", None)
        chat_id = getattr(chat, "id", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Task text cannot be empty.")
        if not isinstance(chat_id, int):
            raise ValueError("Missing chat id on Telegram update.")

        task: dict[str, Any] = {
            "type": "task",
            "text": text.strip(),
            "session_id": str(chat_id),
            "approval_mode": self._approval_mode,
        }
        if self._llm_config is not None:
            task["llm"] = dict(self._llm_config)
        return task

    async def show(self, event: dict[str, Any], *, chat_id: int) -> None:
        """Display an agent event in the Telegram chat."""
        event_type = event.get("type")
        if event_type == "message":
            role = event.get("role")
            content = event.get("content")
            if role == "plan":
                text = self._format_plan(content)
                if self._approval_mode == "review":
                    await self._send_text(chat_id, text, reply_markup=self._plan_keyboard())
                else:
                    await self._send_text(chat_id, text)
                return
            if role == "clarification":
                await self._send_text(chat_id, f"Clarification needed:\n{self._to_text(content)}")
                return
            if role == "status":
                await self._send_text(chat_id, f"Status: {self._to_text(content)}")
                return
            await self._send_text(chat_id, self._to_text(content))
            return

        if event_type == "action":
            await self._send_text(chat_id, self._format_action(event))
            return

        if event_type == "done":
            success = bool(event.get("success"))
            label = "Success" if success else "Failed"
            reply = self._to_text(event.get("reply"))
            await self._send_text(chat_id, f"{label}: {reply}")
            await self._send_done_files(chat_id, event.get("files"))
            return

        await self._send_text(chat_id, self._to_text(event))

    async def get_reply(self, role: str, *, chat_id: int) -> dict[str, Any]:
        """Wait for a user reply for plan review or clarification."""
        if role not in {"plan", "clarification"}:
            raise ValueError(f"Unsupported reply role: {role}")

        session = self._require_session(chat_id)
        if session.pending_reply is not None:
            raise RuntimeError("Chat already has a pending reply.")

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        session.pending_reply = PendingReply(role=role, future=future)
        try:
            return await future
        finally:
            if session.pending_reply is not None and session.pending_reply.future is future:
                session.pending_reply = None

    def run(self) -> None:
        """Drive the Telegram adapter loop."""
        if TELEGRAM_IMPORT_ERROR is not None:
            raise RuntimeError(
                "python-telegram-bot is not installed. Install runtime dependencies first."
            ) from TELEGRAM_IMPORT_ERROR
        if not self._token.strip():
            raise ValueError("telegram.token must be set when adapter=telegram.")
        if (
            TelegramApplication is None
            or TelegramMessageHandler is None
            or TelegramCallbackQueryHandler is None
            or telegram_filters is None
        ):
            raise RuntimeError("Telegram runtime components are unavailable.")

        application = TelegramApplication.builder().token(self._token).build()
        self._application = application

        text_filter = telegram_filters.TEXT & ~telegram_filters.COMMAND
        application.add_handler(TelegramMessageHandler(text_filter, self._on_text_message))
        application.add_handler(
            TelegramCallbackQueryHandler(
                self._on_plan_callback,
                pattern=r"^sc:plan:(approve|reject)$",
            )
        )

        try:
            application.run_polling(drop_pending_updates=True)
        finally:
            self._application = None

    async def _on_text_message(self, update: Any, context: Any) -> None:
        del context
        chat = getattr(update, "effective_chat", None)
        message = getattr(update, "effective_message", None)
        chat_id = getattr(chat, "id", None)
        text = getattr(message, "text", None)
        if not isinstance(chat_id, int) or not isinstance(text, str):
            return

        text = text.strip()
        if not text:
            return

        if not await self._ensure_authorized(chat_id):
            return

        existing = self._sessions.get(chat_id)
        if existing is not None and existing.pending_reply is not None:
            if self._resolve_text_reply(existing, text):
                return

        if existing is not None and existing.runner is not None and not existing.runner.done():
            await self._send_text(
                chat_id,
                "A task is already running. Wait for completion or answer the active prompt.",
            )
            return

        task_event = self.get_task(update)
        sandbox = self._sandbox_factory()
        session = ChatSession(chat_id=chat_id, session_id=str(chat_id), sandbox=sandbox)
        self._sessions[chat_id] = session
        session.runner = asyncio.create_task(self._run_chat_session(session, task_event))

    async def _on_plan_callback(self, update: Any, context: Any) -> None:
        del context
        query = getattr(update, "callback_query", None)
        if query is None:
            return

        message = getattr(query, "message", None)
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None)
        data = getattr(query, "data", "")
        if not isinstance(chat_id, int) or not isinstance(data, str):
            await query.answer()
            return

        if not await self._ensure_authorized(chat_id):
            await query.answer(text="Not authorized.")
            return

        session = self._sessions.get(chat_id)
        if session is None or session.pending_reply is None or session.pending_reply.role != "plan":
            await query.answer(text="No plan is awaiting approval.")
            return

        future = session.pending_reply.future
        if future.done():
            await query.answer()
            return

        if data == PLAN_APPROVE_CALLBACK:
            future.set_result({"approved": True, "text": ""})
            await query.answer(text="Plan approved.")
            return

        if data == PLAN_REJECT_CALLBACK:
            future.set_result({"approved": False, "text": ""})
            await query.answer(text="Plan rejected.")
            return

        await query.answer()

    async def _run_chat_session(self, session: ChatSession, task_event: dict[str, Any]) -> None:
        chat_id = session.chat_id
        try:
            await asyncio.to_thread(session.sandbox.run, task_event)
            while True:
                event = await asyncio.to_thread(session.sandbox.receive, 0.2)
                if event is None:
                    await asyncio.sleep(0)
                    continue

                await self.show(event, chat_id=chat_id)

                event_type = event.get("type")
                if event_type == "message":
                    role = event.get("role")
                    if role == "plan" and task_event.get("approval_mode") == "review":
                        reply = await self.get_reply("plan", chat_id=chat_id)
                        await asyncio.to_thread(
                            session.sandbox.send,
                            {"type": "user_reply", **reply},
                        )
                    elif role == "clarification":
                        reply = await self.get_reply("clarification", chat_id=chat_id)
                        await asyncio.to_thread(
                            session.sandbox.send,
                            {"type": "user_reply", **reply},
                        )

                if event_type == "done":
                    return
        except Exception as exc:
            await self._send_text(chat_id, f"Task failed: {exc}")
        finally:
            self._cancel_pending_reply(session)
            await asyncio.to_thread(session.sandbox.stop)

    def _cancel_pending_reply(self, session: ChatSession) -> None:
        pending = session.pending_reply
        if pending is None:
            return
        if pending.future.done():
            session.pending_reply = None
            return

        if pending.role == "plan":
            pending.future.set_result({"approved": False, "text": ""})
        else:
            pending.future.set_result({"approved": True, "text": ""})
        session.pending_reply = None

    def _resolve_text_reply(self, session: ChatSession, text: str) -> bool:
        pending = session.pending_reply
        if pending is None or pending.future.done():
            return False

        if pending.role == "plan":
            pending.future.set_result({"approved": False, "text": text})
            return True

        if pending.role == "clarification":
            pending.future.set_result({"approved": True, "text": text})
            return True

        return False

    async def _send_done_files(self, chat_id: int, files: Any) -> None:
        if not isinstance(files, list):
            return

        bot = self._require_bot()
        for item in files:
            if not isinstance(item, dict):
                continue
            rel_path = item.get("path")
            content_b64 = item.get("content_b64")
            if not isinstance(rel_path, str) or not isinstance(content_b64, str):
                continue
            try:
                decoded = base64.b64decode(content_b64, validate=True)
            except binascii.Error:
                continue

            file_name = Path(rel_path).name or "output.bin"
            stream = BytesIO(decoded)
            stream.name = file_name
            await bot.send_document(chat_id=chat_id, document=stream, caption=f"Output: {rel_path}")

    async def _send_text(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: Any | None = None,
    ) -> None:
        bot = self._require_bot()
        chunks = self._chunk_text(text, MAX_TELEGRAM_MESSAGE_CHARS)
        for index, chunk in enumerate(chunks):
            markup = reply_markup if index == 0 else None
            await bot.send_message(chat_id=chat_id, text=chunk, reply_markup=markup)

    def _require_bot(self) -> Any:
        if self._application is None:
            raise RuntimeError("Telegram application is not initialized.")
        return self._application.bot

    @staticmethod
    def _chunk_text(text: str, limit: int) -> list[str]:
        if limit <= 0:
            raise ValueError("Message chunk size limit must be positive.")
        if not text:
            return [""]
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        remaining = text
        while len(remaining) > limit:
            split_at = remaining.rfind("\n", 0, limit)
            if split_at <= 0:
                split_at = limit
            chunk = remaining[:split_at]
            chunks.append(chunk)
            remaining = remaining[split_at:]
            if remaining.startswith("\n"):
                remaining = remaining[1:]
        if remaining:
            chunks.append(remaining)
        return chunks

    @staticmethod
    def _plan_keyboard() -> Any:
        if TelegramInlineKeyboardMarkup is None or TelegramInlineKeyboardButton is None:
            return None
        return TelegramInlineKeyboardMarkup(
            [
                [
                    TelegramInlineKeyboardButton("Approve", callback_data=PLAN_APPROVE_CALLBACK),
                    TelegramInlineKeyboardButton("Reject", callback_data=PLAN_REJECT_CALLBACK),
                ]
            ]
        )

    @staticmethod
    def _format_plan(content: Any) -> str:
        if isinstance(content, dict):
            goal = content.get("goal")
            steps = content.get("steps")
            lines: list[str] = ["Plan:"]
            if isinstance(goal, str) and goal.strip():
                lines.append(f"Goal: {goal.strip()}")
            if isinstance(steps, list):
                for index, step in enumerate(steps, start=1):
                    if isinstance(step, str):
                        lines.append(f"{index}. {step}")
            return "\n".join(lines)
        return f"Plan:\n{TelegramAdapter._to_text(content)}"

    @staticmethod
    def _format_action(event: dict[str, Any]) -> str:
        skill = event.get("skill", "unknown")
        action = event.get("action", "unknown")
        result = event.get("result")

        status = ""
        preview = ""
        if isinstance(result, dict):
            exit_code = result.get("exit_code")
            if isinstance(exit_code, int):
                status = f" (exit={exit_code})"
            stdout = result.get("stdout")
            stderr = result.get("stderr")
            if isinstance(stdout, str) and stdout.strip():
                preview = stdout.strip().splitlines()[0]
            elif isinstance(stderr, str) and stderr.strip():
                preview = stderr.strip().splitlines()[0]

        if preview:
            preview = preview[:200]
            return f"Action: {skill}.{action}{status}\n{preview}"
        return f"Action: {skill}.{action}{status}"

    def _require_session(self, chat_id: int) -> ChatSession:
        session = self._sessions.get(chat_id)
        if session is None:
            raise RuntimeError(f"No active session for chat {chat_id}.")
        return session

    async def _ensure_authorized(self, chat_id: int) -> bool:
        if not self._allowed_chat_ids or chat_id in self._allowed_chat_ids:
            return True
        await self._send_text(chat_id, "You are not authorized to use this bot.")
        return False

    @staticmethod
    def _to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True, indent=2)
