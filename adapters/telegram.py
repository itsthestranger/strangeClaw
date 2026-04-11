"""Telegram adapter implementation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from adapters.session_persistence import persist_done_event, state_for_follow_up
from coordinator import Coordinator

TELEGRAM_IMPORT_ERROR: Exception | None = None
TelegramApplication: Any = None
TelegramCallbackQueryHandler: Any = None
TelegramInlineKeyboardButton: Any = None
TelegramInlineKeyboardMarkup: Any = None
TelegramMessageHandler: Any = None
TelegramChatAction: Any = None
telegram_filters: Any = None

try:
    from telegram import InlineKeyboardButton as TelegramInlineKeyboardButton
    from telegram import InlineKeyboardMarkup as TelegramInlineKeyboardMarkup
    from telegram.constants import ChatAction as TelegramChatAction
    from telegram.ext import Application as TelegramApplication
    from telegram.ext import CallbackQueryHandler as TelegramCallbackQueryHandler
    from telegram.ext import MessageHandler as TelegramMessageHandler
    from telegram.ext import filters as telegram_filters
except ImportError as exc:  # pragma: no cover - exercised in environments without telegram deps
    TELEGRAM_IMPORT_ERROR = exc


MAX_TELEGRAM_MESSAGE_CHARS = 4096
MAX_TELEGRAM_UPLOAD_BYTES = 50 * 1024 * 1024
MARKDOWN_V2_PARSE_MODE = "MarkdownV2"
TYPING_INTERVAL_SECONDS = 3.0
RECONNECT_BACKOFF_MIN_SECONDS = 1.0
RECONNECT_BACKOFF_MAX_SECONDS = 30.0
MARKDOWN_V2_SPECIAL_CHARS = "_*[]()~`>#+-=|{}.!"
PLAN_APPROVE_CALLBACK = "sc:plan:approve"
PLAN_REJECT_CALLBACK = "sc:plan:reject"
LOGGER = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class TelegramLimits:
    """Operational safety limits for Telegram adapter runtime behavior."""

    max_active_sessions: int = 8
    max_output_total_bytes: int = MAX_TELEGRAM_UPLOAD_BYTES
    max_output_file_bytes: int = 10 * 1024 * 1024


class TelegramAdapter:
    """Telegram interaction contract."""

    def __init__(
        self,
        *,
        sandbox_factory: Callable[[], Any],
        coordinator: Coordinator | None = None,
        approval_mode: str = "review",
        llm_config: dict[str, Any] | None = None,
        token: str,
        allowed_chat_ids: list[int] | None = None,
        limits: TelegramLimits | None = None,
        session_id_prefix: str = "",
    ) -> None:
        self._sandbox_factory = sandbox_factory
        self._approval_mode = approval_mode
        self._llm_config = llm_config
        self._token = token
        self._allowed_chat_ids = set(allowed_chat_ids or [])
        self._limits = limits or TelegramLimits()
        self._session_id_prefix = session_id_prefix
        self._coordinator = coordinator
        self._sessions: dict[int, ChatSession] = {}
        self._latest_state_by_chat: dict[int, dict[str, Any]] = {}
        self._application: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

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
            "session_id": self._session_id(chat_id),
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
                    await self._send_text(
                        chat_id,
                        text,
                        reply_markup=self._plan_keyboard(),
                        parse_mode=MARKDOWN_V2_PARSE_MODE,
                    )
                else:
                    await self._send_text(chat_id, text, parse_mode=MARKDOWN_V2_PARSE_MODE)
                return
            if role == "clarification":
                text = (
                    "*Clarification Needed*\n"
                    f"{self._escape_markdown_v2(self._to_text(content))}"
                )
                await self._send_text(chat_id, text, parse_mode=MARKDOWN_V2_PARSE_MODE)
                return
            if role == "status":
                text = f"*Status:* {self._escape_markdown_v2(self._to_text(content))}"
                await self._send_text(chat_id, text, parse_mode=MARKDOWN_V2_PARSE_MODE)
                return
            await self._send_text(
                chat_id,
                self._escape_markdown_v2(self._to_text(content)),
                parse_mode=MARKDOWN_V2_PARSE_MODE,
            )
            return

        if event_type == "action":
            await self._send_text(
                chat_id,
                self._format_action(event),
                parse_mode=MARKDOWN_V2_PARSE_MODE,
            )
            return

        if event_type == "done":
            success = bool(event.get("success"))
            label = "Success" if success else "Failed"
            reply = self._escape_markdown_v2(self._to_text(event.get("reply")))
            await self._send_text(
                chat_id,
                f"*{label}:* {reply}",
                parse_mode=MARKDOWN_V2_PARSE_MODE,
            )
            await self._send_done_files(chat_id, event.get("files"))
            return

        await self._send_text(
            chat_id,
            self._escape_markdown_v2(self._to_text(event)),
            parse_mode=MARKDOWN_V2_PARSE_MODE,
        )

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

        backoff_seconds = RECONNECT_BACKOFF_MIN_SECONDS
        while True:
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
                return
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                LOGGER.warning(
                    "Telegram polling failed (%s). Retrying in %.1f seconds.",
                    exc,
                    backoff_seconds,
                )
                time.sleep(backoff_seconds)
                backoff_seconds = min(
                    backoff_seconds * 2.0,
                    RECONNECT_BACKOFF_MAX_SECONDS,
                )
            finally:
                self._application = None

    def stop(self) -> None:
        """Best-effort stop for run_polling loop."""
        application = self._application
        if application is None:
            return
        stop_running = getattr(application, "stop_running", None)
        if callable(stop_running):
            try:
                stop_running()
            except Exception:
                return

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

        if self._coordinator is not None:
            await self._on_text_message_with_coordinator(chat_id=chat_id, text=text)
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

        if self._active_session_count() >= self._limits.max_active_sessions:
            await self._send_text(
                chat_id,
                (
                    "The bot is at capacity right now. "
                    "Please retry after one of the active tasks finishes."
                ),
            )
            return

        task_event = self.get_task(update)
        previous_state = self._latest_state_by_chat.get(chat_id)
        if previous_state is not None:
            task_event["state"] = state_for_follow_up(previous_state)
        sandbox = self._sandbox_factory()
        session = ChatSession(
            chat_id=chat_id,
            session_id=self._session_id(chat_id),
            sandbox=sandbox,
        )
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

        if self._coordinator is not None:
            await self._on_plan_callback_with_coordinator(
                chat_id=chat_id,
                data=data,
                query=query,
            )
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
        last_typing_time = 0.0
        try:
            await asyncio.to_thread(session.sandbox.run, task_event)
            while True:
                event = await asyncio.to_thread(session.sandbox.receive, 0.2)
                if event is None:
                    now = time.monotonic()
                    if now - last_typing_time >= TYPING_INTERVAL_SECONDS:
                        await self._send_typing(chat_id)
                        last_typing_time = now
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
                    await self._persist_chat_done_state(chat_id=chat_id, done_event=event)
                    return
        except Exception as exc:
            await self._send_text(
                chat_id,
                f"*Task Failed:* {self._escape_markdown_v2(str(exc))}",
                parse_mode=MARKDOWN_V2_PARSE_MODE,
            )
        finally:
            self._cancel_pending_reply(session)
            await asyncio.to_thread(session.sandbox.stop)
            self._sessions.pop(chat_id, None)

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

    def _active_session_count(self) -> int:
        count = 0
        for item in self._sessions.values():
            if item.runner is not None and not item.runner.done():
                count += 1
        return count

    async def _persist_chat_done_state(self, *, chat_id: int, done_event: dict[str, Any]) -> None:
        state = done_event.get("state")
        if isinstance(state, dict):
            self._latest_state_by_chat[chat_id] = state
        session = self._sessions.get(chat_id)
        session_id = session.session_id if session is not None else self._session_id(chat_id)
        try:
            await asyncio.to_thread(
                persist_done_event,
                session_id=session_id,
                done_event=done_event,
            )
        except ValueError as exc:
            await self._send_text(
                chat_id,
                f"Failed to persist session outputs: {self._escape_markdown_v2(str(exc))}",
                parse_mode=MARKDOWN_V2_PARSE_MODE,
            )

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

        total_bytes = 0
        skipped_for_size = False
        bot = self._require_bot()
        for item in files:
            if not isinstance(item, dict):
                continue
            rel_path = item.get("path")
            content_b64 = item.get("content_b64")
            if not isinstance(rel_path, str) or not isinstance(content_b64, str):
                continue

            size_bytes = self._declared_or_estimated_size(item=item, content_b64=content_b64)
            if size_bytes > self._limits.max_output_file_bytes:
                skipped_for_size = True
                continue
            if total_bytes + size_bytes > self._limits.max_output_total_bytes:
                skipped_for_size = True
                break

            try:
                decoded = base64.b64decode(content_b64, validate=True)
            except binascii.Error:
                continue
            if len(decoded) > self._limits.max_output_file_bytes:
                skipped_for_size = True
                continue
            if total_bytes + len(decoded) > self._limits.max_output_total_bytes:
                skipped_for_size = True
                break

            total_bytes += len(decoded)
            file_name = Path(rel_path).name or "output.bin"
            stream = BytesIO(decoded)
            stream.name = file_name
            await bot.send_document(chat_id=chat_id, document=stream, caption=f"Output: {rel_path}")

        if skipped_for_size:
            limit_mb = self._limits.max_output_total_bytes // (1024 * 1024)
            file_mb = self._limits.max_output_file_bytes // (1024 * 1024)
            await self._send_text(
                chat_id,
                (
                    "Some output files were not sent because Telegram safety limits were exceeded "
                    f"\\(max {file_mb} MB per file, {limit_mb} MB total\\)\\."
                ),
                parse_mode=MARKDOWN_V2_PARSE_MODE,
            )

    async def _send_text(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: Any | None = None,
        parse_mode: str | None = None,
    ) -> None:
        bot = self._require_bot()
        if parse_mode == MARKDOWN_V2_PARSE_MODE:
            chunks = self._chunk_markdown_text(text, MAX_TELEGRAM_MESSAGE_CHARS)
        else:
            chunks = self._chunk_text(text, MAX_TELEGRAM_MESSAGE_CHARS)
        for index, chunk in enumerate(chunks):
            markup = reply_markup if index == 0 else None
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    reply_markup=markup,
                    parse_mode=parse_mode,
                )
            except Exception:
                if parse_mode is None:
                    raise
                await bot.send_message(chat_id=chat_id, text=chunk, reply_markup=markup)

    async def _send_typing(self, chat_id: int) -> None:
        bot = self._require_bot()
        action = "typing"
        if TelegramChatAction is not None:
            action = TelegramChatAction.TYPING
        try:
            await bot.send_chat_action(chat_id=chat_id, action=action)
        except Exception:
            # Best effort; if chat action fails we continue normal event handling.
            return

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

    @classmethod
    def _chunk_markdown_text(cls, text: str, limit: int) -> list[str]:
        if limit <= 0:
            raise ValueError("Message chunk size limit must be positive.")
        if not text:
            return [""]
        if len(text) <= limit:
            return [text]

        chunks: list[str] = []
        remaining = text
        while len(remaining) > limit:
            split_at = cls._find_markdown_split(remaining, limit)
            chunk = remaining[:split_at]
            if not chunk:
                chunk = remaining[:limit]
                split_at = len(chunk)
            chunks.append(chunk)
            remaining = remaining[split_at:]
            if remaining.startswith("\n"):
                remaining = remaining[1:]
        if remaining:
            chunks.append(remaining)
        return chunks

    @staticmethod
    def _find_markdown_split(text: str, limit: int) -> int:
        pos = text.rfind("\n", 0, limit)
        while pos > 0:
            candidate = text[:pos]
            if candidate.count("```") % 2 == 0:
                return pos
            pos = text.rfind("\n", 0, pos)
        fallback = text.rfind("\n", 0, limit)
        if fallback > 0:
            return fallback
        return limit

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
            lines: list[str] = ["*Plan*"]
            if isinstance(goal, str) and goal.strip():
                lines.append(f"*Goal:* {TelegramAdapter._escape_markdown_v2(goal.strip())}")
            if isinstance(steps, list):
                for index, step in enumerate(steps, start=1):
                    if isinstance(step, str):
                        lines.append(
                            f"{index}\\. {TelegramAdapter._escape_markdown_v2(step)}"
                        )
            return "\n".join(lines)
        return (
            "*Plan*\n"
            f"{TelegramAdapter._escape_markdown_v2(TelegramAdapter._to_text(content))}"
        )

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
                preview = stdout.strip()
            elif isinstance(stderr, str) and stderr.strip():
                preview = stderr.strip()

        header = f"*Action:* {TelegramAdapter._escape_markdown_v2(f'{skill}.{action}{status}')}"
        if preview:
            preview = preview[:3000]
            escaped_preview = TelegramAdapter._escape_markdown_v2_code(preview)
            return f"{header}\n```text\n{escaped_preview}\n```"
        return header

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

    @staticmethod
    def _declared_or_estimated_size(*, item: dict[str, Any], content_b64: str) -> int:
        declared = item.get("size_bytes")
        if isinstance(declared, int) and declared >= 0:
            return declared
        # Approximate decoded size from base64 length and trailing '=' padding.
        padding = 0
        if content_b64.endswith("=="):
            padding = 2
        elif content_b64.endswith("="):
            padding = 1
        return max(0, (len(content_b64) * 3) // 4 - padding)

    def _session_id(self, chat_id: int) -> str:
        return f"{self._session_id_prefix}{chat_id}"

    async def _on_text_message_with_coordinator(self, *, chat_id: int, text: str) -> None:
        coordinator = self._coordinator
        if coordinator is None:
            return
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        session_id = self._session_id(chat_id)
        pending = coordinator.pending_role(session_id=session_id)
        if pending is not None:
            approved = pending != "plan"
            submitted = coordinator.submit_reply(
                session_id=session_id,
                approved=approved,
                text=text,
            )
            if not submitted:
                await self._send_text(
                    chat_id,
                    "No prompt is waiting for a reply right now.",
                )
            return

        def sink(event: dict[str, Any], *, cid: int = chat_id) -> None:
            self._emit_from_coordinator(chat_id=cid, event=event)

        status = coordinator.start_task(
            session_id=session_id,
            text=text,
            sink=sink,
        )
        if status == "busy":
            await self._send_text(
                chat_id,
                "A task is already running. Wait for completion or answer the active prompt.",
            )
            return
        if status == "capacity":
            await self._send_text(
                chat_id,
                (
                    "The bot is at capacity right now. "
                    "Please retry after one of the active tasks finishes."
                ),
            )
            return

    async def _on_plan_callback_with_coordinator(
        self,
        *,
        chat_id: int,
        data: str,
        query: Any,
    ) -> None:
        coordinator = self._coordinator
        if coordinator is None:
            await query.answer()
            return

        session_id = self._session_id(chat_id)
        pending = coordinator.pending_role(session_id=session_id)
        if pending != "plan":
            await query.answer(text="No plan is awaiting approval.")
            return

        if data == PLAN_APPROVE_CALLBACK:
            coordinator.submit_reply(session_id=session_id, approved=True, text="")
            await query.answer(text="Plan approved.")
            return

        if data == PLAN_REJECT_CALLBACK:
            coordinator.submit_reply(session_id=session_id, approved=False, text="")
            await query.answer(text="Plan rejected.")
            return

        await query.answer()

    def _emit_from_coordinator(self, *, chat_id: int, event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self.show(event, chat_id=chat_id), loop)
        future.add_done_callback(_consume_future_exception)

    @staticmethod
    def _escape_markdown_v2(text: str) -> str:
        escaped: list[str] = []
        for char in text:
            if char in MARKDOWN_V2_SPECIAL_CHARS:
                escaped.append(f"\\{char}")
            else:
                escaped.append(char)
        return "".join(escaped)

    @staticmethod
    def _escape_markdown_v2_code(text: str) -> str:
        return text.replace("\\", "\\\\").replace("`", "\\`")


def _consume_future_exception(future: Any) -> None:
    try:
        future.result()
    except Exception:
        return
