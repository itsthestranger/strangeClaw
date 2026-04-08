"""CLI adapter."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import session


class CLIAdapter:
    """CLI interaction contract."""

    def __init__(
        self,
        *,
        sandbox: Any,
        approval_mode: str = "review",
        llm_config: dict[str, Any] | None = None,
        resume_session_id: str | None = None,
        resume_state: dict[str, Any] | None = None,
        input_func: Callable[[str], str] = input,
    ) -> None:
        self._sandbox = sandbox
        self._approval_mode = approval_mode
        self._llm_config = llm_config
        self._resume_session_id = resume_session_id
        self._resume_state = resume_state
        self._input = input_func

    def get_task(self) -> dict[str, Any]:
        """Get initial task from the user."""
        if self._resume_state is not None:
            goal = self._resume_state.get("goal")
            text = goal if isinstance(goal, str) and goal.strip() else "Continue previous session"
            session_id = self._resume_session_id or str(uuid4())
        else:
            text = self._input("Task: ").strip()
            if not text:
                raise ValueError("Task cannot be empty.")
            session_id = str(uuid4())

        task: dict[str, Any] = {
            "type": "task",
            "text": text,
            "session_id": session_id,
            "approval_mode": self._approval_mode,
        }
        if self._llm_config is not None:
            task["llm"] = self._llm_config
        if self._resume_state is not None:
            task["state"] = self._resume_state
        return task

    def show(self, event: dict[str, Any]) -> None:
        """Display an agent event."""
        event_type = event.get("type")
        if event_type == "message":
            role = event.get("role")
            content = self._format_content(event.get("content"))
            if role == "plan":
                print("\nPlan:")
                print(content)
                return
            if role == "clarification":
                print(f"\nClarification: {content}")
                return
            if role == "status":
                print(f"\nStatus: {content}")
                return
            print(f"\nMessage: {content}")
            return

        if event_type == "action":
            skill = event.get("skill", "unknown")
            action = event.get("action", "unknown")
            result = event.get("result", {})
            exit_code = result.get("exit_code")
            print(f"\nAction: {skill}.{action} (exit={exit_code})")
            return

        if event_type == "done":
            success = event.get("success")
            reply = self._format_content(event.get("reply"))
            status = "Success" if success else "Failed"
            print(f"\n{status}: {reply}")
            return

        print(f"\nEvent: {self._format_content(event)}")

    def get_reply(self, role: str) -> dict[str, Any]:
        """Get user feedback for plan review or clarification."""
        if role == "plan":
            answer = self._input("Approve plan? [y/n]: ").strip().lower()
            approved = answer in {"y", "yes"}
            if approved:
                return {"approved": True, "text": ""}
            feedback = self._input("Plan feedback: ").strip()
            return {"approved": False, "text": feedback}

        if role == "clarification":
            text = self._input("Reply: ").strip()
            return {"approved": True, "text": text}

        raise ValueError(f"Unsupported reply role: {role}")

    def run(self) -> None:
        """Drive the adapter event loop."""
        task = self.get_task()
        self._sandbox.run(task)

        try:
            while True:
                event = self._sandbox.receive(timeout_seconds=0.1)
                if event is None:
                    continue
                self.show(event)

                if event.get("type") == "message":
                    role = event.get("role")
                    if role == "plan" and task["approval_mode"] == "review":
                        reply = self.get_reply("plan")
                        self._sandbox.send({"type": "user_reply", **reply})
                    elif role == "clarification":
                        reply = self.get_reply("clarification")
                        self._sandbox.send({"type": "user_reply", **reply})

                if event.get("type") == "done":
                    self._persist_done(session_id=str(task["session_id"]), done_event=event)
                    break
        finally:
            self._sandbox.stop()

    @staticmethod
    def _format_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True, indent=2)

    def _persist_done(self, *, session_id: str, done_event: dict[str, Any]) -> None:
        state = done_event.get("state")
        if not isinstance(state, dict):
            return

        session_dir = session.create(session_id)
        session.save(session_dir, state)
        outputs_dir = session_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        files = done_event.get("files")
        if not isinstance(files, list):
            return

        for item in files:
            if not isinstance(item, dict):
                continue
            rel_path = item.get("path")
            content_b64 = item.get("content_b64")
            if not isinstance(rel_path, str) or not rel_path:
                continue
            if not isinstance(content_b64, str):
                continue

            output_path = self._safe_output_path(outputs_dir, rel_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                decoded = base64.b64decode(content_b64, validate=True)
            except binascii.Error as exc:
                raise ValueError(f"Invalid base64 content for output file: {rel_path}") from exc
            output_path.write_bytes(decoded)

    @staticmethod
    def _safe_output_path(outputs_dir: Path, rel_path: str) -> Path:
        candidate = (outputs_dir / rel_path).resolve()
        root = outputs_dir.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Invalid output file path: {rel_path}")
        return candidate
