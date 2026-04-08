"""CLI adapter."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4


class CLIAdapter:
    """CLI interaction contract."""

    def __init__(
        self,
        *,
        sandbox: Any,
        approval_mode: str = "review",
        llm_config: dict[str, Any] | None = None,
        input_func: Callable[[str], str] = input,
    ) -> None:
        self._sandbox = sandbox
        self._approval_mode = approval_mode
        self._llm_config = llm_config
        self._input = input_func

    def get_task(self) -> dict[str, Any]:
        """Get initial task from the user."""
        text = self._input("Task: ").strip()
        if not text:
            raise ValueError("Task cannot be empty.")

        task: dict[str, Any] = {
            "type": "task",
            "text": text,
            "session_id": str(uuid4()),
            "approval_mode": self._approval_mode,
        }
        if self._llm_config is not None:
            task["llm"] = self._llm_config
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
                    break
        finally:
            self._sandbox.stop()

    @staticmethod
    def _format_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=True, indent=2)
