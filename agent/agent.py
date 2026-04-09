"""Core inspect-choose-act-observe loop."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from agent.llm import LLMClient, LLMResponse, ToolCall
from agent.skills import Skills, SkillsError, ToolResult
from agent.transport import InProcessTransport

PLANNING_SYSTEM_PROMPT = (
    "You are strangeclaw, a self-hosted autonomous agent. "
    "Create a concise, executable plan for the user goal using available skills."
)

EXECUTION_SYSTEM_PROMPT = (
    "You are in the execution loop. Respond with a structured decision every turn. "
    "Use a normal skill/action/args tool call to execute tools. "
    "Use skill_contracts in the user payload to satisfy required args and defaults. "
    "For control decisions, use skill='__agent__' and one of actions: "
    "done, clarify, replan. "
    "For done use args.reply. For clarify use args.question. "
    "For replan you may set args.feedback."
)

SUMMARY_SYSTEM_PROMPT = (
    "Summarize agent execution history into concise bullet-style text that preserves "
    "decisions, tool outcomes, and unresolved questions."
)

EXECUTION_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill": {"type": "string"},
        "action": {"type": "string"},
        "args": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["skill", "action", "args"],
    "additionalProperties": False,
}


class AgentError(RuntimeError):
    """Raised for invalid runtime events/configuration."""


class LLMRuntime(Protocol):
    """Minimal runtime contract for the LLM used by Agent."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        action_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    def count_tokens(self, messages: list[dict[str, Any]]) -> int: ...


class Agent:
    """Main agent runtime."""

    def __init__(
        self,
        *,
        transport: InProcessTransport,
        skills_dir: str,
        max_iterations: int = 50,
        output_dir: str = "/output",
        llm_config_path: str = "/run/strangeclaw/llm.json",
        token_budget: int = 4000,
        summary_threshold: int = 10,
        llm_factory: Callable[[dict[str, Any]], LLMRuntime] | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise AgentError("max_iterations must be greater than zero.")
        if token_budget <= 0:
            raise AgentError("token_budget must be greater than zero.")
        if summary_threshold <= 0:
            raise AgentError("summary_threshold must be greater than zero.")

        self._transport = transport
        self._skills = Skills(skills_dir)
        self._max_iterations = max_iterations
        self._output_dir = Path(output_dir)
        self._llm_config_path = Path(llm_config_path)
        self._token_budget = token_budget
        self._summary_threshold = summary_threshold
        self._llm_factory = llm_factory or LLMClient.from_config
        self._llm: LLMRuntime | None = None
        self._history_summary: str | None = None
        self._history_summarized_count = 0

    def run(self) -> None:
        """Run one task from transport input."""
        task_event = self._wait_for_task_event()
        if task_event is None:
            return

        goal = task_event["text"]
        approval_mode = task_event["approval_mode"]
        llm_config = self._resolve_llm_config(task_event)
        self._llm = self._llm_factory(llm_config)
        self._history_summary = None
        self._history_summarized_count = 0

        history, plan = self._resume_context(task_event)
        if plan is None:
            plan = self._planning_phase(goal=goal, approval_mode=approval_mode)

        for _ in range(self._max_iterations):
            try:
                decision = self._execution_decision(goal=goal, plan=plan, history=history)
            except AgentError as exc:
                error_result = ToolResult(
                    exit_code=1,
                    stdout="",
                    stderr=(
                        "Decision parse error: "
                        f"{exc} "
                        "Return a valid JSON tool call with skill/action/args and re-check "
                        "the skill contracts."
                    ),
                )
                action_event = {
                    "type": "action",
                    "skill": "__agent__",
                    "action": "decision_error",
                    "args": {},
                    "result": asdict(error_result),
                }
                self._send(action_event)
                history.append(action_event)
                continue

            if decision.skill == "__agent__":
                handled = self._handle_control_decision(
                    decision=decision,
                    goal=goal,
                    approval_mode=approval_mode,
                    history=history,
                    current_plan=plan,
                )
                if handled["done"]:
                    return
                if handled["replanned"]:
                    plan = handled["plan"]
                continue

            result = self._execute_tool(decision)
            action_event = {
                "type": "action",
                "skill": decision.skill,
                "action": decision.action,
                "args": decision.args,
                "result": asdict(result),
            }
            self._send(action_event)
            history.append(action_event)

        self._send(
            {
                "type": "message",
                "role": "clarification",
                "content": (
                    "I reached the maximum iteration limit before finishing. "
                    "Please clarify the goal or constraints."
                ),
            }
        )
        self._send(
            {
                "type": "done",
                "success": False,
                "reply": "Stopped after reaching iteration limit.",
                "state": {
                    "goal": goal,
                    "plan": plan,
                    "history": history,
                    "summary": self._history_summary or "",
                },
                "files": self._collect_output_files(),
            }
        )

    def _resume_context(
        self,
        task_event: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], Any | None]:
        state = task_event.get("state")
        if not isinstance(state, dict):
            return [], None

        history: list[dict[str, Any]] = []
        raw_history = state.get("history")
        if isinstance(raw_history, list):
            history = [item for item in raw_history if isinstance(item, dict)]

        summary = state.get("summary")
        if isinstance(summary, str) and summary.strip():
            self._history_summary = summary
        if len(history) > self._summary_threshold:
            self._history_summarized_count = len(history) - self._summary_threshold

        return history, state.get("plan")

    def _wait_for_task_event(self) -> dict[str, Any] | None:
        while True:
            event = self._transport.receive()
            if event is None:
                continue
            event_type = event.get("type")
            if event_type == "stop":
                return None
            if event_type == "task":
                return event

    def _resolve_llm_config(self, task_event: dict[str, Any]) -> dict[str, Any]:
        llm_from_task = task_event.get("llm")
        if isinstance(llm_from_task, dict):
            return llm_from_task

        if not self._llm_config_path.is_file():
            raise AgentError(
                "Task did not contain llm config and MMDS LLM file was not found: "
                f"{self._llm_config_path}"
            )
        raw = self._llm_config_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise AgentError(f"LLM config in {self._llm_config_path} must be a JSON object.")
        return parsed

    def _planning_phase(self, *, goal: str, approval_mode: str) -> Any:
        feedback: str | None = None
        while True:
            plan = self._generate_plan(goal=goal, feedback=feedback)
            self._send({"type": "message", "role": "plan", "content": plan})

            if approval_mode != "review":
                return plan

            reply = self._wait_for_user_reply()
            if reply["approved"]:
                return plan
            feedback = reply.get("text", "")

    def _generate_plan(self, *, goal: str, feedback: str | None) -> Any:
        llm = self._require_llm()
        messages = self.build_planning_prompt(goal=goal, feedback=feedback)
        response = llm.complete(messages)
        return _parse_json_if_possible(response.text)

    def _execution_decision(
        self,
        *,
        goal: str,
        plan: Any,
        history: list[dict[str, Any]],
    ) -> ToolCall:
        llm = self._require_llm()
        messages = self.build_execution_prompt(goal=goal, plan=plan, history=history)
        response: LLMResponse = llm.complete(
            messages,
            action_schema=EXECUTION_ACTION_SCHEMA,
        )
        if response.action is not None:
            return response.action
        fallback = _parse_text_fallback_action(response.text)
        if fallback is not None:
            return fallback
        raise AgentError("LLM response did not contain an executable action decision.")

    def _handle_control_decision(
        self,
        *,
        decision: ToolCall,
        goal: str,
        approval_mode: str,
        history: list[dict[str, Any]],
        current_plan: Any,
    ) -> dict[str, Any]:
        if decision.action == "done":
            reply = _string_arg(decision.args, "reply", fallback="Task completed.")
            self._send(
                {
                    "type": "done",
                    "success": True,
                    "reply": reply,
                    "state": {
                        "goal": goal,
                        "plan": current_plan,
                        "history": history,
                        "summary": self._history_summary or "",
                    },
                    "files": self._collect_output_files(),
                }
            )
            return {"done": True, "replanned": False, "plan": current_plan}

        if decision.action == "clarify":
            question = _string_arg(
                decision.args,
                "question",
                fallback="Please clarify what you want next.",
            )
            self._send({"type": "message", "role": "clarification", "content": question})
            user_reply = self._wait_for_user_reply()
            history.append(
                {
                    "type": "clarification",
                    "question": question,
                    "user_reply": user_reply.get("text", ""),
                }
            )
            return {"done": False, "replanned": False, "plan": current_plan}

        if decision.action == "replan":
            feedback = _string_arg(decision.args, "feedback", fallback="")
            if feedback:
                history.append({"type": "replan", "feedback": feedback})
            new_plan = self._planning_phase(goal=goal, approval_mode=approval_mode)
            return {"done": False, "replanned": True, "plan": new_plan}

        history.append(
            {
                "type": "control_error",
                "action": decision.action,
                "reason": "unsupported __agent__ action",
            }
        )
        return {"done": False, "replanned": False, "plan": current_plan}

    def _execute_tool(self, decision: ToolCall) -> ToolResult:
        try:
            return self._skills.execute(
                {
                    "skill": decision.skill,
                    "action": decision.action,
                    "args": decision.args,
                }
            )
        except SkillsError as exc:
            return ToolResult(exit_code=1, stdout="", stderr=f"Skill execution error: {exc}")

    def _wait_for_user_reply(self) -> dict[str, Any]:
        while True:
            event = self._transport.receive()
            if event is None:
                continue
            event_type = event.get("type")
            if event_type == "stop":
                raise AgentError("Received stop while waiting for user reply.")
            if event_type == "user_reply":
                return event

    def _send(self, event: dict[str, Any]) -> None:
        self._transport.send(event)

    def _collect_output_files(self) -> list[dict[str, Any]]:
        if not self._output_dir.exists():
            return []

        files: list[dict[str, Any]] = []
        for file_path in sorted(self._output_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel_path = file_path.relative_to(self._output_dir).as_posix()
            content = file_path.read_bytes()
            files.append(
                {
                    "path": rel_path,
                    "content_b64": base64.b64encode(content).decode("ascii"),
                    "size_bytes": len(content),
                }
            )
        return files

    def _require_llm(self) -> LLMRuntime:
        if self._llm is None:
            raise AgentError("LLM client is not configured.")
        return self._llm

    def build_planning_prompt(
        self,
        *,
        goal: str,
        feedback: str | None = None,
    ) -> list[dict[str, str]]:
        """Build planning-phase messages."""
        prompt_lines = [
            f"Goal:\n{goal}",
            "",
            "Available skills:",
            json.dumps(self._skills.index(), ensure_ascii=True, indent=2),
            "",
            "Return a short plan as JSON with keys goal and steps (array of strings).",
        ]
        if feedback:
            prompt_lines.extend(["", f"User feedback for re-plan:\n{feedback}"])
        return [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(prompt_lines)},
        ]

    def build_execution_prompt(
        self,
        *,
        goal: str,
        plan: Any,
        history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Build execution-phase messages, enforcing token budget."""
        recent_history = self._recent_history(history)
        summary = self._history_summary or ""

        while True:
            user_payload = {
                "goal": goal,
                "plan": plan,
                "skills": self._skills.index(),
                "skill_contracts": self._skills.contracts(),
                "history_summary": summary,
                "recent_history": recent_history,
                "output_instruction": "Place any files for the user in /output/.",
            }
            messages = [
                {"role": "system", "content": EXECUTION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ]
            token_count = self._require_llm().count_tokens(messages)
            if token_count <= self._token_budget:
                return messages
            if recent_history:
                recent_history = recent_history[1:]
                continue
            if summary:
                summary = ""
                continue
            return messages

    def _recent_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return recent history and update summary when threshold is exceeded."""
        if len(history) <= self._summary_threshold:
            return history

        cutoff = len(history) - self._summary_threshold
        if cutoff > self._history_summarized_count:
            self._summarize_history(history[:cutoff])
            self._history_summarized_count = cutoff
        return history[cutoff:]

    def _summarize_history(self, chunk: list[dict[str, Any]]) -> None:
        llm = self._require_llm()
        summary_input = {
            "previous_summary": self._history_summary or "",
            "new_events": chunk,
        }
        response = llm.complete(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(summary_input, ensure_ascii=True)},
            ]
        )
        text = response.text.strip()
        if text:
            self._history_summary = text


def _parse_json_if_possible(text: str) -> Any:
    value = _first_json_object_or_array(text)
    return value if value is not None else text


def _first_json_object_or_array(text: str) -> dict[str, Any] | list[Any] | None:
    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict | list):
            return parsed
    return None


def _parse_text_fallback_action(text: str) -> ToolCall | None:
    payload = _first_json_object_or_array(text)
    if not isinstance(payload, dict):
        return None
    skill = payload.get("skill")
    action = payload.get("action")
    args = payload.get("args")
    reason = payload.get("reason")
    if not isinstance(skill, str) or not isinstance(action, str) or not isinstance(args, dict):
        return None
    if reason is not None and not isinstance(reason, str):
        reason = None
    return ToolCall(skill=skill, action=action, args=args, reason=reason)


def _string_arg(args: dict[str, Any], key: str, fallback: str) -> str:
    value = args.get(key)
    if isinstance(value, str) and value:
        return value
    return fallback
