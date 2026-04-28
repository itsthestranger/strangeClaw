"""Core inspect-choose-act-observe loop."""

from __future__ import annotations

import argparse
import base64
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from agent.llm import LLMClient, LLMResponse, ToolCall
from agent.skills import Skills, SkillsError, ToolResult
from agent.transport import VsockTransport

PLANNING_SYSTEM_PROMPT = (
    "You are strangeclaw, a self-hosted autonomous agent. "
    "Create a concise, executable plan for the user goal using available skills."
)

EXECUTION_SYSTEM_PROMPT = (
    "You are in an agentic loop: Inspect -> Choose -> Act -> Observe -> Repeat. "
    "On each turn, inspect goal/plan/history and choose exactly one structured decision. "
    "Use a normal tool/args tool call to execute tools. "
    "Use skill_contracts in the user payload to satisfy required args and defaults. "
    "For control decisions, use tool='__agent__.done', '__agent__.clarify', or '__agent__.replan'. "
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
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["tool", "args"],
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


class AgentTransport(Protocol):
    """Transport contract required by Agent."""

    def send(self, event: dict[str, Any]) -> None: ...

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None: ...


class Agent:
    """Main agent runtime."""

    def __init__(
        self,
        *,
        transport: AgentTransport,
        skills_dir: str,
        agent_config: dict[str, Any] | None = None,
        max_iterations: int = 50,
        output_dir: str = "/output",
        llm_config_path: str = "/run/strangeclaw/llm.json",
        token_budget: int = 4000,
        summary_threshold: int = 10,
        max_output_total_bytes: int = 10 * 1024 * 1024,
        allow_task_llm: bool = True,
        llm_factory: Callable[[dict[str, Any]], LLMRuntime] | None = None,
    ) -> None:
        if max_iterations <= 0:
            raise AgentError("max_iterations must be greater than zero.")
        if token_budget <= 0:
            raise AgentError("token_budget must be greater than zero.")
        if summary_threshold <= 0:
            raise AgentError("summary_threshold must be greater than zero.")
        if max_output_total_bytes <= 0:
            raise AgentError("max_output_total_bytes must be greater than zero.")

        self._transport = transport
        self._skills = Skills(skills_dir)
        self._agent_config = dict(agent_config) if isinstance(agent_config, dict) else None
        self._max_iterations = max_iterations
        self._output_dir = Path(output_dir)
        self._llm_config_path = Path(llm_config_path)
        self._token_budget = token_budget
        self._summary_threshold = summary_threshold
        self._max_output_total_bytes = max_output_total_bytes
        self._allow_task_llm = allow_task_llm
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
            decision = self._choose_next_decision(goal=goal, plan=plan, history=history)
            if decision is None:
                continue

            outcome = self._act_on_decision(
                decision=decision,
                goal=goal,
                approval_mode=approval_mode,
                current_plan=plan,
                history=history,
            )
            plan = outcome["plan"]
            self._observe(history=history, observation=outcome["observation"])
            if outcome["done"]:
                return

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
            self._build_done_event(
                goal=goal,
                plan=plan,
                history=history,
                success=False,
                reply="Stopped after reaching iteration limit.",
            )
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
        if self._agent_config is not None:
            llm_from_config = self._agent_config.get("llm")
            if isinstance(llm_from_config, dict):
                return dict(llm_from_config)
            raise AgentError("Agent config is missing required llm mapping.")

        llm_from_task = task_event.get("llm")
        if self._allow_task_llm and isinstance(llm_from_task, dict):
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

    def _choose_next_decision(
        self,
        *,
        goal: str,
        plan: Any,
        history: list[dict[str, Any]],
    ) -> ToolCall | None:
        try:
            return self._execution_decision(goal=goal, plan=plan, history=history)
        except AgentError as exc:
            history.append(self._emit_decision_parse_error(exc))
            return None

    def _emit_decision_parse_error(self, error: AgentError) -> dict[str, Any]:
        error_result = ToolResult(
            exit_code=1,
            stdout="",
            stderr=(
                "Decision parse error: "
                f"{error} "
                "Return a valid JSON tool call with tool/args and re-check "
                "the skill contracts."
            ),
        )
        action_event = {
            "type": "action",
            "tool": "__agent__.decision_error",
            "args": {},
            "result": asdict(error_result),
        }
        self._send(action_event)
        return action_event

    def _act_on_decision(
        self,
        *,
        decision: ToolCall,
        goal: str,
        approval_mode: str,
        current_plan: Any,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if decision.tool.startswith("__agent__."):
            control_action = decision.tool.split(".", 1)[1]
            return self._handle_control_decision(
                decision=decision,
                control_action=control_action,
                goal=goal,
                approval_mode=approval_mode,
                current_plan=current_plan,
                history=history,
            )

        skill_name, action_name = _split_tool_name(decision.tool)
        if skill_name is None or action_name is None:
            invalid_result = ToolResult(
                exit_code=1,
                stdout="",
                stderr=(
                    f"Invalid tool name '{decision.tool}'. "
                    "Expected '<skill>.<action>' during legacy execution path."
                ),
            )
            action_event = {
                "type": "action",
                "tool": "__agent__.invalid_tool_name",
                "args": {"tool": decision.tool},
                "result": asdict(invalid_result),
            }
            self._send(action_event)
            return {"done": False, "plan": current_plan, "observation": action_event}

        result = self._execute_tool(decision)
        action_event = {
            "type": "action",
            "tool": f"{skill_name}.{action_name}",
            "args": decision.args,
            "result": asdict(result),
        }
        self._send(action_event)
        return {"done": False, "plan": current_plan, "observation": action_event}

    @staticmethod
    def _observe(*, history: list[dict[str, Any]], observation: dict[str, Any] | None) -> None:
        if observation is None:
            return
        history.append(observation)

    def _handle_control_decision(
        self,
        *,
        decision: ToolCall,
        control_action: str,
        goal: str,
        approval_mode: str,
        current_plan: Any,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if control_action == "done":
            reply = _string_arg(decision.args, "reply", fallback="Task completed.")
            self._send(
                self._build_done_event(
                    goal=goal,
                    plan=current_plan,
                    history=history,
                    success=True,
                    reply=reply,
                )
            )
            return {"done": True, "plan": current_plan, "observation": None}

        if control_action == "clarify":
            question = _string_arg(
                decision.args,
                "question",
                fallback="Please clarify what you want next.",
            )
            self._send({"type": "message", "role": "clarification", "content": question})
            user_reply = self._wait_for_user_reply()
            return {
                "done": False,
                "plan": current_plan,
                "observation": {
                    "type": "clarification",
                    "question": question,
                    "user_reply": user_reply.get("text", ""),
                },
            }

        if control_action == "replan":
            feedback = _string_arg(decision.args, "feedback", fallback="")
            new_plan = self._planning_phase(goal=goal, approval_mode=approval_mode)
            observation: dict[str, Any] | None = None
            if feedback:
                observation = {"type": "replan", "feedback": feedback}
            return {"done": False, "plan": new_plan, "observation": observation}

        return {
            "done": False,
            "plan": current_plan,
            "observation": {
                "type": "control_error",
                "action": control_action,
                "reason": "unsupported __agent__ action",
            },
        }

    def _execute_tool(self, decision: ToolCall) -> ToolResult:
        skill_name, action_name = _split_tool_name(decision.tool)
        if skill_name is None or action_name is None:
            return ToolResult(
                exit_code=1,
                stdout="",
                stderr=(
                    f"Invalid tool name '{decision.tool}'. "
                    "Expected '<skill>.<action>' during legacy execution path."
                ),
            )
        try:
            return self._skills.execute(
                {
                    "skill": skill_name,
                    "action": action_name,
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

    def _build_done_event(
        self,
        *,
        goal: str,
        plan: Any,
        history: list[dict[str, Any]],
        success: bool,
        reply: str,
    ) -> dict[str, Any]:
        files, output_error = self._collect_output_files()
        final_success = success
        final_reply = reply
        if output_error is not None:
            final_success = False
            final_reply = f"{reply}\n\nOutput export error: {output_error}"
        return {
            "type": "done",
            "success": final_success,
            "reply": final_reply,
            "state": {
                "goal": goal,
                "plan": plan,
                "history": history,
                "summary": self._history_summary or "",
            },
            "files": files,
        }

    def _collect_output_files(self) -> tuple[list[dict[str, Any]], str | None]:
        if not self._output_dir.exists():
            return [], None

        files: list[dict[str, Any]] = []
        root = self._output_dir.resolve()
        limit_bytes = self._max_output_total_bytes
        total_bytes = 0
        for file_path in sorted(self._output_dir.rglob("*")):
            if not file_path.is_file() or file_path.is_symlink():
                continue
            resolved_path = file_path.resolve()
            if root not in resolved_path.parents:
                return [], f"Invalid output file path: {file_path}"
            rel_path = resolved_path.relative_to(root).as_posix()
            size_bytes = resolved_path.stat().st_size
            if size_bytes > limit_bytes:
                return (
                    [],
                    f"Output file '{rel_path}' exceeds output limit of {limit_bytes} bytes.",
                )
            if total_bytes + size_bytes > limit_bytes:
                return [], f"Total output size exceeds output limit of {limit_bytes} bytes."
            content = resolved_path.read_bytes()
            total_bytes += len(content)
            files.append(
                {
                    "path": rel_path,
                    "content_b64": base64.b64encode(content).decode("ascii"),
                    "size_bytes": len(content),
                }
            )
        return files, None

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
    tool = payload.get("tool")
    if not isinstance(tool, str):
        skill = payload.get("skill")
        action = payload.get("action")
        if isinstance(skill, str) and isinstance(action, str):
            tool = f"{skill}.{action}"
    args = payload.get("args")
    reason = payload.get("reason")
    if not isinstance(tool, str) or not isinstance(args, dict):
        return None
    if reason is not None and not isinstance(reason, str):
        reason = None
    return ToolCall(tool=tool, args=args, reason=reason)


def _string_arg(args: dict[str, Any], key: str, fallback: str) -> str:
    value = args.get(key)
    if isinstance(value, str) and value:
        return value
    return fallback


def _split_tool_name(tool: str) -> tuple[str | None, str | None]:
    if "." not in tool:
        return None, None
    skill_name, action_name = tool.split(".", 1)
    if not skill_name or not action_name:
        return None, None
    return skill_name, action_name


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m agent.agent")
    parser.add_argument("--vsock-port", type=int, default=None, help="Guest vsock listen port.")
    parser.add_argument("--skills-dir", type=str, default="/opt/strangeclaw/skills")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--summary-threshold", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="/output")
    parser.add_argument("--llm-config-path", type=str, default="/run/strangeclaw/llm.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for guest execution."""
    args = _parse_args(argv)
    if args.vsock_port is None:
        raise ValueError("Missing required --vsock-port for guest execution.")

    transport = VsockTransport(guest_port=int(args.vsock_port))
    try:
        agent = Agent(
            transport=transport,
            skills_dir=str(args.skills_dir),
            max_iterations=int(args.max_iterations),
            output_dir=str(args.output_dir),
            llm_config_path=str(args.llm_config_path),
            token_budget=int(args.token_budget),
            summary_threshold=int(args.summary_threshold),
            allow_task_llm=False,
        )
        agent.run()
    finally:
        transport.close()


if __name__ == "__main__":
    main()
