"""Core inspect-choose-act-observe loop."""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from agent.broker_client import BrokerClient, HostServiceError
from agent.llm_proxy import LLMProxyRuntime
from agent.llm_types import LLMResponse, LLMRuntime, LLMRuntimeError, ToolCall
from agent.skills import Skills, SkillsError
from agent.tools import ToolResult, Tools, wrap_external_data
from agent.transport import VsockTransport

PLANNING_SYSTEM_PROMPT = (
    "You are strangeclaw, a self-hosted autonomous agent. "
    "Create a concise, executable plan for the user goal using available tools and skills."
)

EXECUTION_SYSTEM_PROMPT = (
    "You are in an agentic loop: Inspect -> Choose -> Act -> Observe -> Repeat. "
    "On each turn, inspect goal/plan/history and choose exactly one structured decision. "
    "Use a normal tool/args tool call to execute tools. "
    "Use activated_skills in the user payload for workflow guidance and references. "
    "For control decisions, use tool='agent_done', 'agent_clarify', or 'agent_replan'. "
    "For stage-3 skill references, use tool='agent_read_skill_file' "
    "with args.skill and args.path. "
    "For done use args.reply. For clarify use args.question. "
    "For replan you may set args.feedback."
)

# Execution-loop invariants:
# 1. The model issues exactly one structured decision per turn.
# 2. The runtime never chooses decisions for the model.
# 3. The runtime only validates, executes, and feeds observations back.
# 4. Control decisions are model-issued tools: agent_done, agent_clarify,
#    agent_replan, and agent_read_skill_file.
# 5. Free-form prose is not a valid execution-loop decision.
# Hard safety exits (iteration limits, stop events, sandbox/transport failures)
# remain runtime-owned and are not model decisions.

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

_CONTROL_TOOL_DONE = "agent_done"
_CONTROL_TOOL_CLARIFY = "agent_clarify"
_CONTROL_TOOL_REPLAN = "agent_replan"
_CONTROL_TOOL_READ_SKILL_FILE = "agent_read_skill_file"
_CONTROL_TOOL_DECISION_ERROR = "agent_decision_error"
_PROVIDER_SAFE_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_CONTROL_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": _CONTROL_TOOL_DONE,
        "description": "Finish execution and return the final user-facing reply.",
        "parameters": {
            "type": "object",
            "properties": {"reply": {"type": "string"}},
            "required": ["reply"],
            "additionalProperties": False,
        },
    },
    {
        "name": _CONTROL_TOOL_CLARIFY,
        "description": "Ask the user a clarification question before continuing.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": _CONTROL_TOOL_REPLAN,
        "description": "Request a fresh plan before continuing execution.",
        "parameters": {
            "type": "object",
            "properties": {"feedback": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": _CONTROL_TOOL_READ_SKILL_FILE,
        "description": "Read a bundled file from an activated skill by relative path.",
        "parameters": {
            "type": "object",
            "properties": {
                "skill": {"type": "string"},
                "path": {"type": "string"},
            },
            "required": ["skill", "path"],
            "additionalProperties": False,
        },
    },
]
_CONTROL_TOOL_NAMES = {schema["name"] for schema in _CONTROL_TOOL_SCHEMAS}

_SPAWN_SUBAGENT_TOOL_NAME = "spawn_subagent"
_SUBAGENT_EVENTS_SUMMARY_MAX = 50

# Final reply emitted when the execution loop exhausts its iteration budget.
# Shared so the subagent runner can map this outcome to a `max_iterations` status.
MAX_ITERATIONS_REPLY = "Stopped after reaching iteration limit."
_SPAWN_SUBAGENT_SCHEMA: dict[str, Any] = {
    "name": _SPAWN_SUBAGENT_TOOL_NAME,
    "description": (
        "Delegate a separable subtask to a child agent that runs to completion in "
        "the same sandbox/session and returns a single structured result. Sequential "
        "and blocking: one child at a time. The child uses a subset of your tools, "
        "cannot ask the user anything, and reports gaps in its reply."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "task": {"type": "string"},
            "context": {"type": "string"},
            "expected_output": {"type": "string"},
            "allowed_tools": {"type": "array", "items": {"type": "string"}},
            "referenced_skills": {"type": "array", "items": {"type": "string"}},
            "max_iterations": {"type": "integer"},
            "timeout_seconds": {"type": "integer"},
        },
        "required": ["task"],
        "additionalProperties": False,
    },
}


class AgentError(RuntimeError):
    """Raised for invalid runtime events/configuration."""


class AgentTransport(Protocol):
    """Transport contract required by Agent."""

    def send(self, event: dict[str, Any]) -> None: ...

    def receive(self, timeout_seconds: float | None = None) -> dict[str, Any] | None: ...


class SubagentRunnerProtocol(Protocol):
    """Runner contract used by Agent to execute one child agent synchronously."""

    def run(self, request: dict[str, Any]) -> dict[str, Any]: ...


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
        agent_config_path: str = "/run/strangeclaw/config.json",
        token_budget: int = 4000,
        summary_threshold: int = 10,
        max_output_total_bytes: int = 10 * 1024 * 1024,
        llm_runtime: LLMRuntime | None = None,
        broker: BrokerClient | None = None,
        subagent_runner: SubagentRunnerProtocol | None = None,
        clarify_enabled: bool = True,
        task_timeout_seconds: float | None = None,
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
        self._agent_config_path = Path(agent_config_path)
        self._agent_config = dict(agent_config) if isinstance(agent_config, dict) else None
        if self._agent_config is None:
            if self._agent_config_path.is_file():
                self._agent_config = _load_agent_config_file(self._agent_config_path)
            elif llm_runtime is None:
                self._agent_config = _load_agent_config_file(self._agent_config_path)

        if isinstance(self._agent_config, dict):
            configured_skills_dir = _read_skills_directory(self._agent_config)
            if configured_skills_dir is not None:
                skills_dir = configured_skills_dir
            max_iterations = _read_loop_max_iterations(self._agent_config, fallback=max_iterations)
            token_budget = _read_context_int(
                self._agent_config,
                "token_budget",
                fallback=token_budget,
            )
            summary_threshold = _read_context_int(
                self._agent_config,
                "summary_threshold",
                fallback=summary_threshold,
            )

        skills_max_file_chars = _read_skills_max_file_chars(self._agent_config)
        self._skills_dir = skills_dir
        self._skills = Skills(skills_dir, max_file_chars=skills_max_file_chars)
        self._broker = broker
        self._tools = Tools(self._agent_config or {}, broker=broker)
        self._subagent_runner = subagent_runner
        self._subagents_settings = _read_subagents_settings(self._agent_config)
        self._subagents_enabled = _read_subagents_enabled(self._agent_config)
        self._clarify_enabled = clarify_enabled
        self._task_timeout_seconds = task_timeout_seconds
        self._deadline: float | None = None
        self._subagent_children_spawned = 0
        self._active_session_id = ""
        self._integrations: list[str] = []
        control_schemas = (
            _CONTROL_TOOL_SCHEMAS
            if clarify_enabled
            else [s for s in _CONTROL_TOOL_SCHEMAS if s["name"] != _CONTROL_TOOL_CLARIFY]
        )
        extra_action_schemas = (
            [_SPAWN_SUBAGENT_SCHEMA] if self._subagents_enabled else []
        )
        self._execution_action_surface = _build_execution_action_surface(
            self._tools.schema(),
            control_schemas=control_schemas,
            extra_schemas=extra_action_schemas,
        )
        self._max_iterations = max_iterations
        self._output_dir = Path(output_dir)
        self._token_budget = token_budget
        self._summary_threshold = summary_threshold
        self._max_output_total_bytes = max_output_total_bytes
        self._llm = llm_runtime or _build_default_llm_runtime(self._require_llm_config())
        self._history_summary: str | None = None
        self._history_summarized_count = 0
        # Important: do not call broker-backed _load_integrations() here.
        # In Fire mode, BrokerClient._call_fire() reads from the same transport
        # used for host task delivery; doing this before run_forever() can race
        # and consume/discard the first task event.
        self._integrations = []

    def _load_integrations(self) -> list[str]:
        if self._broker is None:
            return []
        try:
            response = self._broker.call("broker", {"action": "list_integrations"})
        except HostServiceError:
            return []
        names = response.get("integrations")
        if not isinstance(names, list):
            return []
        return sorted(name for name in names if isinstance(name, str) and name.strip())

    def run(self) -> None:
        """Run one task from transport input."""
        task_event = self._wait_for_task_event()
        if task_event is None:
            return
        self._run_task(task_event)

    def run_forever(self) -> None:
        """Run tasks from transport input until a stop event arrives while idle."""
        while True:
            task_event = self._wait_for_task_event()
            if task_event is None:
                return
            self._run_task(task_event)

    def _run_task(self, task_event: dict[str, Any]) -> None:
        """Run one task event using this agent instance."""
        goal = task_event["text"]
        approval_mode = task_event["approval_mode"]
        self._active_session_id = str(task_event.get("session_id") or "")
        self._subagent_children_spawned = 0
        self._deadline = (
            time.monotonic() + self._task_timeout_seconds
            if self._task_timeout_seconds is not None
            else None
        )
        self._integrations = self._load_integrations()
        self._history_summary = None
        self._history_summarized_count = 0

        history, plan = self._resume_context(task_event)
        try:
            if plan is None:
                plan = self._planning_phase(
                    goal=goal,
                    approval_mode=approval_mode,
                    history=history,
                )
            plan = self._normalize_plan(plan, goal=goal)
            plan, activated_skills = self._activate_referenced_skills(
                goal=goal,
                approval_mode=approval_mode,
                plan=plan,
                history=history,
            )
        except AgentError as exc:
            error_event = self._emit_decision_parse_error(exc)
            history.append(error_event)
            self._send(
                self._build_done_event(
                    goal=goal,
                    plan=plan,
                    history=history,
                    success=False,
                    reply=f"Unable to continue: {exc}",
                )
            )
            return

        # Strict Inspect -> Choose -> Act -> Observe -> Repeat loop.
        # Exactly one model-issued structured decision is required per turn.
        for _ in range(self._max_iterations):
            if self._deadline is not None and time.monotonic() >= self._deadline:
                # Time budget reached at an iteration boundary (a subagent timeout).
                # Return without a done event; the caller (SubagentRunner) maps a
                # missing done to a timeout. No effect for parent agents, which run
                # with no task_timeout_seconds.
                return
            decision = self._choose_next_decision(
                goal=goal,
                plan=plan,
                history=history,
                activated_skills=activated_skills,
            )
            if decision is None:
                continue

            outcome = self._act_on_decision(
                decision=decision,
                goal=goal,
                approval_mode=approval_mode,
                current_plan=plan,
                activated_skills=activated_skills,
                history=history,
            )
            plan = outcome["plan"]
            if outcome.get("replanned"):
                plan = self._normalize_plan(plan, goal=goal)
                plan, activated_skills = self._activate_referenced_skills(
                    goal=goal,
                    approval_mode=approval_mode,
                    plan=plan,
                    history=history,
                )
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
                reply=MAX_ITERATIONS_REPLY,
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

    def _planning_phase(
        self,
        *,
        goal: str,
        approval_mode: str,
        history: list[dict[str, Any]] | None = None,
    ) -> Any:
        feedback: str | None = None
        llm_runtime_error_count = 0
        while True:
            try:
                plan = self._generate_plan(goal=goal, feedback=feedback)
                llm_runtime_error_count = 0
            except LLMRuntimeError as exc:
                observation = self._emit_llm_runtime_error(exc)
                if history is not None:
                    history.append(observation)
                llm_runtime_error_count += 1
                if llm_runtime_error_count >= self._max_iterations:
                    raise AgentError(
                        "LLM runtime failed repeatedly during planning."
                    ) from exc
                continue
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

    def _normalize_plan(self, plan: Any, *, goal: str) -> dict[str, Any]:
        if isinstance(plan, dict):
            normalized_goal = plan.get("goal")
            if not isinstance(normalized_goal, str) or not normalized_goal.strip():
                normalized_goal = goal
            raw_steps = plan.get("steps")
            steps: list[str] = []
            if isinstance(raw_steps, list):
                steps = [str(step).strip() for step in raw_steps if str(step).strip()]
            raw_refs = plan.get("referenced_skills")
            referenced_skills: list[str] = []
            if isinstance(raw_refs, list):
                referenced_skills = [
                    str(name).strip() for name in raw_refs if isinstance(name, str) and name.strip()
                ]
            return {
                "goal": normalized_goal,
                "steps": steps,
                "referenced_skills": referenced_skills,
            }
        if isinstance(plan, str):
            return {"goal": goal, "steps": [plan], "referenced_skills": []}
        return {"goal": goal, "steps": [], "referenced_skills": []}

    def _activate_referenced_skills(
        self,
        *,
        goal: str,
        approval_mode: str,
        plan: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        current_plan = plan
        attempts = 0
        while True:
            attempts += 1
            if attempts > 3:
                raise AgentError(
                    "Failed to produce a valid plan after 3 replan attempts "
                    "due to unknown referenced_skills."
                )
            activated: dict[str, dict[str, Any]] = {}
            refs = current_plan.get("referenced_skills", [])
            if not isinstance(refs, list):
                refs = []
            activation_error: str | None = None
            for name in refs:
                if not isinstance(name, str) or not name.strip():
                    continue
                skill_name = name.strip()
                try:
                    activated[skill_name] = self._skills.get_doc(skill_name)
                except SkillsError as exc:
                    activation_error = (
                        f"Unknown referenced skill '{skill_name}' in plan: {exc}. "
                        "Re-planning with available skills only."
                    )
                    break
            if activation_error is None:
                return current_plan, activated

            self._send({"type": "message", "role": "status", "content": activation_error})
            current_plan = self._normalize_plan(
                self._planning_phase(
                    goal=goal,
                    approval_mode=approval_mode,
                    history=history,
                ),
                goal=goal,
            )

    def _execution_decision(
        self,
        *,
        goal: str,
        plan: Any,
        history: list[dict[str, Any]],
        activated_skills: dict[str, dict[str, Any]],
    ) -> ToolCall:
        llm = self._require_llm()
        messages = self.build_execution_prompt(
            goal=goal,
            plan=plan,
            history=history,
            activated_skills=activated_skills,
        )
        response: LLMResponse = llm.complete(
            messages,
            action_schema=self._execution_action_surface,
        )
        if response.action is not None:
            return response.action
        raise AgentError(
            "LLM response did not contain exactly one structured execution decision."
        )

    def _choose_next_decision(
        self,
        *,
        goal: str,
        plan: Any,
        history: list[dict[str, Any]],
        activated_skills: dict[str, dict[str, Any]],
    ) -> ToolCall | None:
        try:
            return self._execution_decision(
                goal=goal,
                plan=plan,
                history=history,
                activated_skills=activated_skills,
            )
        except LLMRuntimeError as exc:
            history.append(self._emit_llm_runtime_error(exc))
            return None
        except AgentError as exc:
            history.append(self._emit_decision_parse_error(exc))
            return None

    def _emit_llm_runtime_error(self, error: LLMRuntimeError) -> dict[str, Any]:
        error_result = ToolResult(exit_code=1, stdout="", stderr=error.message)
        action_event = {
            "type": "action",
            "tool": _CONTROL_TOOL_DECISION_ERROR,
            "args": {"category": "llm_runtime_error"},
            "result": asdict(error_result),
        }
        self._send(action_event)
        return action_event

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
            "tool": _CONTROL_TOOL_DECISION_ERROR,
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
        activated_skills: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if decision.tool == _SPAWN_SUBAGENT_TOOL_NAME:
            return self._handle_spawn_subagent(decision=decision, current_plan=current_plan)

        if decision.tool in _CONTROL_TOOL_NAMES:
            return self._handle_control_decision(
                decision=decision,
                control_tool=decision.tool,
                goal=goal,
                approval_mode=approval_mode,
                current_plan=current_plan,
                activated_skills=activated_skills,
                history=history,
            )

        result = self._execute_tool(decision)
        action_event = {
            "type": "action",
            "tool": decision.tool,
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
        control_tool: str,
        goal: str,
        approval_mode: str,
        current_plan: Any,
        activated_skills: dict[str, dict[str, Any]],
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if control_tool == _CONTROL_TOOL_DONE:
            reply_raw = decision.args.get("reply")
            if not isinstance(reply_raw, str) or not reply_raw.strip():
                error_event = self._emit_control_action_error(
                    tool=control_tool,
                    args=decision.args,
                    message="agent_done requires args.reply as a non-empty string.",
                )
                return {"done": False, "plan": current_plan, "observation": error_event}
            reply = reply_raw
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

        if control_tool == _CONTROL_TOOL_CLARIFY:
            if not self._clarify_enabled:
                error_event = self._emit_control_action_error(
                    tool=control_tool,
                    args=decision.args,
                    message=(
                        "clarification is unavailable in this context; proceed on stated "
                        "assumptions or finish via agent_done and report what you need."
                    ),
                )
                return {"done": False, "plan": current_plan, "observation": error_event}
            question_raw = decision.args.get("question")
            if question_raw is None:
                question = "Please clarify what you want next."
            elif isinstance(question_raw, str):
                question = question_raw or "Please clarify what you want next."
            else:
                error_event = self._emit_control_action_error(
                    tool=control_tool,
                    args=decision.args,
                    message="agent_clarify args.question must be a string when provided.",
                )
                return {"done": False, "plan": current_plan, "observation": error_event}
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

        if control_tool == _CONTROL_TOOL_REPLAN:
            feedback_raw = decision.args.get("feedback")
            if feedback_raw is None:
                feedback = ""
            elif isinstance(feedback_raw, str):
                feedback = feedback_raw
            else:
                error_event = self._emit_control_action_error(
                    tool=control_tool,
                    args=decision.args,
                    message="agent_replan args.feedback must be a string when provided.",
                )
                return {"done": False, "plan": current_plan, "observation": error_event}
            new_plan = self._planning_phase(
                goal=goal,
                approval_mode=approval_mode,
                history=history,
            )
            observation: dict[str, Any] | None = None
            if feedback:
                observation = {"type": "replan", "feedback": feedback}
            return {
                "done": False,
                "plan": new_plan,
                "observation": observation,
                "replanned": True,
            }

        if control_tool == _CONTROL_TOOL_READ_SKILL_FILE:
            skill_raw = decision.args.get("skill")
            path_raw = decision.args.get("path")
            if (
                not isinstance(skill_raw, str)
                or not skill_raw
                or not isinstance(path_raw, str)
                or not path_raw
            ):
                error_event = self._emit_control_action_error(
                    tool=control_tool,
                    args=decision.args,
                    message=(
                        "agent_read_skill_file requires args.skill and args.path "
                        "as non-empty strings."
                    ),
                )
                return {"done": False, "plan": current_plan, "observation": error_event}
            skill_name = skill_raw
            relative_path = path_raw
            if skill_name not in activated_skills:
                denied_event = self._emit_control_action_error(
                    tool=control_tool,
                    args={"skill": skill_name, "path": relative_path},
                    message=(
                        f"Skill file read denied: skill '{skill_name}' is not activated "
                        "for this plan."
                    ),
                )
                return {"done": False, "plan": current_plan, "observation": denied_event}
            read_result: ToolResult
            try:
                content = self._skills.read_file(skill_name, relative_path)
                read_result = ToolResult(exit_code=0, stdout=content, stderr="")
            except SkillsError as exc:
                read_result = ToolResult(
                    exit_code=1,
                    stdout="",
                    stderr=f"Skill file read error: {exc}",
                )
            action_event = {
                "type": "action",
                "tool": _CONTROL_TOOL_READ_SKILL_FILE,
                "args": {"skill": skill_name, "path": relative_path},
                "result": asdict(read_result),
            }
            self._send(action_event)
            return {"done": False, "plan": current_plan, "observation": action_event}

        error_event = self._emit_control_action_error(
            tool=control_tool,
            args=decision.args,
            message="Unsupported control tool.",
        )
        return {"done": False, "plan": current_plan, "observation": error_event}

    def _execute_tool(self, decision: ToolCall) -> ToolResult:
        return self._tools.execute(decision)

    def _handle_spawn_subagent(
        self,
        *,
        decision: ToolCall,
        current_plan: Any,
    ) -> dict[str, Any]:
        args = decision.args if isinstance(decision.args, dict) else {}
        if not self._subagents_enabled:
            return self._emit_subagent_observation(
                args=args,
                envelope={
                    "success": False,
                    "status": "disabled",
                    "reason": (
                        "subagents are disabled; both tools.spawn_subagent and "
                        "subagents.enabled must be true."
                    ),
                },
                plan=current_plan,
            )

        request, error = self._build_subagent_request(args)
        if request is None:
            return self._emit_subagent_observation(
                args=args,
                envelope={
                    "success": False,
                    "status": "invalid_request",
                    "reason": error or "invalid spawn_subagent request.",
                },
                plan=current_plan,
            )

        max_children = _read_pos_int_field(
            self._subagents_settings, "max_children_per_task", 3
        )
        if self._subagent_children_spawned >= max_children:
            return self._emit_subagent_observation(
                args=args,
                envelope={
                    "success": False,
                    "status": "invalid_request",
                    "reason": (
                        f"max_children_per_task ({max_children}) reached for this task."
                    ),
                },
                plan=current_plan,
            )

        runner = self._resolve_subagent_runner()
        if runner is None:
            return self._emit_subagent_observation(
                args=args,
                envelope={
                    "success": False,
                    "status": "child_failed",
                    "reason": "subagent runtime is not available in this sandbox.",
                },
                plan=current_plan,
            )

        self._subagent_children_spawned += 1
        try:
            envelope = runner.run(request)
        except Exception as exc:  # child failures are observations, never parent crashes
            envelope = {"success": False, "status": "child_failed", "reason": str(exc)}
        return self._emit_subagent_observation(args=args, envelope=envelope, plan=current_plan)

    def _resolve_subagent_runner(self) -> SubagentRunnerProtocol | None:
        if self._subagent_runner is not None:
            return self._subagent_runner
        if self._llm is None:
            return None
        from agent.subagents import SubagentRunner

        return SubagentRunner(
            llm_runtime=self._llm,
            broker=self._broker,
            skills_dir=self._skills_dir,
            base_config=self._agent_config or {},
            parent_enabled_tools=self._tools.list_enabled(),
            parent_session_id=self._active_session_id,
            output_root=str(self._output_dir / "subagents"),
            limits=self._subagents_settings,
        )

    def _build_subagent_request(
        self,
        args: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None]:
        task = args.get("task")
        if not isinstance(task, str) or not task.strip():
            return None, "spawn_subagent requires args.task as a non-empty string."

        context = args.get("context", "")
        if context is None:
            context = ""
        if not isinstance(context, str):
            return None, "spawn_subagent args.context must be a string when provided."

        expected_output = args.get("expected_output", "")
        if expected_output is None:
            expected_output = ""
        if not isinstance(expected_output, str):
            return None, "spawn_subagent args.expected_output must be a string when provided."

        allowed_tools, allowed_error = self._validate_subagent_allowed_tools(
            args.get("allowed_tools")
        )
        if allowed_error is not None:
            return None, allowed_error

        referenced_skills, skills_error = self._validate_subagent_referenced_skills(
            args.get("referenced_skills")
        )
        if skills_error is not None:
            return None, skills_error

        max_iterations, iter_error = self._clamp_subagent_int(
            args.get("max_iterations"),
            cap=_read_pos_int_field(self._subagents_settings, "max_iterations", 20),
            field="max_iterations",
        )
        if iter_error is not None:
            return None, iter_error

        timeout_seconds, timeout_error = self._clamp_subagent_int(
            args.get("timeout_seconds"),
            cap=_read_pos_int_field(self._subagents_settings, "timeout_seconds", 600),
            field="timeout_seconds",
        )
        if timeout_error is not None:
            return None, timeout_error

        max_context_chars = _read_pos_int_field(
            self._subagents_settings, "max_context_chars", 20000
        )
        return {
            "task": task.strip(),
            "context": _truncate_to(context, max_context_chars),
            "expected_output": expected_output,
            "allowed_tools": allowed_tools,
            "referenced_skills": referenced_skills,
            "max_iterations": max_iterations,
            "timeout_seconds": timeout_seconds,
        }, None

    def _validate_subagent_allowed_tools(
        self,
        raw: Any,
    ) -> tuple[list[str] | None, str | None]:
        parent_tools = set(self._tools.list_enabled())
        if raw is None:
            return sorted(parent_tools), None
        if not isinstance(raw, list):
            return None, "spawn_subagent args.allowed_tools must be an array of tool names."
        result: list[str] = []
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                return None, "spawn_subagent args.allowed_tools must contain only tool names."
            name = entry.strip()
            if name == _SPAWN_SUBAGENT_TOOL_NAME:
                return None, "spawn_subagent cannot be delegated to a child (no recursion)."
            if name not in parent_tools:
                return None, (
                    "spawn_subagent args.allowed_tools includes a tool not enabled for "
                    f"the parent: {name}."
                )
            if name not in result:
                result.append(name)
        return sorted(result), None

    def _validate_subagent_referenced_skills(
        self,
        raw: Any,
    ) -> tuple[list[str] | None, str | None]:
        if raw is None:
            return [], None
        if not isinstance(raw, list):
            return None, "spawn_subagent args.referenced_skills must be an array of skill names."
        result: list[str] = []
        for entry in raw:
            if not isinstance(entry, str) or not entry.strip():
                return None, (
                    "spawn_subagent args.referenced_skills must contain only skill names."
                )
            name = entry.strip()
            if name not in result:
                result.append(name)
        return result, None

    def _clamp_subagent_int(
        self,
        raw: Any,
        *,
        cap: int,
        field: str,
    ) -> tuple[int | None, str | None]:
        if raw is None:
            return cap, None
        if isinstance(raw, bool) or not isinstance(raw, int):
            return None, f"spawn_subagent args.{field} must be a positive integer when provided."
        if raw <= 0:
            return None, f"spawn_subagent args.{field} must be a positive integer when provided."
        return min(raw, cap), None

    def _emit_subagent_observation(
        self,
        *,
        args: dict[str, Any],
        envelope: dict[str, Any],
        plan: Any,
    ) -> dict[str, Any]:
        shaped = self._shape_subagent_envelope(envelope)
        wrapped = wrap_external_data(shaped)
        result = ToolResult(
            exit_code=0 if bool(envelope.get("success")) else 1,
            stdout=wrapped,
            stderr="",
        )
        action_event = {
            "type": "action",
            "tool": _SPAWN_SUBAGENT_TOOL_NAME,
            "args": dict(args),
            "result": asdict(result),
        }
        self._send(action_event)
        return {"done": False, "plan": plan, "observation": action_event}

    def _shape_subagent_envelope(self, envelope: Any) -> dict[str, Any]:
        """Reduce a child result envelope to the bounded view the parent observes."""
        if not isinstance(envelope, dict):
            return {
                "success": False,
                "status": "child_failed",
                "reason": "subagent returned an invalid result.",
            }
        limit = _read_pos_int_field(self._subagents_settings, "max_result_chars", 20000)
        shaped: dict[str, Any] = {
            "success": bool(envelope.get("success")),
            "status": str(envelope.get("status", "child_failed")),
        }
        for key in ("reason", "reply", "state_summary"):
            if key in envelope and envelope.get(key) is not None:
                shaped[key] = _truncate_to(str(envelope.get(key)), limit)
        files = envelope.get("files")
        if isinstance(files, list):
            shaped["files"] = [
                {
                    "path": str(item.get("path", "")),
                    "size_bytes": int(item.get("size_bytes", 0) or 0),
                }
                for item in files
                if isinstance(item, dict)
            ]
        events = envelope.get("events_summary")
        if isinstance(events, list):
            shaped["events_summary"] = events[:_SUBAGENT_EVENTS_SUMMARY_MAX]
        return shaped

    def _emit_control_action_error(
        self,
        *,
        tool: str,
        args: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        action_event = {
            "type": "action",
            "tool": tool,
            "args": dict(args),
            "result": asdict(ToolResult(exit_code=1, stdout="", stderr=message)),
        }
        self._send(action_event)
        return action_event

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

    def _require_llm_config(self) -> dict[str, Any]:
        llm_config = self._agent_config.get("llm") if isinstance(self._agent_config, dict) else None
        if not isinstance(llm_config, dict):
            raise AgentError("Agent config is missing required llm mapping.")
        return dict(llm_config)

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
            "Available tools:",
            json.dumps(self._tools.list_enabled(), ensure_ascii=True, indent=2),
            "",
            "Available skills:",
            json.dumps(self._skills.index(), ensure_ascii=True, indent=2),
            "",
            "Return JSON with keys: goal (string), steps (array of strings), "
            "referenced_skills (array of skill names, may be empty).",
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
        activated_skills: dict[str, dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Build execution-phase messages, enforcing token budget."""
        recent_history = self._recent_history(history)
        summary = self._history_summary or ""

        while True:
            user_payload = {
                "goal": goal,
                "plan": plan,
                "enabled_tools": self._tools.list_enabled(),
                "tool_schemas": self._execution_action_surface,
                "activated_skills": activated_skills,
                "history_summary": summary,
                "recent_history": recent_history,
                "output_instruction": "Place any files for the user in /output/.",
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"{EXECUTION_SYSTEM_PROMPT}\nConfigured integrations: "
                        f"{', '.join(self._integrations) if self._integrations else 'none'}"
                    ),
                },
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
            ]
            try:
                token_count = self._require_llm().count_tokens(messages)
            except LLMRuntimeError:
                token_count = _estimate_message_tokens(messages)
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


def _build_execution_action_surface(
    capability_tool_schemas: list[dict[str, Any]],
    *,
    control_schemas: list[dict[str, Any]] | None = None,
    extra_schemas: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    controls = _CONTROL_TOOL_SCHEMAS if control_schemas is None else control_schemas
    surface: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for schema in [*capability_tool_schemas, *controls, *(extra_schemas or [])]:
        name_raw = schema.get("name")
        params = schema.get("parameters")
        if not isinstance(name_raw, str) or not name_raw.strip():
            raise AgentError("Execution action schema entry is missing a non-empty name.")
        name = name_raw.strip()
        if not _PROVIDER_SAFE_TOOL_NAME_RE.fullmatch(name):
            raise AgentError(
                f"Execution action schema name '{name}' is not provider-safe."
            )
        if name in seen_names:
            raise AgentError(f"Duplicate execution action schema name: {name}")
        if not isinstance(params, dict):
            raise AgentError(
                f"Execution action schema '{name}' is missing object parameters."
            )
        description = schema.get("description")
        if not isinstance(description, str) or not description.strip():
            description = f"Run {name}."
        surface.append(
            {
                "name": name,
                "description": description,
                "parameters": params,
            }
        )
        seen_names.add(name)
    return surface


def _read_skills_max_file_chars(agent_config: dict[str, Any] | None) -> int:
    if not isinstance(agent_config, dict):
        return 20000
    skills_cfg = agent_config.get("skills")
    if not isinstance(skills_cfg, dict):
        return 20000
    raw = skills_cfg.get("max_file_chars", 20000)
    if isinstance(raw, bool):
        return 20000
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 20000
    if value <= 0:
        return 20000
    return value


def _read_skills_directory(agent_config: dict[str, Any]) -> str | None:
    skills_cfg = agent_config.get("skills")
    if not isinstance(skills_cfg, dict):
        return None
    raw = skills_cfg.get("directory")
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    return value


def _read_loop_max_iterations(agent_config: dict[str, Any], *, fallback: int) -> int:
    loop_cfg = agent_config.get("loop")
    if isinstance(loop_cfg, dict):
        raw = loop_cfg.get("max_iterations")
    else:
        raw = agent_config.get("max_iterations")
    if isinstance(raw, bool):
        return fallback
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback
    if value <= 0:
        return fallback
    return value


def _read_context_int(agent_config: dict[str, Any], key: str, *, fallback: int) -> int:
    context_cfg = agent_config.get("context")
    if not isinstance(context_cfg, dict):
        return fallback
    raw = context_cfg.get(key)
    if isinstance(raw, bool):
        return fallback
    if raw is None:
        return fallback
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback
    if value <= 0:
        return fallback
    return value


def _read_subagents_settings(agent_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(agent_config, dict):
        return {}
    settings = agent_config.get("subagents")
    return dict(settings) if isinstance(settings, dict) else {}


def _read_subagents_enabled(agent_config: dict[str, Any] | None) -> bool:
    if not isinstance(agent_config, dict):
        return False
    tools_cfg = agent_config.get("tools")
    spawn_toggle = (
        bool(tools_cfg.get("spawn_subagent", False)) if isinstance(tools_cfg, dict) else False
    )
    subagents_cfg = agent_config.get("subagents")
    if isinstance(subagents_cfg, dict):
        enabled = bool(subagents_cfg.get("enabled", False))
    else:
        enabled = False
    return spawn_toggle and enabled


def _read_pos_int_field(mapping: dict[str, Any], key: str, fallback: int) -> int:
    if not isinstance(mapping, dict):
        return fallback
    raw = mapping.get(key, fallback)
    if isinstance(raw, bool):
        return fallback
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def _truncate_to(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated {len(text) - limit} chars]..."


def _read_host_service_llm_timeout(agent_config: dict[str, Any]) -> float:
    host_services = agent_config.get("host_services")
    if not isinstance(host_services, dict):
        return 120.0
    raw_timeout = host_services.get("llm_timeout_seconds", 120)
    if isinstance(raw_timeout, bool):
        return 120.0
    try:
        timeout = float(raw_timeout)
    except (TypeError, ValueError):
        return 120.0
    if timeout <= 0:
        return 120.0
    return timeout


def _estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=True, default=str)
    return max(1, len(serialized) // 4)


def _load_agent_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AgentError(f"Agent config file was not found: {path}")
    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise AgentError(f"Agent config in {path} must be a JSON object.")
    if "llm" in parsed:
        llm = parsed.get("llm")
        if not isinstance(llm, dict):
            raise AgentError(f"Agent config in {path} must contain llm as an object.")
        return parsed
    if _looks_like_llm_config(parsed):
        return {"llm": parsed}
    if "host_services" in parsed:
        host_services = parsed.get("host_services")
        if not isinstance(host_services, dict):
            raise AgentError(
                f"Agent config in {path} must contain host_services as an object."
            )
        return parsed
    raise AgentError(f"Agent config in {path} must contain an llm object or host_services object.")


def _looks_like_llm_config(payload: dict[str, Any]) -> bool:
    required = {"model", "api_key"}
    return required.issubset(payload.keys())


def _build_default_llm_runtime(llm_config: dict[str, Any]) -> LLMRuntime:
    from agent.llm import LLMClient

    return LLMClient.from_config(llm_config)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m agent.agent")
    parser.add_argument("--vsock-port", type=int, default=None, help="Guest vsock listen port.")
    parser.add_argument("--skills-dir", type=str, default="/opt/strangeclaw/skills")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--token-budget", type=int, default=4000)
    parser.add_argument("--summary-threshold", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="/output")
    parser.add_argument("--agent-config-path", type=str, default="/run/strangeclaw/config.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for guest execution."""
    args = _parse_args(argv)
    if args.vsock_port is None:
        raise ValueError("Missing required --vsock-port for guest execution.")
    config_path = args.agent_config_path

    agent_config = _load_agent_config_file(Path(config_path))
    _validate_guest_proxy_config(agent_config, Path(config_path))
    transport = VsockTransport(guest_port=int(args.vsock_port))
    try:
        broker = BrokerClient(
            mode="fire",
            send_fn=transport.send,
            receive_fn=transport.receive,
            service_timeouts={
                "_default": 60.0,
                "broker": 60.0,
                "llm": _read_host_service_llm_timeout(agent_config),
            },
        )
        llm_runtime = LLMProxyRuntime(broker)
        agent = Agent(
            transport=transport,
            skills_dir=str(args.skills_dir),
            max_iterations=int(args.max_iterations),
            output_dir=str(args.output_dir),
            agent_config=agent_config,
            agent_config_path=str(config_path),
            token_budget=int(args.token_budget),
            summary_threshold=int(args.summary_threshold),
            llm_runtime=llm_runtime,
            broker=broker,
        )
        agent.run_forever()
    finally:
        transport.close()


def _validate_guest_proxy_config(agent_config: dict[str, Any], path: Path) -> None:
    """Reject legacy direct-LLM guest configs for the Fire proxy entrypoint."""
    if "llm" in agent_config:
        raise AgentError(
            f"Guest proxy config in {path} must not contain an llm block. "
            "LLM provider configuration is host-only and served through the llm host service."
        )
    host_services = agent_config.get("host_services")
    if not isinstance(host_services, dict):
        raise AgentError(f"Guest proxy config in {path} must contain host_services.")


if __name__ == "__main__":
    main()
