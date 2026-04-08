"""Skill discovery and execution."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema  # type: ignore[import-untyped]
import yaml


@dataclass(slots=True)
class ToolResult:
    """Result returned by a skill invocation."""

    exit_code: int
    stdout: str
    stderr: str


class SkillsError(RuntimeError):
    """Raised when skill discovery or execution fails."""


@dataclass(slots=True)
class SkillAction:
    """Runtime metadata for one action."""

    args_schema: dict[str, Any]
    invoke: list[str]
    timeout_seconds: float


@dataclass(slots=True)
class SkillDefinition:
    """Runtime metadata for one skill."""

    name: str
    description: str
    doc: str
    path: Path
    actions: dict[str, SkillAction]


class Skills:
    """Skill loader/executor."""

    def __init__(self, skills_dir: str, default_timeout_seconds: float = 60.0) -> None:
        """Initialize skill discovery."""
        root = Path(skills_dir).expanduser()
        if not root.is_dir():
            raise SkillsError(f"Skills directory does not exist: {root}")
        if default_timeout_seconds <= 0:
            raise SkillsError("default_timeout_seconds must be greater than zero.")

        self._skills_dir = root
        self._default_timeout_seconds = default_timeout_seconds
        self._skills = self._discover(root)

    def index(self) -> list[dict[str, str]]:
        """Return a one-line index for all skills."""
        items: list[dict[str, str]] = []
        for skill_name in sorted(self._skills):
            definition = self._skills[skill_name]
            items.append({"name": definition.name, "description": definition.description})
        return items

    def get_doc(self, skill_name: str) -> str:
        """Return full SKILL.md content for one skill."""
        definition = self._skills.get(skill_name)
        if definition is None:
            raise SkillsError(f"Unknown skill: {skill_name}")
        return definition.doc

    def execute(self, tool_call: dict[str, Any]) -> ToolResult:
        """Validate and execute one tool call."""
        skill_name, action_name, args = _normalize_tool_call(tool_call)

        definition = self._skills.get(skill_name)
        if definition is None:
            raise SkillsError(f"Unknown skill: {skill_name}")

        action = definition.actions.get(action_name)
        if action is None:
            raise SkillsError(f"Unknown action '{action_name}' for skill '{skill_name}'")

        try:
            jsonschema.validate(instance=args, schema=action.args_schema)
        except jsonschema.ValidationError as exc:
            raise SkillsError(
                f"Invalid args for {skill_name}.{action_name}: {exc.message}"
            ) from exc

        command = _render_invoke(action.invoke, args)
        try:
            completed = subprocess.run(
                command,
                cwd=definition.path,
                capture_output=True,
                text=True,
                check=False,
                timeout=action.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            timeout_message = f"Command timed out after {action.timeout_seconds:.1f}s."
            stderr = f"{stderr}\n{timeout_message}" if stderr else timeout_message
            return ToolResult(
                exit_code=124,
                stdout=_truncate_output(stdout),
                stderr=_truncate_output(stderr),
            )
        except OSError as exc:
            command_text = shlex.join(command)
            raise SkillsError(
                f"Failed to execute invoke command for {skill_name}.{action_name}: "
                f"{command_text}: {exc}"
            ) from exc

        return ToolResult(
            exit_code=completed.returncode,
            stdout=_truncate_output(completed.stdout),
            stderr=_truncate_output(completed.stderr),
        )

    def _discover(self, skills_dir: Path) -> dict[str, SkillDefinition]:
        discovered: dict[str, SkillDefinition] = {}
        for skill_dir in sorted(path for path in skills_dir.iterdir() if path.is_dir()):
            skill_doc_path = skill_dir / "SKILL.md"
            skill_schema_path = skill_dir / "schema.json"
            if not skill_doc_path.is_file() or not skill_schema_path.is_file():
                continue

            doc_text = skill_doc_path.read_text(encoding="utf-8")
            description = _extract_description(doc_text, fallback=skill_dir.name)

            schema_payload: Any
            try:
                schema_payload = json.loads(skill_schema_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise SkillsError(f"Invalid JSON in {skill_schema_path}: {exc}") from exc
            if not isinstance(schema_payload, dict):
                raise SkillsError(f"Skill schema must be an object: {skill_schema_path}")

            actions = _parse_actions(
                schema_payload,
                skill_name=skill_dir.name,
                default_timeout_seconds=self._default_timeout_seconds,
            )
            discovered[skill_dir.name] = SkillDefinition(
                name=skill_dir.name,
                description=description,
                doc=doc_text,
                path=skill_dir,
                actions=actions,
            )
        return discovered


def _normalize_tool_call(tool_call: Any) -> tuple[str, str, dict[str, Any]]:
    if isinstance(tool_call, dict):
        raw_skill = tool_call.get("skill")
        raw_action = tool_call.get("action")
        raw_args = tool_call.get("args")
    else:
        raw_skill = getattr(tool_call, "skill", None)
        raw_action = getattr(tool_call, "action", None)
        raw_args = getattr(tool_call, "args", None)

    if not isinstance(raw_skill, str) or not raw_skill:
        raise SkillsError("tool_call.skill must be a non-empty string.")
    if not isinstance(raw_action, str) or not raw_action:
        raise SkillsError("tool_call.action must be a non-empty string.")
    if not isinstance(raw_args, dict):
        raise SkillsError("tool_call.args must be an object.")
    return raw_skill, raw_action, raw_args


def _parse_actions(
    schema_payload: dict[str, Any],
    *,
    skill_name: str,
    default_timeout_seconds: float,
) -> dict[str, SkillAction]:
    raw_actions = schema_payload.get("actions")
    if not isinstance(raw_actions, dict) or not raw_actions:
        raise SkillsError(
            f"Skill schema for '{skill_name}' must contain a non-empty 'actions' object."
        )

    actions: dict[str, SkillAction] = {}
    for action_name, action_payload in raw_actions.items():
        if not isinstance(action_name, str) or not action_name:
            raise SkillsError(f"Skill '{skill_name}' has an invalid action name.")
        if not isinstance(action_payload, dict):
            raise SkillsError(f"Action '{skill_name}.{action_name}' must be an object.")

        args_schema = action_payload.get("args_schema", action_payload.get("schema"))
        if not isinstance(args_schema, dict):
            raise SkillsError(
                f"Action '{skill_name}.{action_name}' must define an object args_schema/schema."
            )
        try:
            jsonschema.Draft202012Validator.check_schema(args_schema)
        except jsonschema.SchemaError as exc:
            raise SkillsError(
                f"Invalid schema for action '{skill_name}.{action_name}': {exc}"
            ) from exc

        invoke = _normalize_invoke(action_payload.get("invoke"), skill_name, action_name)
        timeout_raw = action_payload.get("timeout_seconds", default_timeout_seconds)
        if not isinstance(timeout_raw, int | float) or timeout_raw <= 0:
            raise SkillsError(
                f"Action '{skill_name}.{action_name}' timeout_seconds must be a positive number."
            )

        actions[action_name] = SkillAction(
            args_schema=args_schema,
            invoke=invoke,
            timeout_seconds=float(timeout_raw),
        )

    return actions


def _normalize_invoke(invoke_raw: Any, skill_name: str, action_name: str) -> list[str]:
    if isinstance(invoke_raw, str):
        command = shlex.split(invoke_raw)
    elif isinstance(invoke_raw, list):
        if not all(isinstance(item, str) for item in invoke_raw):
            raise SkillsError(
                f"Action '{skill_name}.{action_name}' invoke list must contain only strings."
            )
        command = invoke_raw
    else:
        raise SkillsError(f"Action '{skill_name}.{action_name}' must define an invoke command.")

    if not command:
        raise SkillsError(f"Action '{skill_name}.{action_name}' invoke command cannot be empty.")
    return command


def _render_invoke(invoke: list[str], args: dict[str, Any]) -> list[str]:
    substitutions = {key: _format_arg_value(value) for key, value in args.items()}
    rendered: list[str] = []
    for segment in invoke:
        try:
            rendered.append(segment.format_map(substitutions))
        except KeyError as exc:
            missing_key = str(exc).strip("'")
            raise SkillsError(
                f"Missing argument '{missing_key}' required by invoke command."
            ) from exc
    return rendered


def _format_arg_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _extract_description(doc_text: str, *, fallback: str) -> str:
    lines = doc_text.splitlines()
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() != "---":
                continue
            frontmatter = "\n".join(lines[1:idx])
            parsed = yaml.safe_load(frontmatter)
            if isinstance(parsed, dict):
                description = parsed.get("description")
                if isinstance(description, str) and description.strip():
                    return description.strip()
            break

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            line = line.lstrip("#").strip()
        if line:
            return line
    return fallback


def _truncate_output(text: str, *, chunk_size: int = 4000) -> str:
    if len(text) <= chunk_size * 2:
        return text

    omitted_chars = len(text) - (chunk_size * 2)
    return (
        f"{text[:chunk_size]}\n"
        f"...[truncated {omitted_chars} chars]...\n"
        f"{text[-chunk_size:]}"
    )
