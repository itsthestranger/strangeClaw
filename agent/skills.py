"""Agent Skills discovery and progressive-disclosure file access."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)


class SkillsError(RuntimeError):
    """Raised when skill discovery or access fails."""


@dataclass(slots=True)
class SkillDefinition:
    """Discovered Agent Skill metadata and resources."""

    name: str
    description: str
    body: str
    path: Path
    files: list[str]


class Skills:
    """Context-only Agent Skills loader."""

    def __init__(self, skills_dir: str, max_file_chars: int = 20000) -> None:
        root = Path(skills_dir).expanduser()
        if not root.is_dir():
            raise SkillsError(f"Skills directory does not exist: {root}")
        if max_file_chars <= 0:
            raise SkillsError("max_file_chars must be greater than zero.")

        self._skills_dir = root
        self._max_file_chars = max_file_chars
        self._skills = self._discover(root)

    def index(self) -> list[dict[str, str]]:
        """Stage 1 — discovery: return name/description for each skill."""
        return [
            {
                "name": definition.name,
                "description": definition.description,
            }
            for _, definition in sorted(self._skills.items())
        ]

    def get_doc(self, skill_name: str) -> dict[str, Any]:
        """Stage 2 — activation: return body-only SKILL.md and file manifest."""
        definition = self._skills.get(skill_name)
        if definition is None:
            raise SkillsError(f"Unknown skill: {skill_name}")
        return {
            "skill_md": definition.body,
            "files": list(definition.files),
        }

    def read_file(self, skill_name: str, relative_path: str) -> str:
        """Stage 3 — execution-time file read with traversal protection."""
        definition = self._skills.get(skill_name)
        if definition is None:
            raise SkillsError(f"Unknown skill: {skill_name}")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise SkillsError("relative_path must be a non-empty string.")

        requested = Path(relative_path.strip())
        if requested.is_absolute():
            raise SkillsError("read_file path must be relative.")
        if ".." in requested.parts:
            raise SkillsError("read_file path traversal is not allowed.")

        skill_root = definition.path.resolve()
        target = (definition.path / requested).resolve()
        if not _is_within(skill_root, target):
            raise SkillsError("read_file target must stay within skill directory.")
        if not target.is_file():
            raise SkillsError(f"Skill file not found: {relative_path}")

        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise SkillsError(f"Skill file is not UTF-8 decodable: {relative_path}") from exc

        if len(content) <= self._max_file_chars:
            return content
        return (
            f"{content[:self._max_file_chars]}\n\n"
            f"[... truncated, original {len(content)} chars ...]"
        )

    def _discover(self, skills_dir: Path) -> dict[str, SkillDefinition]:
        discovered: dict[str, SkillDefinition] = {}
        for skill_dir in sorted(path for path in skills_dir.iterdir() if path.is_dir()):
            skill_doc_path = skill_dir / "SKILL.md"
            if not skill_doc_path.is_file():
                continue

            try:
                metadata, body = _parse_skill_markdown(skill_doc_path)
                name = _required_non_empty_string(metadata, "name", skill_doc_path)
                description = _required_non_empty_string(metadata, "description", skill_doc_path)
                files = _skill_file_manifest(skill_dir)
            except SkillsError as exc:
                LOGGER.warning("Skipping skill '%s': %s", skill_dir.name, exc)
                continue

            if name in discovered:
                LOGGER.warning(
                    "Skipping skill '%s': duplicate skill name '%s' already loaded.",
                    skill_dir.name,
                    name,
                )
                continue

            discovered[name] = SkillDefinition(
                name=name,
                description=description,
                body=body,
                path=skill_dir,
                files=files,
            )
        return discovered


def _parse_skill_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        raise SkillsError(
            f"Skill frontmatter is required and must start with '---': {path}"
        )

    end_index: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break
    if end_index is None:
        raise SkillsError(f"Unterminated skill frontmatter in {path}")

    frontmatter_text = "\n".join(lines[1:end_index])
    body = "\n".join(lines[end_index + 1 :])
    if text.endswith("\n"):
        body = f"{body}\n" if body else ""

    try:
        parsed = yaml.safe_load(frontmatter_text)
    except yaml.YAMLError as exc:
        raise SkillsError(f"Invalid YAML frontmatter in {path}: {exc}") from exc

    if not isinstance(parsed, dict):
        raise SkillsError(f"Skill frontmatter must be a YAML mapping in {path}")
    return parsed, body


def _required_non_empty_string(metadata: dict[str, Any], key: str, path: Path) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillsError(f"Skill frontmatter missing required '{key}' in {path}")
    return value.strip()


def _skill_file_manifest(skill_root: Path) -> list[str]:
    resolved_root = skill_root.resolve()
    manifest: list[str] = []

    for dirname in ("scripts", "references", "assets"):
        subdir = skill_root / dirname
        if not subdir.is_dir():
            continue

        for entry in sorted(subdir.rglob("*")):
            if not entry.is_file():
                continue
            resolved = entry.resolve()
            if not _is_within(resolved_root, resolved):
                LOGGER.warning(
                    "Skipping manifest file outside skill directory: %s",
                    entry,
                )
                continue
            manifest.append(resolved.relative_to(resolved_root).as_posix())

    return sorted(manifest)


def _is_within(root: Path, path: Path) -> bool:
    return path == root or root in path.parents
