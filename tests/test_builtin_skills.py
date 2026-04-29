"""Repository-level checks for the skills directory state."""

from __future__ import annotations

from pathlib import Path

from agent.skills import Skills, SkillsError


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def test_repository_skills_directory_starts_empty() -> None:
    skills = Skills(str(_skills_root()))
    assert skills.index() == []


def test_unknown_skill_lookup_raises() -> None:
    skills = Skills(str(_skills_root()))
    try:
        skills.get_doc("does-not-exist")
        raise AssertionError("Expected unknown skill lookup to fail")
    except SkillsError:
        pass
