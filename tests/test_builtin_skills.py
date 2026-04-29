"""Tests for built-in skill documents under the Agent Skills API."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.skills import Skills, SkillsError


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def test_builtin_skills_discoverable() -> None:
    skills = Skills(str(_skills_root()))
    index = skills.index()
    names = {entry["name"] for entry in index}
    assert {"shell", "web-search", "http-request"} <= names


def test_builtin_skill_docs_strip_frontmatter_and_include_manifest() -> None:
    skills = Skills(str(_skills_root()))

    shell_bundle = skills.get_doc("shell")
    assert shell_bundle["skill_md"].lstrip().startswith("# shell")
    assert "name: shell" not in shell_bundle["skill_md"]
    assert shell_bundle["files"] == []

    web_search_bundle = skills.get_doc("web-search")
    assert web_search_bundle["files"] == []


def test_builtin_skill_files_are_readable_via_stage3_read_file() -> None:
    skills = Skills(str(_skills_root()))

    search_script = skills.read_file("web-search", "search.py")
    assert "def" in search_script

    request_script = skills.read_file("http-request", "request.py")
    assert "def" in request_script


def test_builtin_skills_unknown_skill_raises() -> None:
    skills = Skills(str(_skills_root()))
    with pytest.raises(SkillsError, match="Unknown skill"):
        skills.get_doc("does-not-exist")
