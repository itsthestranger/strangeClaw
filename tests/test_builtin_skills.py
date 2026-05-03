"""Repository-level checks for the skills directory state."""

from __future__ import annotations

from pathlib import Path

from agent.skills import Skills, SkillsError


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def test_repository_web_research_skill_is_discoverable() -> None:
    skills = Skills(str(_skills_root()))
    assert skills.index() == [
        {
            "name": "web-research",
            "description": (
                "Research a topic by formulating searches, evaluating results, "
                "and synthesizing well-sourced findings."
            ),
        }
    ]

    doc = skills.get_doc("web-research")
    assert doc["files"] == []
    skill_md = doc["skill_md"]
    assert "`web_search`" in skill_md
    assert "`web_fetch`" in skill_md
    assert "Source Evaluation" in skill_md
    assert "Iterate" in skill_md or "iterate" in skill_md
    assert "Synthesis Format" in skill_md


def test_unknown_skill_lookup_raises() -> None:
    skills = Skills(str(_skills_root()))
    try:
        skills.get_doc("does-not-exist")
        raise AssertionError("Expected unknown skill lookup to fail")
    except SkillsError:
        pass
