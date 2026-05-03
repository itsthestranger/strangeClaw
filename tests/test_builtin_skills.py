"""Repository-level checks for the skills directory state."""

from __future__ import annotations

from pathlib import Path

from agent.skills import Skills, SkillsError


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def test_repository_skills_are_discoverable() -> None:
    skills = Skills(str(_skills_root()))
    assert skills.index() == [
        {
            "name": "github",
            "description": (
                "Work with the GitHub REST API using configured auth, issues, "
                "pull requests, and repository contents."
            ),
        },
        {
            "name": "notion",
            "description": (
                "Work with the Notion API using configured auth, data sources, "
                "pages, querying, and updates."
            ),
        },
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


def test_repository_notion_skill_manifest_and_reference() -> None:
    skills = Skills(str(_skills_root()))
    doc = skills.get_doc("notion")

    assert doc["files"] == ["references/notion.md"]
    skill_md = doc["skill_md"]
    assert "`http_request`" in skill_md
    assert "`agent_read_skill_file`" in skill_md
    assert 'integration: "notion"' in skill_md
    assert "integrations.notion.token" in skill_md

    notion = skills.read_file("notion", "references/notion.md")
    assert "https://api.notion.com" in notion
    assert "POST /v1/data_sources/{data_source_id}/query" in notion
    assert "Notion-Version" in notion
    assert '"integration": "notion"' in notion


def test_repository_github_skill_manifest_and_reference() -> None:
    skills = Skills(str(_skills_root()))
    doc = skills.get_doc("github")

    assert doc["files"] == ["references/github.md"]
    skill_md = doc["skill_md"]
    assert "`http_request`" in skill_md
    assert "`agent_read_skill_file`" in skill_md
    assert 'integration: "github"' in skill_md
    assert "integrations.github.token" in skill_md

    github = skills.read_file("github", "references/github.md")
    assert "https://api.github.com" in github
    assert "POST /repos/{owner}/{repo}/issues" in github
    assert "GET /repos/{owner}/{repo}/pulls" in github
    assert "GET /repos/{owner}/{repo}/contents/{path}" in github
    assert '"integration": "github"' in github


def test_unknown_skill_lookup_raises() -> None:
    skills = Skills(str(_skills_root()))
    try:
        skills.get_doc("does-not-exist")
        raise AssertionError("Expected unknown skill lookup to fail")
    except SkillsError:
        pass
