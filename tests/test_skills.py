"""Tests for Agent Skills discovery and context-file access."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from agent.skills import Skills, SkillsError


def _write_skill(
    root: Path,
    dir_name: str,
    *,
    name: str,
    description: str,
    body: str,
) -> Path:
    skill_dir = root / dir_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "version: 1.0.0\n"
        "---\n"
        f"{body}\n"
    )
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
    return skill_dir


def test_skills_discovery_index_doc_and_directory_changes(tmp_path: Path) -> None:
    alpha = _write_skill(
        tmp_path,
        "alpha-dir",
        name="alpha",
        description="Alpha description",
        body="# Alpha\n\nAlpha body.",
    )
    (alpha / "references").mkdir()
    (alpha / "references" / "guide.md").write_text("alpha guide", encoding="utf-8")

    _write_skill(
        tmp_path,
        "beta-dir",
        name="beta",
        description="Beta description",
        body="# Beta\n\nBeta body.",
    )
    (tmp_path / "missing-skill-md").mkdir()

    skills = Skills(str(tmp_path))
    assert skills.index() == [
        {"name": "alpha", "description": "Alpha description"},
        {"name": "beta", "description": "Beta description"},
    ]

    alpha_doc = skills.get_doc("alpha")
    assert alpha_doc["skill_md"].startswith("# Alpha")
    assert alpha_doc["files"] == ["references/guide.md"]

    _write_skill(
        tmp_path,
        "gamma-dir",
        name="gamma",
        description="Gamma description",
        body="# Gamma\n\nGamma body.",
    )
    updated = Skills(str(tmp_path))
    assert [entry["name"] for entry in updated.index()] == ["alpha", "beta", "gamma"]

    shutil.rmtree(tmp_path / "alpha-dir")
    removed = Skills(str(tmp_path))
    assert [entry["name"] for entry in removed.index()] == ["beta", "gamma"]


def test_skills_skips_missing_or_invalid_frontmatter(tmp_path: Path) -> None:
    valid = _write_skill(
        tmp_path,
        "valid-skill",
        name="valid",
        description="Valid description",
        body="Valid body.",
    )
    del valid

    no_frontmatter_dir = tmp_path / "no-frontmatter"
    no_frontmatter_dir.mkdir()
    (no_frontmatter_dir / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")

    malformed_yaml_dir = tmp_path / "bad-yaml"
    malformed_yaml_dir.mkdir()
    (malformed_yaml_dir / "SKILL.md").write_text(
        "---\nname: [unterminated\ndescription: bad\n---\nBody\n",
        encoding="utf-8",
    )

    missing_name_dir = tmp_path / "missing-name"
    missing_name_dir.mkdir()
    (missing_name_dir / "SKILL.md").write_text(
        "---\ndescription: missing name\n---\nBody\n",
        encoding="utf-8",
    )

    missing_description_dir = tmp_path / "missing-description"
    missing_description_dir.mkdir()
    (missing_description_dir / "SKILL.md").write_text(
        "---\nname: missing-description\n---\nBody\n",
        encoding="utf-8",
    )

    skills = Skills(str(tmp_path))
    assert skills.index() == [{"name": "valid", "description": "Valid description"}]


def test_skills_manifest_uses_standard_subdirectories_only(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "skill",
        name="api-integration",
        description="API integration skill",
        body="Body.",
    )

    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "guide.md").write_text("guide", encoding="utf-8")
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (skill_dir / "assets").mkdir()
    (skill_dir / "assets" / "template.txt").write_text("x", encoding="utf-8")
    (skill_dir / "notes").mkdir()
    (skill_dir / "notes" / "ignored.txt").write_text("ignore me", encoding="utf-8")

    skills = Skills(str(tmp_path))
    bundle = skills.get_doc("api-integration")
    assert bundle["files"] == [
        "assets/template.txt",
        "references/guide.md",
        "scripts/run.sh",
    ]


def test_read_file_returns_content_and_truncates(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "skill",
        name="reader",
        description="Read files",
        body="Body.",
    )
    (skill_dir / "references").mkdir()
    content_path = skill_dir / "references" / "data.txt"
    content_path.write_text("abcdef", encoding="utf-8")

    skills = Skills(str(tmp_path), max_file_chars=4)
    expected = "abcd\n\n[... truncated, original 6 chars ...]"
    assert skills.read_file("reader", "references/data.txt") == expected


def test_read_file_rejects_traversal_absolute_and_missing_paths(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "skill",
        name="reader",
        description="Read files",
        body="Body.",
    )
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "ok.txt").write_text("ok", encoding="utf-8")

    skills = Skills(str(tmp_path))

    with pytest.raises(SkillsError, match="path traversal"):
        skills.read_file("reader", "../outside.txt")

    with pytest.raises(SkillsError, match="must be relative"):
        skills.read_file("reader", str((tmp_path / "absolute.txt").resolve()))

    with pytest.raises(SkillsError, match="not found"):
        skills.read_file("reader", "references/missing.txt")


def test_read_file_rejects_symlink_escape(tmp_path: Path) -> None:
    if not hasattr(os, "symlink"):
        pytest.skip("symlink not supported on this platform")

    skill_dir = _write_skill(
        tmp_path,
        "skill",
        name="reader",
        description="Read files",
        body="Body.",
    )
    (skill_dir / "references").mkdir()

    external = tmp_path / "external.txt"
    external.write_text("secret", encoding="utf-8")
    link = skill_dir / "references" / "outside.txt"
    link.symlink_to(external)

    skills = Skills(str(tmp_path))
    with pytest.raises(SkillsError, match="must stay within skill directory"):
        skills.read_file("reader", "references/outside.txt")


def test_read_file_reports_utf8_decode_errors(tmp_path: Path) -> None:
    skill_dir = _write_skill(
        tmp_path,
        "skill",
        name="reader",
        description="Read files",
        body="Body.",
    )
    (skill_dir / "assets").mkdir()
    binary_file = skill_dir / "assets" / "blob.bin"
    binary_file.write_bytes(b"\xff\xfe\x00")

    skills = Skills(str(tmp_path))
    with pytest.raises(SkillsError, match="UTF-8 decodable"):
        skills.read_file("reader", "assets/blob.bin")


def test_skills_has_no_execute_api(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "skill",
        name="alpha",
        description="Alpha description",
        body="Alpha body",
    )
    skills = Skills(str(tmp_path))
    assert not hasattr(skills, "execute")
