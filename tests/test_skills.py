"""Tests for skill discovery and execution."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from agent.skills import Skills, SkillsError


def _write_skill(tmp_path: Path, name: str, doc: str, schema: dict[str, Any]) -> Path:
    skill_dir = tmp_path / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(doc, encoding="utf-8")
    (skill_dir / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    return skill_dir


def test_skills_discovery_index_doc_and_directory_changes(tmp_path: Path) -> None:
    _write_skill(
        tmp_path,
        "alpha",
        "---\nname: alpha\ndescription: Alpha description\n---\n# Alpha\n",
        {
            "actions": {
                "run": {
                    "args_schema": {"type": "object", "additionalProperties": False},
                    "invoke": ["python3", "-c", "print('ok')"],
                }
            }
        },
    )
    (tmp_path / "missing-schema").mkdir()
    (tmp_path / "missing-schema" / "SKILL.md").write_text("# Missing schema", encoding="utf-8")

    skills = Skills(str(tmp_path))
    assert skills.index() == [{"name": "alpha", "description": "Alpha description"}]
    assert "Alpha" in skills.get_doc("alpha")

    _write_skill(
        tmp_path,
        "beta",
        "# Beta\n",
        {
            "actions": {
                "run": {
                    "args_schema": {"type": "object", "additionalProperties": False},
                    "invoke": ["python3", "-c", "print('beta')"],
                }
            }
        },
    )
    updated = Skills(str(tmp_path))
    assert [entry["name"] for entry in updated.index()] == ["alpha", "beta"]

    shutil.rmtree(tmp_path / "alpha")
    removed = Skills(str(tmp_path))
    assert removed.index() == [{"name": "beta", "description": "Beta"}]


def test_execute_validates_args_and_captures_stdout_stderr(tmp_path: Path) -> None:
    skills = Skills(
        str(
            _write_skill(
                tmp_path,
                "echo",
                "# Echo\n",
                {
                    "actions": {
                        "run": {
                            "args_schema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                                "additionalProperties": False,
                            },
                            "invoke": [
                                "python3",
                                "-c",
                                "import sys; print(sys.argv[1]); print('err', file=sys.stderr)",
                                "{text}",
                            ],
                        }
                    }
                },
            ).parent
        )
    )

    result = skills.execute({"skill": "echo", "action": "run", "args": {"text": "hello"}})
    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.stderr == "err\n"

    with pytest.raises(SkillsError, match="Invalid args"):
        skills.execute({"skill": "echo", "action": "run", "args": {}})


def test_execute_truncates_long_output(tmp_path: Path) -> None:
    skills = Skills(
        str(
            _write_skill(
                tmp_path,
                "long-output",
                "# Long output\n",
                {
                    "actions": {
                        "run": {
                            "args_schema": {"type": "object", "additionalProperties": False},
                            "invoke": [
                                "python3",
                                "-c",
                                (
                                    "import sys; "
                                    "sys.stdout.write('a' * 9001); "
                                    "sys.stderr.write('b' * 9002)"
                                ),
                            ],
                        }
                    }
                },
            ).parent
        )
    )

    result = skills.execute({"skill": "long-output", "action": "run", "args": {}})
    assert result.exit_code == 0

    assert result.stdout.startswith("a" * 4000)
    assert result.stdout.endswith("a" * 4000)
    assert "...[truncated " in result.stdout

    assert result.stderr.startswith("b" * 4000)
    assert result.stderr.endswith("b" * 4000)
    assert "...[truncated " in result.stderr


def test_execute_returns_timeout_result(tmp_path: Path) -> None:
    skills = Skills(
        str(
            _write_skill(
                tmp_path,
                "slow",
                "# Slow\n",
                {
                    "actions": {
                        "run": {
                            "args_schema": {"type": "object", "additionalProperties": False},
                            "invoke": ["python3", "-c", "import time; time.sleep(1.0)"],
                            "timeout_seconds": 0.05,
                        }
                    }
                },
            ).parent
        )
    )

    result = skills.execute({"skill": "slow", "action": "run", "args": {}})
    assert result.exit_code == 124
    assert "timed out" in result.stderr


def test_execute_applies_schema_defaults_before_validation_and_invoke(tmp_path: Path) -> None:
    skills = Skills(
        str(
            _write_skill(
                tmp_path,
                "defaults",
                "# Defaults\n",
                {
                    "actions": {
                        "run": {
                            "args_schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "prefix": {"type": "string", "default": "hello"},
                                },
                                "required": ["name"],
                                "additionalProperties": False,
                            },
                            "invoke": [
                                "python3",
                                "-c",
                                "import sys; print(sys.argv[1] + ' ' + sys.argv[2])",
                                "{prefix}",
                                "{name}",
                            ],
                        }
                    }
                },
            ).parent
        )
    )

    result = skills.execute({"skill": "defaults", "action": "run", "args": {"name": "world"}})
    assert result.exit_code == 0
    assert result.stdout == "hello world\n"
