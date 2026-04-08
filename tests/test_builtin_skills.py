"""Tests for built-in skill definitions and scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.skills import Skills, SkillsError

BEGIN_DATA = "--- BEGIN DATA ---"
END_DATA = "--- END DATA ---"


def _skills_root() -> Path:
    return Path(__file__).resolve().parents[1] / "skills"


def _unwrap_wrapped_data(payload: str) -> str:
    assert payload.startswith(f"{BEGIN_DATA}\n")
    assert payload.endswith(f"\n{END_DATA}")
    return payload[len(BEGIN_DATA) + 1 : -(len(END_DATA) + 1)]


def test_builtin_skills_discoverable() -> None:
    skills = Skills(str(_skills_root()))
    index = skills.index()
    names = {entry["name"] for entry in index}
    assert {"shell", "web-search", "http-request"} <= names


def test_shell_skill_executes_command() -> None:
    skills = Skills(str(_skills_root()))
    result = skills.execute(
        {"skill": "shell", "action": "run", "args": {"command": "printf shell-ok"}}
    )
    assert result.exit_code == 0
    assert result.stdout == "shell-ok"


def test_web_search_skill_search_and_fetch_wrap_external_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skills = Skills(str(_skills_root()))
    search_payload_path = tmp_path / "search.json"
    search_payload_path.write_text(
        json.dumps(
            {
                "RelatedTopics": [
                    {"Text": "cats result 1", "FirstURL": "https://example.com/1"},
                    {"Text": "cats result 2", "FirstURL": "https://example.com/2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SC_WEB_SEARCH_ENDPOINT", search_payload_path.as_uri())

    search_result = skills.execute(
        {
            "skill": "web-search",
            "action": "search",
            "args": {"query": "cats", "limit": 2},
        }
    )
    assert search_result.exit_code == 0
    search_payload = json.loads(search_result.stdout)
    assert search_payload["count"] == 2
    assert search_payload["data"].startswith(BEGIN_DATA)
    assert search_payload["data"].endswith(END_DATA)
    assert "cats result 1" in search_payload["data"]

    page_path = tmp_path / "page.txt"
    page_path.write_text("remote page text from test fixture", encoding="utf-8")
    fetch_result = skills.execute(
        {
            "skill": "web-search",
            "action": "fetch",
            "args": {"url": page_path.as_uri(), "max_chars": 1024},
        }
    )
    assert fetch_result.exit_code == 0
    fetch_payload = json.loads(fetch_result.stdout)
    assert fetch_payload["status_code"] == 200
    assert fetch_payload["data"].startswith(BEGIN_DATA)
    assert fetch_payload["data"].endswith(END_DATA)
    assert "remote page text from test fixture" in fetch_payload["data"]


def test_http_request_skill_wraps_external_data(tmp_path: Path) -> None:
    skills = Skills(str(_skills_root()))
    response_file = tmp_path / "response.json"
    response_file.write_text('{"message":"hello"}', encoding="utf-8")
    result = skills.execute(
        {
            "skill": "http-request",
            "action": "request",
            "args": {
                "method": "GET",
                "url": response_file.as_uri(),
                "headers": {},
                "body_json": None,
                "timeout_seconds": 5,
                "max_chars": 4096,
            },
        }
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status_code"] == 200
    assert payload["data"].startswith(BEGIN_DATA)
    assert payload["data"].endswith(END_DATA)
    wrapped_json = json.loads(_unwrap_wrapped_data(payload["data"]))
    assert json.loads(wrapped_json["body"]) == {"message": "hello"}


def test_builtin_skill_schemas_reject_invalid_args() -> None:
    skills = Skills(str(_skills_root()))

    with pytest.raises(SkillsError, match="Invalid args"):
        skills.execute({"skill": "shell", "action": "run", "args": {}})

    with pytest.raises(SkillsError, match="Invalid args"):
        skills.execute(
            {
                "skill": "web-search",
                "action": "search",
                "args": {"query": "cats", "limit": "2"},
            }
        )

    with pytest.raises(SkillsError, match="Invalid args"):
        skills.execute(
            {
                "skill": "http-request",
                "action": "request",
                "args": {
                    "method": "TRACE",
                    "url": "https://example.com",
                    "headers": {},
                    "body_json": None,
                    "timeout_seconds": 5,
                    "max_chars": 2000,
                },
            }
        )
