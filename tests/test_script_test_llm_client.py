"""Regression checks for scripts/test-llm-client.py action output shape."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_test_llm_client_default_action_schema_is_tool_based() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "test-llm-client.py"
    namespace = runpy.run_path(str(script_path))

    schema = namespace["DEFAULT_ACTION_SCHEMA"]
    assert isinstance(schema, dict)
    assert schema.get("required") == ["tool", "args"]
    properties = schema.get("properties")
    assert isinstance(properties, dict)
    assert "tool" in properties
    assert "args" in properties
    assert "skill" not in properties
    assert "action" not in properties


def test_test_llm_client_script_uses_toolcall_fields() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "test-llm-client.py"
    source = script_path.read_text(encoding="utf-8")

    assert "response.action.tool" in source
    assert "response.action.args" in source
    assert "response.action.reason" in source
    legacy_skill_attr = "response.action." + "skill"
    legacy_action_attr = "response.action." + "action"
    assert legacy_skill_attr not in source
    assert legacy_action_attr not in source
