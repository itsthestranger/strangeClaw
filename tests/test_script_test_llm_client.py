"""Regression checks for scripts/test-llm-client.py action output shape."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest


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


def test_test_llm_client_runtime_prints_tool_based_action(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "test-llm-client.py"
    namespace = runpy.run_path(str(script_path))

    import agent.llm as llm_module
    import config as config_module

    class FakeLLMClient:
        def __init__(self) -> None:
            self.model = "fake/model"
            self.structured_output = "prompt"
            self.provider_settings: dict[str, Any] = {"api_base": "http://mock.local/v1"}

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> FakeLLMClient:
            assert config["model"] == "fake/model"
            return cls()

        def complete(
            self,
            *,
            messages: list[dict[str, Any]],
            action_schema: dict[str, Any] | None = None,
        ) -> llm_module.LLMResponse:
            assert messages
            assert action_schema is not None
            return llm_module.LLMResponse(
                text="mock-response",
                action=llm_module.ToolCall(
                    tool="shell",
                    args={"command": "pwd"},
                    reason="Need cwd",
                ),
                usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            )

        def count_tokens(self, messages: list[dict[str, Any]]) -> int:
            assert messages
            return 7

    def fake_load_config(_: Path | None = None) -> dict[str, Any]:
        return {
            "llm": {
                "model": "fake/model",
                "api_key": "sk-test",
                "provider_settings": {},
            }
        }

    monkeypatch.setattr(llm_module, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(config_module, "load_config", fake_load_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script_path), "--prompt", "use tool", "--test-action"],
    )

    exit_code = namespace["main"]()
    assert exit_code == 0
    stdout = capsys.readouterr().out
    tool_section = stdout.split("=== Tool Call ===", maxsplit=1)[1].split(
        "=== Usage ===",
        maxsplit=1,
    )[0]
    payload = json.loads(tool_section.strip())
    assert payload == {
        "tool": "shell",
        "args": {"command": "pwd"},
        "reason": "Need cwd",
    }
    assert "skill" not in payload
    assert "action" not in payload
