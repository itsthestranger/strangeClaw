#!/usr/bin/env python3
"""Manual LLMClient runner for validating provider wiring before main.py integration."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "reason": {"type": "string"},
    },
    "required": ["tool", "args"],
}
MISSING_ENV_RE = re.compile(
    r"Missing environment variable '([^']+)' required by config field '([^']+)'\."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a direct LLMClient completion using current config + optional overrides."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config YAML (defaults to load_config() behavior).",
    )
    parser.add_argument("--prompt", required=True, help="User prompt text.")
    parser.add_argument("--system", default="", help="Optional system message.")
    parser.add_argument("--model", help="Override llm.model from config.")
    parser.add_argument("--api-key", help="Override llm.api_key from config.")
    parser.add_argument("--api-base", help="Set llm.provider_settings.api_base.")
    parser.add_argument(
        "--structured-output",
        choices=["native", "prompt"],
        help="Override llm.structured_output.",
    )
    parser.add_argument("--max-tokens", type=int, help="Override llm.max_tokens.")
    parser.add_argument("--temperature", type=float, help="Override llm.temperature.")
    parser.add_argument("--timeout-seconds", type=float, help="Override llm.timeout_seconds.")
    parser.add_argument("--max-retries", type=int, help="Override llm.max_retries.")
    parser.add_argument(
        "--provider-setting",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional llm.provider_settings entries. VALUE can be JSON or plain string.",
    )
    parser.add_argument(
        "--action-schema",
        type=Path,
        help=(
            "Path to action schema JSON file. "
            "If omitted with --test-action, uses built-in schema."
        ),
    )
    parser.add_argument(
        "--test-action",
        action="store_true",
        help="Include action_schema to test structured output behavior.",
    )
    return parser.parse_args()


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from agent.llm import LLMClient
    from config import ConfigError, load_config

    args = parse_args()
    try:
        config = _load_config_with_api_key_override(
            load_config=load_config,
            config_error_type=ConfigError,
            config_path=args.config,
            api_key_override=args.api_key,
        )
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    llm_cfg_raw = config.get("llm")
    if not isinstance(llm_cfg_raw, dict):
        print("Config error: missing llm section.", file=sys.stderr)
        return 2

    llm_cfg = dict(llm_cfg_raw)
    provider_settings = llm_cfg.get("provider_settings")
    if provider_settings is None:
        parsed_provider_settings: dict[str, Any] = {}
    elif isinstance(provider_settings, dict):
        parsed_provider_settings = dict(provider_settings)
    else:
        print("Config error: llm.provider_settings must be an object.", file=sys.stderr)
        return 2

    apply_overrides(args, llm_cfg, parsed_provider_settings)
    llm_cfg["provider_settings"] = parsed_provider_settings

    try:
        client = LLMClient.from_config(llm_cfg)
    except Exception as exc:
        print(f"LLM client setup error: {exc}", file=sys.stderr)
        return 2

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    action_schema = load_action_schema(args) if args.test_action else None

    try:
        response = client.complete(messages=messages, action_schema=action_schema)
    except Exception as exc:
        print(f"Completion error: {exc}", file=sys.stderr)
        return 1

    print("=== LLM Run ===")
    print(f"model: {client.model}")
    print(f"structured_output: {client.structured_output}")
    print(f"provider_settings: {json.dumps(client.provider_settings, ensure_ascii=True)}")
    print("\n=== Response Text ===")
    print(response.text)
    print("\n=== Tool Call ===")
    if response.action is None:
        print("none")
    else:
        print(
            json.dumps(
                {
                    "tool": response.action.tool,
                    "args": response.action.args,
                    "reason": response.action.reason,
                },
                indent=2,
                ensure_ascii=True,
            )
        )
    print("\n=== Usage ===")
    print(json.dumps(response.usage, indent=2, ensure_ascii=True))

    try:
        token_count = client.count_tokens(messages)
    except Exception as exc:
        print(f"\nToken count error: {exc}", file=sys.stderr)
        return 1
    print(f"\nEstimated tokens for input messages: {token_count}")
    return 0


def apply_overrides(
    args: argparse.Namespace,
    llm_cfg: dict[str, Any],
    provider_settings: dict[str, Any],
) -> None:
    if args.model:
        llm_cfg["model"] = args.model
    if args.api_key is not None:
        llm_cfg["api_key"] = args.api_key
    if args.structured_output:
        llm_cfg["structured_output"] = args.structured_output
    if args.max_tokens is not None:
        llm_cfg["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        llm_cfg["temperature"] = args.temperature
    if args.timeout_seconds is not None:
        llm_cfg["timeout_seconds"] = args.timeout_seconds
    if args.max_retries is not None:
        llm_cfg["max_retries"] = args.max_retries
    if args.api_base is not None:
        provider_settings["api_base"] = args.api_base
    for item in args.provider_setting:
        key, value = split_key_value(item)
        provider_settings[key] = parse_json_or_string(value)


def load_action_schema(args: argparse.Namespace) -> dict[str, Any]:
    if args.action_schema is None:
        return DEFAULT_ACTION_SCHEMA
    with args.action_schema.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Action schema must be a JSON object.")
    return loaded


def split_key_value(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise ValueError(f"Invalid --provider-setting '{item}'. Expected KEY=VALUE.")
    key, value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Provider setting key cannot be empty.")
    return key, value


def parse_json_or_string(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_config_with_api_key_override(
    load_config: Any,
    config_error_type: type[Exception],
    config_path: Path | None,
    api_key_override: str | None,
) -> dict[str, Any]:
    attempts = 0
    while True:
        attempts += 1
        try:
            loaded = load_config(config_path)
        except config_error_type as exc:
            missing_env = _parse_missing_env(str(exc))
            if (
                missing_env is None
                or api_key_override is None
                or missing_env[1] != "llm.api_key"
                or attempts > 4
            ):
                raise
            os.environ[missing_env[0]] = api_key_override
            continue
        if not isinstance(loaded, dict):
            raise ValueError("Loaded config must be an object.")
        return loaded


def _parse_missing_env(error_message: str) -> tuple[str, str] | None:
    match = MISSING_ENV_RE.search(error_message)
    if not match:
        return None
    env_name, field_path = match.groups()
    return env_name, field_path


if __name__ == "__main__":
    raise SystemExit(main())
