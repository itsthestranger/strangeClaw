"""Top-level strangeclaw entrypoint module."""

from __future__ import annotations

import argparse
from pathlib import Path

from adapters.cli import CLIAdapter
from config import load_config
from sandbox.yolo import YoloSandbox
from session import load as load_session


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="strangeclaw")
    parser.add_argument("--resume", type=str, default=None, help="Resume from session id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the strangeclaw application."""
    args = _parse_args(argv)
    config = load_config()
    if config["mode"] != "yolo":
        raise ValueError(f"Unsupported mode: {config['mode']}")
    if config["adapter"] != "cli":
        raise ValueError(f"Unsupported adapter: {config['adapter']}")

    sandbox = YoloSandbox(
        skills_dir=str(config["skills"]["directory"]),
        max_iterations=int(config["loop"]["max_iterations"]),
        token_budget=int(config["context"]["token_budget"]),
        summary_threshold=int(config["context"]["summary_threshold"]),
    )

    resume_state = None
    if args.resume:
        session_dir = Path.home() / ".strangeclaw" / "sessions" / args.resume
        resume_state = load_session(session_dir)
        if resume_state is None:
            raise ValueError(f"Cannot resume: session state not found for {args.resume}")

    adapter = CLIAdapter(
        sandbox=sandbox,
        approval_mode=str(config["approval_mode"]),
        llm_config=dict(config["llm"]),
        resume_session_id=args.resume,
        resume_state=resume_state,
    )
    try:
        adapter.run()
    except KeyboardInterrupt:
        sandbox.stop()


if __name__ == "__main__":
    main()
