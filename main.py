"""Top-level strangeclaw entrypoint module."""

from __future__ import annotations

from adapters.cli import CLIAdapter
from config import load_config
from sandbox.yolo import YoloSandbox


def main() -> None:
    """Run the strangeclaw application."""
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
    adapter = CLIAdapter(
        sandbox=sandbox,
        approval_mode=str(config["approval_mode"]),
        llm_config=dict(config["llm"]),
    )
    try:
        adapter.run()
    except KeyboardInterrupt:
        sandbox.stop()


if __name__ == "__main__":
    main()
