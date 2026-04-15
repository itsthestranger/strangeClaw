#!/usr/bin/env python3
"""Host-level smoke test for B2.4 FireSandbox lifecycle."""

from __future__ import annotations

import argparse
import uuid

from config import load_config
from sandbox.fire import FireSandbox, load_firecracker_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FireSandbox B2.4 lifecycle.")
    parser.add_argument(
        "--goal",
        default="Say hello briefly.",
        help="Task goal text to send to the agent.",
    )
    parser.add_argument(
        "--approval-mode",
        default="auto",
        choices=["auto", "review"],
        help="Task approval mode.",
    )
    parser.add_argument(
        "--session-prefix",
        default="smoke-b24",
        help="Session id prefix.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=60,
        help="Maximum receive attempts before giving up.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Per-receive timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_config()
    fire_cfg = load_firecracker_config(config)
    llm_cfg = dict(config["llm"])

    session_id = f"{args.session_prefix}-{uuid.uuid4().hex[:8]}"
    task = {
        "type": "task",
        "text": args.goal,
        "session_id": session_id,
        "approval_mode": args.approval_mode,
    }

    sandbox = FireSandbox(
        firecracker_config=fire_cfg,
        llm_config=llm_cfg,
    )

    print(f"START session_id={session_id}")
    try:
        sandbox.run(task)
        print("RUN_OK")
        for idx in range(1, args.max_events + 1):
            try:
                event = sandbox.receive(timeout_seconds=args.timeout)
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                print(f"EVENT {idx}: error={exc}")
                process = getattr(sandbox, "_process", None)
                if process is not None:
                    print(f"PROCESS_EXIT={process.poll()}")
                log_tail_fn = getattr(sandbox, "_read_log_tail_best_effort", None)
                if callable(log_tail_fn):
                    log_tail = str(log_tail_fn())
                    if log_tail:
                        print("LOG_TAIL_BEGIN")
                        print(log_tail)
                        print("LOG_TAIL_END")
                return 3
            if event is None:
                print(f"EVENT {idx}: timeout")
                continue

            event_type = str(event.get("type"))
            role = str(event.get("role", ""))
            success = event.get("success")
            print(f"EVENT {idx}: type={event_type} role={role} success={success}")

            if event_type == "done":
                print("DONE_RECEIVED")
                return 0

        print("ERROR: did not receive done event within max-events limit")
        return 2
    finally:
        sandbox.stop()
        print("STOPPED")


if __name__ == "__main__":
    raise SystemExit(main())
