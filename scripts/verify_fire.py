#!/usr/bin/env python3
"""Unified host-level Fire mode verification checks."""

from __future__ import annotations

import argparse
import uuid
from typing import Literal

from config import load_config
from sandbox.fire import FireSandbox, IptablesManager, TapDeviceManager, load_firecracker_config

CheckKind = Literal["network", "lifecycle", "all"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fire mode host-level verification checks.")
    parser.add_argument(
        "--check",
        choices=["network", "lifecycle", "all"],
        default="all",
        help="Which verification check to run.",
    )
    parser.add_argument(
        "--goal",
        default="Say hello briefly.",
        help="Task goal text for lifecycle check.",
    )
    parser.add_argument(
        "--approval-mode",
        default="auto",
        choices=["auto", "review"],
        help="Task approval mode for lifecycle check.",
    )
    parser.add_argument(
        "--session-prefix",
        default="verify-fire",
        help="Session id prefix for generated verification sessions.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=60,
        help="Maximum receive attempts before lifecycle check times out.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Per-receive timeout in seconds for lifecycle check.",
    )
    return parser.parse_args()


def _verify_network(*, session_prefix: str) -> int:
    tap = TapDeviceManager(host_iface=None)
    fw = IptablesManager()
    session_id = f"{session_prefix}-net-{uuid.uuid4().hex[:8]}"

    print(f"NETWORK_START session_id={session_id}")
    allocation = tap.create(session_id=session_id)
    print("NETWORK_ALLOC:", allocation)

    try:
        fw.apply(allocation)
        print("NETWORK_RULES_APPLIED")
        return 0
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"NETWORK_ERROR: {exc}")
        return 1
    finally:
        try:
            fw.cleanup(allocation)
            print("NETWORK_RULES_CLEANED")
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"NETWORK_CLEANUP_WARNING: {exc}")
        try:
            tap.destroy(allocation.tap_name)
            print("NETWORK_TAP_CLEANED")
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"NETWORK_TAP_CLEANUP_WARNING: {exc}")


def _verify_lifecycle(args: argparse.Namespace) -> int:
    config = load_config()
    fire_cfg = load_firecracker_config(config)
    llm_cfg = dict(config["llm"])
    session_id = f"{args.session_prefix}-life-{uuid.uuid4().hex[:8]}"
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

    print(f"LIFECYCLE_START session_id={session_id}")
    try:
        sandbox.run(task)
        print("LIFECYCLE_RUN_OK")
        for idx in range(1, args.max_events + 1):
            try:
                event = sandbox.receive(timeout_seconds=args.timeout)
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                print(f"LIFECYCLE_EVENT {idx}: error={exc}")
                process = getattr(sandbox, "_process", None)
                if process is not None:
                    print(f"LIFECYCLE_PROCESS_EXIT={process.poll()}")
                log_tail_fn = getattr(sandbox, "_read_log_tail_best_effort", None)
                if callable(log_tail_fn):
                    log_tail = str(log_tail_fn())
                    if log_tail:
                        print("LIFECYCLE_LOG_TAIL_BEGIN")
                        print(log_tail)
                        print("LIFECYCLE_LOG_TAIL_END")
                return 3
            if event is None:
                print(f"LIFECYCLE_EVENT {idx}: timeout")
                continue

            event_type = str(event.get("type"))
            role = str(event.get("role", ""))
            success = event.get("success")
            print(f"LIFECYCLE_EVENT {idx}: type={event_type} role={role} success={success}")

            if event_type == "done":
                print("LIFECYCLE_DONE_RECEIVED")
                return 0

        print("LIFECYCLE_ERROR: did not receive done event within max-events limit")
        return 2
    finally:
        sandbox.stop()
        print("LIFECYCLE_STOPPED")


def _run(args: argparse.Namespace) -> int:
    selected: CheckKind = args.check
    if selected == "network":
        return _verify_network(session_prefix=args.session_prefix)
    if selected == "lifecycle":
        return _verify_lifecycle(args)

    network_result = _verify_network(session_prefix=args.session_prefix)
    if network_result != 0:
        return network_result
    return _verify_lifecycle(args)


def main() -> int:
    args = _parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
