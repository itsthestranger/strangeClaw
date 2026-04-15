#!/usr/bin/env python3
"""Minimal host-level smoke test for B2.3a/B2.3b."""

from __future__ import annotations

from sandbox.fire import IptablesManager, TapDeviceManager


def main() -> int:
    tap = TapDeviceManager(host_iface=None)
    fw = IptablesManager()

    alloc = tap.create(session_id="verify-b23")
    print("ALLOC:", alloc)

    try:
        fw.apply(alloc)
        print("RULES_APPLIED")
    finally:
        fw.cleanup(alloc)
        tap.destroy(alloc.tap_name)
        print("CLEANED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
