#!/usr/bin/env python3
"""Entry point that registers the planning task and runs kvskill's seed_init or grpo, with the
frozen-backbone guard for grpo. Usage: python kv_entry.py seed_init|grpo [args]."""

from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODS = {"grpo": "kvskill.grpo", "seed_init": "kvskill.seed_init"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in MODS:
        sys.exit(f"usage: kv_entry.py {{{'|'.join(MODS)}}} [args]")
    cmd = sys.argv.pop(1)
    import planning_task
    planning_task.register()
    if cmd == "grpo":
        import backbone
        backbone.install_grpo_guard()
    mod = importlib.import_module(MODS[cmd])
    sys.argv[0] = MODS[cmd]
    return mod.main()


if __name__ == "__main__":
    main()
