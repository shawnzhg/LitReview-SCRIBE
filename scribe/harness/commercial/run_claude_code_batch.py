#!/usr/bin/env python3
"""Runs Claude Code once on every task under both entry conditions. Usage: python
run_claude_code_batch.py [workers]."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_claude_code as R

OUT = R.OUT / "runs"
lock = threading.Lock()


def log(msg: str) -> None:
    with lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def ok(d: Path) -> bool:
    r = json.loads((d / "result.json").read_text())
    return (r.get("subtype") == "success" and r.get("model_ok") is True and r.get("words", 0) > 0
            and r.get("n_tool_errors", 1) == 0)


def one(task: str, cond: str) -> None:
    d = OUT / cond / task
    if d.exists():
        log(f"skip {cond} {task} (already run)")
        return
    log(f"start {cond} {task}")
    try:
        R.run(task, cond)
    except Exception as e:
        log(f"EXC {cond} {task}: {e!r}")
    log(f"{'OK' if (d / 'result.json').exists() and ok(d) else 'FAILED'} {cond} {task}")


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    tasks = json.loads(Path(os.environ["COMMERCIAL_TASKS"]).read_text())
    jobs = [(t, c) for t in tasks for c in ("fixinput", "samepool")]
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: one(*j), jobs))
    log("all done")
