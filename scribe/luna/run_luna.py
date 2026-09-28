#!/usr/bin/env python3
"""Runs one SCRIBE driver with the Luna client in place of the vLLM client and records every harness
module it loaded. Usage: python run_luna.py fixed|acquisition|generation <driver arguments>."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import luna_client as LC

DRIVERS = {"fixed": LC.REPO / "scribe" / "launchers" / "generate.py",
           "acquisition": LC.REPO / "scribe" / "retrieval" / "run_acquisition.py",
           "generation": LC.REPO / "scribe" / "launchers" / "generate.py"}
LEVERS_DIR = LC.REPO / "scribe" / "levers"
SINGLE_BACKBONE = ("SCRIBE_RENDEZVOUS_WRITING", "SCRIBE_RENDEZVOUS_PLANNING")


def module_record(modules=None):
    modules = sys.modules if modules is None else modules
    out = {}
    for name in LC.HARNESS_MODULES:
        f = getattr(modules.get(name), "__file__", None)
        if f:
            out[name] = {"file": str(Path(f).resolve()), "md5": hashlib.md5(Path(f).read_bytes()).hexdigest()}
    return out


def check_env(env):
    raw = env.get("SCRIBE_RENDEZVOUS")
    if not raw:
        return "SCRIBE_RENDEZVOUS is not set (eval_luna.sh writes it)"
    try:
        why = LC.rendezvous_problem(json.loads(raw))
    except json.JSONDecodeError:
        return "SCRIBE_RENDEZVOUS is not JSON"
    if why:
        return why
    extra = [v for v in SINGLE_BACKBONE if env.get(v)]
    if extra:
        return f"{extra} set: one backbone serves every window"
    return None


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in DRIVERS:
        sys.exit(f"usage: run_luna.py {'|'.join(DRIVERS)} <driver arguments>")
    why = check_env(os.environ)
    if why:
        sys.exit(f"luna: {why}")
    LC.install(LC.load_llm())
    driver = DRIVERS[argv[0]]
    sys.argv = [str(driver)] + argv[1:]
    sys.path.insert(0, str(driver.parent))
    if str(LEVERS_DIR) not in sys.path:
        sys.path.append(str(LEVERS_DIR))
    code = 0
    try:
        runpy.run_path(str(driver), run_name="__main__")
    except SystemExit as e:
        code = e.code
    rec = os.environ.get("LUNA_MODULES_RECORD")
    if rec:
        Path(rec).write_text(json.dumps({"driver": str(driver), "argv": argv, "exit": code,
                                         "modules": module_record()}, indent=1, sort_keys=True))
    bad = LC.foreign_modules()
    if bad:
        sys.exit(f"luna: harness modules loaded from outside {LC.REPO}: {bad}")
    sys.exit(code)


if __name__ == "__main__":
    main()
