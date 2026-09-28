#!/usr/bin/env python3
"""Runs run_campaign under the pool backend the driver installed (run_acquisition.py or
run_budgeted_campaign.py) and stops with a failure marker on a pool or vLLM transport failure."""

from __future__ import annotations

import http.client
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INFRA_EXIT = 86
_ABORT = threading.Event()
_ABORT_WHY: list = []
_UNIT = threading.local()
_NOT_INFRA = re.compile(r"^HTTPError: HTTP Error 4\d\d")


def _note_infra(msg):
    lst = getattr(_UNIT, "infra", None)
    if lst is not None:
        lst.append(str(msg)[:300])


def is_infra_error(err) -> bool:
    return bool(err) and not _NOT_INFRA.match(str(err))


def watch_chat_transport():
    import llm as LLM
    if getattr(LLM.VLLMClient.chat, "_infra_watched", False):
        return
    orig = LLM.VLLMClient.chat

    def chat(self, *a, **k):
        try:
            d = orig(self, *a, **k)
        except http.client.HTTPException as e:
            _note_infra(f"{type(e).__name__}: {e}")
            raise
        if is_infra_error(getattr(d, "error", None)):
            _note_infra(d.error)
        return d
    chat._infra_watched = True
    LLM.VLLMClient.chat = chat


def _phase_of(argv):
    for i, a in enumerate(argv):
        if a == "--phase" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--phase="):
            return a.split("=", 1)[1]
    return None


def _arg(argv, name, default=None):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def write_infra_marker(level, why, task=None, phase=None):
    from common import RUNS, utcnow
    d = RUNS / f"level{level}"
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    p = d / f"_level_infra_failure_{os.environ.get('SLURM_JOB_ID', 'nojob')}_{stamp}_{os.getpid()}.json"
    rec = {"schema": "infra_failure/1.0", "level": level, "phase": phase, "task": task, "why": why,
           "utc": utcnow(), "job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
           "rule": "this level must not be resumed, exposed or certified; re-run at a FRESH level"}
    if not p.exists():
        p.write_text(json.dumps(rec, indent=1))
    print(f"### INFRA FAILURE ({phase}, {task}): {why} -> {p}", flush=True)
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    phase = _phase_of(argv)
    mode = _arg(argv, "--mode", "native_chain")
    if phase not in ("acquisition", "generation"):
        sys.exit("--phase acquisition|generation is required")
    if mode != "native_chain":
        sys.exit(f"run_campaign_pool.py serves --mode native_chain only (got {mode!r})")
    if "--redo" in argv:
        sys.exit("--redo is refused on a pool-backed level: a partial or finished unit is never regenerated in "
                 "place (fresh per-task caches). Use a fresh level.")
    level = _arg(argv, "--level")
    if level is None:
        sys.exit("--level is required (a fresh private level)")

    import pool_backend as PB
    if phase == "acquisition":
        PRT = PB._INSTALLED.get("acq")
        if PRT is None:
            sys.exit("### pool: the acquisition driver installs the pool backend first (run_acquisition.py)")
        guard_errs = (PRT.PoolServiceDead, PRT.PoolContractViolation)
    else:
        if not PB._INSTALLED.get("gen"):
            sys.exit("### pool: the generation driver installs the pool backend first (run_budgeted_campaign.py)")
        guard_errs = ()
    import run_campaign as RC
    import runner as R
    watch_chat_transport()

    orig_one = RC.one

    def guarded_one(ph, task, system, md, seed, lvl, client, entry_mode, client_write=None):
        if _ABORT.is_set():
            raise SystemExit(f"campaign aborted earlier: {_ABORT_WHY[0]}")
        d = R.run_dir(lvl, system, md, task, seed)
        PB.set_unit_dir(d)
        _UNIT.infra = []
        try:
            row = orig_one(ph, task, system, md, seed, lvl, client, entry_mode, client_write)
        except guard_errs as e:
            why = f"{type(e).__name__}: {e}"
            if not _ABORT.is_set():
                _ABORT_WHY.append(why)
                _ABORT.set()
                write_infra_marker(lvl, why, task=task, phase=ph)
            raise SystemExit(why)
        finally:
            PB.set_unit_dir(None)
            infra, _UNIT.infra = list(getattr(_UNIT, "infra", None) or []), None
        if infra:
            why = f"{len(infra)} LLM transport error(s) during {task} (first: {infra[0]})"
            if not _ABORT.is_set():
                _ABORT_WHY.append(why)
                _ABORT.set()
                write_infra_marker(lvl, why, task=task, phase=ph)
            raise SystemExit(why)
        bad = [c.url for c in (client, client_write) if c is not None and not c.health()]
        if bad:
            why = f"vLLM endpoint(s) unhealthy after {task}: {bad}"
            if not _ABORT.is_set():
                _ABORT_WHY.append(why)
                _ABORT.set()
                write_infra_marker(lvl, why, task=task, phase=ph)
            raise SystemExit(why)
        if ph == "generation":
            leaked = PB.assert_generation_clean()
            if leaked:
                why = f"retrieval modules present in the generation process: {leaked}"
                _ABORT_WHY.append(why)
                _ABORT.set()
                write_infra_marker(lvl, why, task=task, phase=ph)
                raise SystemExit(why)
        row["pool_backend"] = "pool" if ph == "acquisition" else "pool_texts"
        return row

    RC.one = guarded_one
    sys.argv = [sys.argv[0]] + argv
    try:
        RC.main()
    except SystemExit as e:
        if _ABORT.is_set():
            print(f"### pool: campaign stopped by the infra guard: {_ABORT_WHY[0]}", flush=True)
            sys.exit(INFRA_EXIT)
        raise
    if _ABORT.is_set():
        sys.exit(INFRA_EXIT)
