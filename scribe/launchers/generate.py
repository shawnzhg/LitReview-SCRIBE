#!/usr/bin/env python3
"""Generation driver of SCRIBE: installs the lever writer and the planning endpoint and runs the
fixed-input or same-pool campaign. Usage: SCRIBE_LEVERS=<json> SCRIBE_DRIVER_RECORD=<json> python generate.py
--phase generation --mode bundle_entry|native_chain --level <n> --seeds <s> --tasks-file <file>."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

SCRIBE = Path(__file__).resolve().parents[1]
RUNNERS = str(SCRIBE / "harness" / "runners")
LEVERS = str(SCRIBE / "levers")
RETRIEVAL = str(SCRIBE / "retrieval")


def arg(av, name):
    return av[av.index(name) + 1] if name in av and av.index(name) + 1 < len(av) else None


def route_planning(W, R, rec):
    rp = os.environ.get("SCRIBE_RENDEZVOUS_PLANNING")
    if not rp:
        return
    from llm import VLLMClient
    pc = VLLMClient(rendezvous=json.loads(rp))
    if not pc.health():
        sys.exit(f"planning endpoint {pc.url} is not healthy")
    orig_planning = W.planning

    def planning(spec, graph, client, log, *a, **k):
        out = orig_planning(spec, graph, pc, log, *a, **k)
        if not pc.health():
            raise RuntimeError(f"planning endpoint {pc.url} unhealthy after the planning window")
        return out
    planning._planning_routed = pc.url
    W.planning = planning
    orig_manifest = R.manifest

    def manifest(*a, **k):
        window = a[3] if len(a) > 3 else k.get("window")
        if window == "planning":
            k["serving"] = pc.describe()
        return orig_manifest(*a, **k)
    R.manifest = manifest
    rec["planning_endpoint"] = {"url": pc.url, "rendezvous": json.loads(rp)}
    print(f"### planning routed to {pc.url}", flush=True)


def main():
    rec_path = os.environ.get("SCRIBE_DRIVER_RECORD")
    if not rec_path:
        sys.exit("SCRIBE_DRIVER_RECORD is required")
    lf = os.environ.get("SCRIBE_LEVERS") or ""
    if not lf:
        sys.exit("generation driver: SCRIBE_LEVERS (writing_levers.json) is required: SCRIBE always runs the lever writer")
    av = sys.argv[1:]
    mode, lvl = arg(av, "--mode"), arg(av, "--level")
    if arg(av, "--phase") != "generation" or mode not in ("bundle_entry", "native_chain") or not lvl:
        sys.exit("generation driver: --phase generation --mode bundle_entry|native_chain --level <n> are required")
    sys.path.insert(0, RUNNERS)
    for p in (LEVERS, RETRIEVAL):
        if p not in sys.path:
            sys.path.append(p)
    import windows as W
    import runner as R
    import writing_levers as LV
    rec = {"schema": "generation_driver/1", "argv": av, "mode": mode, "windows_file": W.__file__,
           "planning_endpoint": None}
    route_planning(W, R, rec)
    raw = open(lf, "rb").read()
    rec["levers"] = {"file": lf, "md5": hashlib.md5(raw).hexdigest(), "module": LV.__file__,
                     **LV.install(W, R, LV.load(lf))}
    rec["runner_prompt_hash"] = R.PROMPT_HASH
    print(f"### levers {lf} md5 {rec['levers']['md5']}: {rec['levers']['installed']}", flush=True)
    rec["writing_endpoint"] = json.loads(os.environ["SCRIBE_RENDEZVOUS_WRITING"]) if os.environ.get("SCRIBE_RENDEZVOUS_WRITING") else None
    rec["base_endpoint"] = json.loads(os.environ["SCRIBE_RENDEZVOUS"]) if os.environ.get("SCRIBE_RENDEZVOUS") else None
    json.dump(rec, open(rec_path, "w"), indent=1, sort_keys=True)
    import run_campaign as RC
    if os.path.dirname(os.path.realpath(RC.__file__)) != os.path.realpath(RUNNERS):
        sys.exit(f"generation driver: run_campaign was not loaded from {RUNNERS} ({RC.__file__})")
    if mode == "bundle_entry":
        has_t, has_tf = "--tasks" in av, "--tasks-file" in av
        if has_t == has_tf:
            sys.exit("fixed input: exactly one of --tasks <comma list of dev ids> / --tasks-file <file> is required")
        rec["n_tasks"] = len(arg(av, "--tasks").split(",")) if has_t else len(RC.tasks_from_file(arg(av, "--tasks-file")))
        json.dump(rec, open(rec_path, "w"), indent=1, sort_keys=True)
        print(f"### generation driver: bundle_entry, {rec['n_tasks']} tasks, level {lvl}", flush=True)
        sys.argv[0] = RC.__file__
        RC.main()
        return
    tf, sy = arg(av, "--tasks-file"), arg(av, "--systems") or ""
    if not tf or "--tasks" in av:
        sys.exit("same pool: --tasks-file is required and --tasks is refused")
    if not sy or "," in sy:
        sys.exit(f"same pool: exactly one --systems <name> is required (got {sy!r})")
    seeds = [int(x) for x in (arg(av, "--seeds") or "0").split(",")]
    want = [t for t in open(tf).read().split() if t]
    sel = RC.tasks_from_file(tf)
    units = [(t, s) for s in seeds for t in want]
    miss = [u for u in units if not (R.run_dir(int(lvl), sy, "native_chain", *u) / "evidence_bundle.json").exists()]
    done = [u for u in units if (R.run_dir(int(lvl), sy, "native_chain", *u) / "report_artifact.json").exists()]
    if sel != want or miss or done or not want:
        sys.exit(f"same pool: task selection {len(sel)} != {len(want)}, {len(miss)} units without a bundle, "
                 f"{len(done)} already reported")
    rec["n_tasks"] = len(want)
    rec["seeds"] = seeds
    json.dump(rec, open(rec_path, "w"), indent=1, sort_keys=True)
    print(f"### generation driver: native_chain, {len(units)} units selected, all acquired, none generated", flush=True)
    import run_budgeted_campaign as RCK
    RCK.main(sys.argv[1:])


if __name__ == "__main__":
    main()
