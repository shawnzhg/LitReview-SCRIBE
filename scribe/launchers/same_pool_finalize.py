#!/usr/bin/env python3
"""Checks every shard and unit of a same-pool generation run (window statuses and the hash chain)
and writes level_record.json for the scoring gate. Usage: python same_pool_finalize.py <run dir>."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "harness", "tools")


def lj(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


WINDOWS = ("acquisition", "synthesis", "planning", "writing")
EXITS = {"acquisition": "evidence_bundle.json", "synthesis": "synthesis_graph.json",
         "planning": "outline_plan.json", "writing": "report_artifact.json"}


def content_hash(obj):
    d = {k: v for k, v in obj.items() if k != "content_hash"}
    return "sha256:" + hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def unit_problems(u):
    name = os.path.basename(os.path.dirname(u))
    try:
        rows = [json.loads(l) for l in open(f"{u}/manifests.jsonl") if l.strip()]
    except Exception as e:
        return [f"{name}: manifests unreadable {e}"]
    ws = [m.get("window") for m in rows]
    if ws != list(WINDOWS):
        return [f"{name}: windows {ws}"]
    bad = []
    for m in rows:
        w = m["window"]
        if m.get("status") not in (("ok", "budget_exhausted") if w == "acquisition" else ("ok",)):
            bad.append(f"{name}: {w} status {m.get('status')!r}")
        art = lj(f"{u}/{EXITS[w]}")
        if art is None:
            bad.append(f"{name}: {EXITS[w]} missing")
        elif art.get("content_hash") != content_hash(art):
            bad.append(f"{name}: {EXITS[w]} does not re-hash")
        elif m.get("exit_hash") != art.get("content_hash"):
            bad.append(f"{name}: {w} exit_hash != {EXITS[w]} content_hash")
    for prev, nxt in zip(rows, rows[1:]):
        if nxt.get("entry_hash") != prev.get("exit_hash"):
            bad.append(f"{name}: {nxt['window']} entry_hash != {prev['window']} exit_hash")
    return bad


def main():
    run = sys.argv[1]
    cfg = json.load(open(f"{run}/config.json"))
    L = int(cfg["level"])
    runs = cfg.get("runs_root") or os.environ.get("SCRIBE_RUNS_ROOT")
    if not runs:
        sys.exit("config.json has no runs_root and SCRIBE_RUNS_ROOT is not set")
    sys.path.insert(0, TOOLS)
    import pool_retrieval_tool as PRT
    lvl = f"{runs}/level{L}"
    problems = []
    shards = {}
    for i in sorted(cfg["shards"], key=int):
        r = lj(f"{run}/shard_{i}/shard_record.json")
        shards[i] = r
        if not r:
            problems.append(f"shard {i}: no shard_record.json")
        elif not r.get("ok"):
            problems.append(f"shard {i}: not ok (rc {r.get('rc')}, verify_post {r.get('verify_post_rc')}, reports {r.get('n_reports')}/{len(r.get('tasks') or [])})")
    v = subprocess.run([sys.executable, os.path.join(HERE, "same_pool_prepare.py"), "verify", "--provenance", f"{run}/provenance.json",
                        "--phase", "post", "--out", f"{run}/verify_final.json"], capture_output=True, text=True)
    print(v.stdout.strip())
    if v.returncode != 0:
        problems.append("verify --phase post (all units) refused")
    tasks = cfg["tasks"]
    units = sorted(glob.glob(f"{lvl}/{cfg.get('system', 'SCRIBE')}/native_chain/*/seed{int(cfg['seed'])}"))
    def cnt(n):
        return sum(os.path.exists(f"{u}/{n}") for u in units)
    win_bad = [p for u in units for p in unit_problems(u)]
    problems += win_bad[:10]
    prov_ok = sum(bool((lj(f"{u}/pool_provenance.json") or {}).get("ok")) for u in units)
    marks = sorted(glob.glob(f"{lvl}/_level_infra_failure_*.json") + glob.glob(f"{lvl}/_level_*failed*.json"))
    if marks:
        problems.append(f"failure markers: {marks[:3]}")
    okrec = lj(f"{lvl}/_level_acq_audit_ok.json")
    acq_ok = bool(okrec and okrec.get("ok") is True and int(okrec.get("level", -1)) == L)
    if not acq_ok:
        problems.append("no passing _level_acq_audit_ok.json at this level")
    rc = 0 if not problems and len(units) == len(tasks) and cnt("report_artifact.json") == len(tasks) else 92
    rec = {"schema": "level_record/1.0", "state": "final", "rc": rc, "level": L, "tag": cfg["tag"], "arm": cfg["arm"],
           "theta": {s: cfg["carriers"][s] for s in ("synthesis", "planning", "writing")}, "seeds": str(cfg["seed"]),
           "job": cfg.get("prepare_job"), "mode": "native_chain",
           "retrieval_backend": "pool (frozen static pool); retrieval not re-run: byte-identical copies of the source level's units",
           "corpus_snapshot_id": PRT.POOL_SNAPSHOT_ID, "board": "native_chain",
           "retrieval_budget": "cap",
           "n_tasks_expected": len(tasks), "acq_audit_ok": acq_ok, "failure_markers": marks,
           "n_units": len(units), "n_bundles": cnt("evidence_bundle.json"), "n_reports": cnt("report_artifact.json"),
           "n_pool_provenance_ok": prov_ok, "infra_markers": sorted(glob.glob(f"{lvl}/_level_infra_failure_*.json")),
           "problems": problems,
           "same_pool_generation": {"src": cfg["src"], "src_desc": cfg.get("src_desc"), "pins": cfg["pins"], "pins_sha256": cfg["pins_sha256"],
                   "levers": cfg.get("levers"), "levers_md5": cfg.get("levers_md5"),
                   "carriers": cfg["carriers"], "servers": cfg["servers"],
                   "shards": {i: ({k: r.get(k) for k in ("job", "rc", "verify_post_rc", "n_reports", "ok")} if r else None) for i, r in shards.items()},
                   "prepare_record": lj(f"{run}/prepare_record.json"),
                   "driver_prompt_hash": sorted({(r or {}).get("driver", {}).get("runner_prompt_hash") for r in shards.values() if r and r.get("driver")} - {None}),
                   "payload_sha256": sorted({s.get("payload_sha256") for r in shards.values() if r for s in r.get("servers") or []} - {None})},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(rec, open(f"{run}/level_record.json", "w"), indent=1)
    print(f"### level record rc={rc}: units {len(units)} bundles {rec['n_bundles']} reports {rec['n_reports']} / {len(tasks)}; "
          f"acq_audit_ok={acq_ok}; problems {len(problems)}" + (f": {problems[:4]}" if problems else ""))
    sys.exit(0 if rc == 0 else 1)


if __name__ == "__main__":
    main()
