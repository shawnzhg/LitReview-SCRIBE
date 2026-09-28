#!/usr/bin/env python3
"""Precomputes, per training task, the train-split peer band and the human top-section titles that
the planning-exit reward normalises against, without evaluation ids. Usage: python
precompute_train_bands.py --tasks <file> --embed_url <url> --out <json>."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "evaluation"))
import planning_exit_readouts as PE

DEF_EVAL_TASKS = os.environ.get("SCRIBE_EVAL_TASKS") or "/nonexistent/SCRIBE_EVAL_TASKS"
DEF_EVAL_EXCLUDED = str(ROOT / "configs" / "train_tasks" / "eval_excluded.txt")


def sha16(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def band_peers(task: str, peers_mod, excluded: set) -> list:
    return [p for p in peers_mod.peers(task, split="train") if p not in excluded]


def org_size_fit_of(review: str, target: str, n_hum: int, human):
    ro = human.adapt(review, target)
    levels = [sec.get("level", 1) for sec in ro.report.sections if sec.get("title")] if ro.report else []
    ns = sum(1 for l in levels if l == min(levels)) if levels else 0
    return PE.org_size_fit(ns, n_hum)


def band_rows(task: str, peers: list, cc: dict, *, units, human) -> tuple:
    cco = cc["cco"]
    ut = units.build(task)
    hum_top, n_hum = PE.human_top_titles(ut.top_sections, cco)
    notes = {"peers_without_top_sections": [], "peer_errors": []}
    rows = []

    def add(src, r, v):
        if v is not None:
            rows.append({"task": task, "readout": r, "value": float(v), "source_review": src})

    st, _ = PE.sys_top_titles(PE.nodes_from_top_sections(ut.top_sections), cco)
    add(task, "outline_title_f1_lex", PE.title_f1_lex(st, hum_top, cc))
    add(task, "outline_title_f1_emb", PE.title_f1_emb(st, hum_top, cc))
    add(task, "org_size_fit", org_size_fit_of(task, task, n_hum, human))
    for p in peers:
        try:
            up = units.build(p)
        except Exception as e:
            notes["peer_errors"].append(f"{p}: {type(e).__name__}")
            up = None
        if up is not None:
            nodes = PE.nodes_from_top_sections(up.top_sections)
            if nodes:
                sp, _ = PE.sys_top_titles(nodes, cco)
                add(p, "outline_title_f1_lex", PE.title_f1_lex(sp, hum_top, cc))
                add(p, "outline_title_f1_emb", PE.title_f1_emb(sp, hum_top, cc))
            else:
                notes["peers_without_top_sections"].append(p)
        add(p, "org_size_fit", org_size_fit_of(p, task, n_hum, human))
    return rows, hum_top, n_hum, notes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks", required=True, help="the training task list")
    ap.add_argument("--embed_url", required=True, help="embed_service.py serving nomic-embed-text-v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval_tasks", default=DEF_EVAL_TASKS)
    a = ap.parse_args(argv)
    t0 = time.time()
    sys.dont_write_bytecode = True
    import pandas as pd
    from ccbench.gt import units, peers as P
    from ccbench.adapters import human
    from ccbench.rankability import normalise as NRM

    tasks = [l.strip() for l in open(a.tasks) if l.strip()]
    evals = set(json.load(open(a.eval_tasks))["tasks"])
    excluded = {l.strip() for l in open(DEF_EVAL_EXCLUDED) if l.strip()}
    if not evals or not excluded:
        raise SystemExit("evaluation task / exclusion list empty: refusing")
    bad = sorted(set(tasks) & (evals | excluded))
    if bad:
        raise SystemExit(f"{len(bad)} tasks are evaluation tasks or excluded ids: {bad[:5]}")
    emb = PE.NomicHTTPEmbed(a.embed_url)
    model = str(emb.health().get("model") or "")
    if "nomic" not in model.lower():
        raise SystemExit(f"{a.embed_url} serves {model!r}, not nomic-embed-text-v1")
    cc = PE.make_cc(emb)
    out_tasks = {}
    for t in tasks:
        tt = time.time()
        peers = band_peers(t, P, evals | excluded)
        rows, hum_top, n_hum, notes = band_rows(t, peers, cc, units=units, human=human)
        cal = NRM.calibration(pd.DataFrame(rows)) if rows else pd.DataFrame()
        bands = {}
        for r in PE.READOUTS:
            c = cal[cal.readout == r] if len(cal) else cal
            if not len(c):
                continue
            c = c.iloc[0]
            f = lambda x: None if pd.isna(x) else float(x)
            bands[r] = {"own": f(c.own), "q_lo": f(c.q_lo), "q_hi": f(c.q_hi), "median": f(c["median"]),
                        "n_peers": int(c.n_peers), "peer_sd": f(c.peer_sd), "direction": PE.DIRECTION[r]}
        hum_vec = emb.encode(hum_top) if hum_top else None
        complete = all(r in bands and bands[r]["n_peers"] > 0 for r in PE.READOUTS) and hum_vec is not None
        out_tasks[t] = {"hum_top": hum_top, "n_hum_top": n_hum, "peers_train": peers,
                        "n_peers_train": len(peers), "bands": bands,
                        "peer_values": {r: {x["source_review"]: round(x["value"], 6) for x in rows
                                            if x["readout"] == r} for r in PE.READOUTS},
                        "hum_top_emb_b64": (PE.b64_f32(hum_vec) if hum_vec is not None else None),
                        "emb_dim": (int(hum_vec.shape[1]) if hum_vec is not None and hum_vec.size else None),
                        "notes": notes, "complete": bool(complete)}
        print(f"  {t:22s} n_hum_top {n_hum:2d} train peers {len(peers)} band n "
              f"{ {r: (bands.get(r) or {}).get('n_peers') for r in PE.READOUTS} } complete {complete} "
              f"({time.time() - tt:.1f}s)", flush=True)
    body = {"schema": "scribe_train_bands/1",
            "rule": ("hum_top = the task's own top-level section titles; peers = its same-topic "
                     "train-split reviews minus the evaluation tasks and the excluded ids; each "
                     "readout's band is the peers' values against hum_top, own = the task's own "
                     "value; calibration = ccbench rankability.normalise.calibration"),
            "prereg": {"title_match_fuzzy": cc["TITLE_FUZZ"], "section_match_cosine": cc["SEC_COS"]},
            "encoder": f"service:{model}", "doc_prefix": PE.DOC_PREFIX, "peers_split": "train",
            "tasks_file": {"path": str(a.tasks), "sha256_16": sha16(a.tasks)},
            "inputs": {"eval_tasks": sha16(a.eval_tasks), "eval_excluded": sha16(DEF_EVAL_EXCLUDED)},
            "n_tasks": len(out_tasks), "n_complete": sum(1 for v in out_tasks.values() if v["complete"]),
            "tasks": out_tasks}
    body["sha256_16"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1, sort_keys=True))
    print(f"[precompute_train_bands] {len(out_tasks)} tasks ({body['n_complete']} complete) -> {out} "
          f"sha256_16 {body['sha256_16']} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
