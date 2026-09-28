#!/usr/bin/env python3
"""Writes the per-task bundle cap K_cap, the largest final bundle any same-pool baseline produced on
the task, with the budget derived from it and the caps file. Usage: python build_kcap.py --caps
budget_caps.json --rollouts <rollouts dir> --out kcap.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics as st
import sys
import time
from pathlib import Path

BASELINES = ["autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr", "drtulu"]
SERVICE_K_MAX = 1000
NATIVE_OPENS = 60
SCHEMA = "kcap/1.0"
CAPS_SCHEMA = "budget_caps/1.0"
RULE = ("K_cap(task) = max over the same-pool baselines of |P| on the task (<= 1000); rows per call = "
        "min(CAP.rpc, 1000); search calls = CAP.calls and docs read = CAP.docs; opens = max(60, K_cap)")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def effective(K_cap: int, cap: dict) -> dict:
    return {"max_search_calls": int(cap["search_calls"]["value"]), "max_ranked_output_K": int(K_cap),
            "max_document_opens": max(NATIVE_OPENS, int(K_cap)),
            "max_results_per_call": min(int(cap["results_per_call_max"]["value"]), SERVICE_K_MAX),
            "max_docs_read": int(cap["docs_read"]["value"])}


def bundle_size(rollouts: Path, arm: str, task: str):
    p = rollouts / arm / f"{task}.json"
    if not p.exists():
        return None
    return len(json.loads(p.read_text()).get("papers") or [])


def build(caps: Path, rollouts: Path, arms: list) -> dict:
    C = json.loads(caps.read_text())
    if C.get("schema") != CAPS_SCHEMA:
        raise SystemExit(f"REFUSED: {caps} is not a {CAPS_SCHEMA} file")
    tasks = sorted(C["caps"])
    bad, out = [], {}
    for t in tasks:
        P = {}
        for a in arms:
            v = bundle_size(rollouts, a, t)
            if v is None:
                bad.append(f"{a}/{t}: no rollout under {rollouts}")
                continue
            P[a] = int(v)
        if len(P) != len(arms):
            continue
        vmax = max(P.values())
        tied = sorted(a for a, v in P.items() if v == vmax)
        K_cap = min(vmax, SERVICE_K_MAX)
        if K_cap < 1:
            bad.append(f"{t}: K_cap {K_cap} < 1")
            continue
        cap = C["caps"][t]
        out[t] = {"K_cap": K_cap, "arm": tied[0], "P_max": vmax, "clamped_to_1000": vmax > SERVICE_K_MAX,
                  "P_by_arm": P, "tied_arms": tied, "search_calls_cap": cap["search_calls"],
                  "docs_read_cap": cap["docs_read"], "effective": effective(K_cap, cap)}
    if bad:
        raise SystemExit("REFUSED:\n  " + "\n  ".join(bad[:20]))
    ks = [v["K_cap"] for v in out.values()]
    return {"schema": SCHEMA, "rule": RULE, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "inputs": {"caps": str(caps), "caps_sha256": sha256(caps), "rollouts": str(rollouts),
                       "raw_sources_md5": C.get("raw_sources_md5"), "baselines": arms},
            "service_k_max": SERVICE_K_MAX, "native_opens": NATIVE_OPENS,
            "summary": {"n_tasks": len(ks), "K_cap_median": st.median(ks), "K_cap_min": min(ks), "K_cap_max": max(ks),
                        "n_clamped": sum(1 for v in out.values() if v["clamped_to_1000"]),
                        "argmax_arm_counts": {a: sum(1 for v in out.values() if v["arm"] == a) for a in arms}},
            "tasks": out}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", required=True, help="budget_caps.json written by build_caps.py")
    ap.add_argument("--rollouts", required=True, help="scorer rollouts dir holding <arm>/<task>.json")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--out", required=True, help="output kcap.json")
    a = ap.parse_args(argv)
    doc = build(Path(a.caps), Path(a.rollouts), [x for x in a.baselines.split(",") if x])
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(doc, indent=1, sort_keys=True).encode()
    out.write_bytes(raw)
    print(f"{out} sha256 {hashlib.sha256(raw).hexdigest()} {doc['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
