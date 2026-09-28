#!/usr/bin/env python3
"""Writes the per-task retrieval caps: the largest search calls, rows per call and documents read
that any same-pool baseline spent. Usage: python build_caps.py --runs <dir> --tasks <tasks> --out
budget_caps.json [--arm-dir ARM=<dir>]."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

BASELINES = ["autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr", "drtulu"]
TAP_ARMS = ("autosurvey", "surveyforge")
SEARCH_ROUTES = {"s2_search", "s2_snippet_search", "bing_search", "plain_search", "search"}
TAP_SEARCH = {"get_ids_from_query", "get_ids_from_queries", "batch_search", "retrieve_id",
              "retrieve_id4citation"}
OPENS_FLOOR = 60
SERVICE_K_MAX = 1000
SCHEMA = "budget_caps/1.0"
QUANTITIES = ("search_calls", "results_per_call_max", "docs_read")


def jl(p: Path):
    if not p.exists():
        return None
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    out.append({"_unparseable": True})
    return out


def pid_of(x) -> str:
    s = str(x)
    return s.split(".", 1)[1] if "." in s else s


def read_tasks(path: Path) -> list:
    txt = path.read_text()
    if path.suffix == ".json":
        d = json.loads(txt)
        return sorted(d["tasks"] if isinstance(d, dict) else d)
    lines = (ln.strip() for ln in txt.splitlines())
    return sorted(t for t in lines if t and not t.startswith("#"))


def measure(arm: str, d: Path):
    calls = []
    if arm in TAP_ARMS:
        src = d / "retrieval_tap.jsonl"
        rows = jl(src)
        if rows is None:
            return None
        for r in rows:
            if r.get("_unparseable") or int(r.get("depth") or 0):
                continue
            calls.append((r.get("method") in TAP_SEARCH, [pid_of(x) for x in (r.get("ids") or [])]))
    else:
        src = d / "search_calls.jsonl"
        rows = jl(src)
        if rows is None:
            return None
        for r in rows:
            if r.get("_unparseable"):
                continue
            ids = [str(x.get("pmid")) for x in (r.get("returned") or []) if x.get("pmid") is not None]
            calls.append((r.get("route") in SEARCH_ROUTES, ids))
    rpc = [len(ids) for is_search, ids in calls if is_search]
    docs = set()
    for _, ids in calls:
        docs.update(ids)
    return {"source": str(src), "search_calls": len(rpc), "results_per_call_max": max(rpc) if rpc else 0,
            "docs_read": len(docs)}


def files_md5(paths) -> str:
    h = hashlib.md5()
    for p in sorted(paths):
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()


def effective_cap(c: dict) -> dict:
    C_calls, C_rpc, C_docs = (int(c[q]["value"]) for q in QUANTITIES)
    K = min(C_rpc, SERVICE_K_MAX)
    return {"max_search_calls": C_calls, "max_ranked_output_K": K,
            "max_document_opens": max(OPENS_FLOOR, K),
            "max_results_per_call": K, "max_docs_read": C_docs,
            "service_clamped_rpc": C_rpc > SERVICE_K_MAX}


def build(arm_dirs: dict, tasks: list) -> dict:
    per, srcs, missing = {}, [], []
    for arm, root in arm_dirs.items():
        per[arm] = {}
        for t in tasks:
            m = measure(arm, Path(root) / t)
            if m is None:
                missing.append(f"{arm}/{t}")
                continue
            srcs.append(m["source"])
            per[arm][t] = m
    if missing:
        raise SystemExit("REFUSED: no retrieval log for " + ", ".join(missing[:20]))
    caps = {}
    for t in tasks:
        caps[t] = {}
        for q in QUANTITIES:
            v, who = max((per[a][t][q], a) for a in arm_dirs)
            caps[t][q] = {"value": int(v), "arm": who}
    return {"schema": SCHEMA, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "raw_sources_md5": files_md5(srcs), "n_raw_sources": len(srcs),
            "service_k_max": SERVICE_K_MAX, "caps": caps,
            "effective": {"cap": {t: effective_cap(caps[t]) for t in tasks}}}


def arm_dirs_from(runs: Path, baselines: list, overrides: list) -> dict:
    dirs = {a: runs / a for a in baselines}
    for o in overrides:
        arm, _, path = o.partition("=")
        if arm not in dirs or not path:
            raise SystemExit(f"--arm-dir {o!r}: expected ARM=DIR with ARM in {baselines}")
        dirs[arm] = Path(path)
    return dirs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="same-pool campaign root holding <arm>/<task>/")
    ap.add_argument("--arm-dir", action="append", default=[], help="ARM=DIR for an arm stored elsewhere")
    ap.add_argument("--tasks", required=True, help="task list: JSON with 'tasks' or one id per line")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--out", required=True, help="output budget_caps.json")
    a = ap.parse_args(argv)
    baselines = [x for x in a.baselines.split(",") if x]
    doc = build(arm_dirs_from(Path(a.runs), baselines, a.arm_dir), read_tasks(Path(a.tasks)))
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"REFUSED: {out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(doc, indent=1, sort_keys=True).encode()
    out.write_bytes(raw)
    print(f"{out} sha256 {hashlib.sha256(raw).hexdigest()} tasks {len(doc['caps'])} "
          f"sources {doc['n_raw_sources']} raw_sources_md5 {doc['raw_sources_md5']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
