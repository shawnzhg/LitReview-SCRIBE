"""Entry and exit interface distances of the published pipelines per task and window, with the
entry measured against the task's canonical reference bundle; used by the nuisance radius and the
controllability radius. Usage: python -m ccbench.fair.units [--tasks ...]."""

from __future__ import annotations

import argparse
import itertools
import time

import pandas as pd

from ccbench import paths
from ccbench.build import load_rollouts
from ccbench.composition import bound as CB
from ccbench.ingest import agent_runs, gold
from ccbench.metrics import interfaces as M
from ccbench.model import Rollout

WINDOW_K = {"retrieval": 1, "writing": 4, "system": 4}
WINDOWS = ["retrieval", "writing", "system"]
REFERENCE_FED = {"lira"}


def key(ro: Rollout) -> str:
    return ro.system if ro.mode in (None, "campaign", "native_chain", "human") else f"{ro.system}.{ro.mode}"


def base_arm(system_key: str) -> str:
    return system_key[:-4] if system_key.endswith(".ref") else system_key


def same_model(a: str, b: str, groups: dict[str, list[str]]) -> bool:
    a, b = base_arm(a), base_arm(b)
    ga = [g for g, m in groups.items() if a in m]
    gb = [g for g, m in groups.items() if b in m]
    return bool(ga) and ga == gb


def provenance_tier(system_key: str) -> str:
    return "reference_fed" if (system_key in REFERENCE_FED or system_key.endswith(".ref")) else "self"


def canonical_entry(task: str) -> dict | None:
    try:
        rel = paths.canonical_bundle_rel(paths.split_of(task))
    except FileNotFoundError:
        rel = paths.canonical_bundle_rel("dev")
    p = paths.resolve_opt(f"{rel}/{task}.json")
    return agent_runs.load_json(p) if p is not None else None


def entry_mismatch(ro: Rollout, window: str, canon: dict | None) -> tuple[float | None, str]:
    if window in ("system", "retrieval"):
        return 0.0, f"common_taskspec|{provenance_tier(key(ro))}"
    if canon is None:
        return None, "no_canonical_bundle"
    ref_p = [str(p.get("paper_id")) for p in canon.get("papers", []) if p.get("paper_id")]
    if not ref_p:
        return None, "no_canonical_papers"
    if ro.is_bot and not ro.papers:
        return None, "bot"
    return 1 - M.jaccard(ro.papers, ref_p), "d1_canonical"


def exit_distance(a: Rollout, b: Rollout, window: str) -> float | None:
    if a.is_bot or b.is_bot:
        return None
    k = WINDOW_K[window]
    if k == 1:
        return M.d1_evidence(a, b) if (a.papers or b.papers) else None
    return CB.stage_exit_distance(k, a, b)


def compute_distances(tasks: list[str] | None = None) -> pd.DataFrame:
    tasks = tasks or gold.campaign50_tasks()
    ro_all: dict[str, dict[str, Rollout]] = {}
    for r in load_rollouts(None, tasks):
        if r.panel == "A":
            ro_all.setdefault(key(r), {})[r.task] = r
    systems = sorted(ro_all)
    rows = []
    t0 = time.time()
    for t in tasks:
        canon = canonical_entry(t)
        for w in WINDOWS:
            for s in systems:
                ro = ro_all[s].get(t)
                if ro is None:
                    continue
                d, note = entry_mismatch(ro, w, canon)
                rows.append({"task": t, "panel": "A", "window": w, "kind": "entry", "a": s, "b": "reference", "d": d, "note": note})
            for sa, sb in itertools.combinations(systems, 2):
                ra, rb = ro_all[sa].get(t), ro_all[sb].get(t)
                if ra is None or rb is None:
                    continue
                d = exit_distance(ra, rb, w)
                rows.append({"task": t, "panel": "A", "window": w, "kind": "exit", "a": sa, "b": sb, "d": d, "note": "bot" if d is None else f"d{WINDOW_K[w]}"})
        print(f"distances {t} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None)
    a = ap.parse_args(argv)
    df = compute_distances(a.tasks.split(",") if a.tasks else None)
    out = paths.out_dir("E13")
    dest = out / "distances.parquet"
    if a.tasks and dest.exists():
        old = pd.read_parquet(dest)
        df = pd.concat([old[~old.task.isin(df.task.unique())], df], ignore_index=True)
    df.to_parquet(dest, index=False)
    print(f"wrote {dest} ({len(df)} rows)")
    e = df[df.kind == "entry"].dropna(subset=["d"])
    print("\nmean entry mismatch to the reference bundle, by system x window:")
    print(e.pivot_table(index="a", columns="window", values="d", aggfunc="mean").round(3).to_string())
    x = df[df.kind == "exit"].dropna(subset=["d"])
    print("\nmean exit distance, by window (over pairs):")
    print(x.groupby("window").d.agg(["mean", "max", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
