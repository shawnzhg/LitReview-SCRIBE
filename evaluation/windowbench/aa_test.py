"""Calibrates the leaderboard's decision rule under a sign-flip null on a board's per-task composite.
Usage: python -m windowbench.aa_test --root <board dir> [--obs system_trunc,system]."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import merged_board as MB


def null_cells(merge, U: pd.DataFrame, clusters: dict) -> tuple[list[dict], int]:
    names = list(U.index)
    npair = len(names) * (len(names) - 1) // 2
    cells = []
    for a, b in itertools.combinations(names, 2):
        d = (U.loc[a] - U.loc[b]).dropna()
        if len(d) < C.REGISTERED["min_tasks"]:
            continue
        tasks = list(d.index)
        q, _se, _K = merge.cluster_t(d.values, tasks, clusters, merge.DELTA / npair)
        cells.append({"a": a, "b": b, "tasks": tasks, "gap": float(d.mean()), "q": q,
                      "d0": d.values - d.values.mean()})
    return cells, npair


def calibrate(merge, U: pd.DataFrame, clusters: dict, reps: int, seed: int = 0) -> dict:
    cells, npair = null_cells(merge, U, clusters)
    keys = sorted({clusters.get(t, t) for c in cells for t in c["tasks"]})
    idx = {k: i for i, k in enumerate(keys)}
    for c in cells:
        c["ci"] = np.array([idx[clusters.get(t, t)] for t in c["tasks"]])
    rng = np.random.default_rng(seed)
    stats = np.zeros(reps)
    for j in range(reps):
        sign = rng.choice([-1.0, 1.0], size=len(keys))
        m = 0.0
        for c in cells:
            v = c["d0"] * sign[c["ci"]]
            q, _se, _K = merge.cluster_t(v, c["tasks"], clusters, merge.DELTA / npair)
            if np.isfinite(q) and q > 0:
                m = max(m, abs(float(v.mean())) / q)
        stats[j] = m
    nominal = merge.DELTA
    c_star = float(np.quantile(stats, 1 - nominal))
    certified = [c for c in cells if np.isfinite(c["q"]) and abs(c["gap"]) > c["q"]]
    removed = [f"{c['a']}>{c['b']}" if c["gap"] > 0 else f"{c['b']}>{c['a']}"
               for c in certified if abs(c["gap"]) <= c_star * c["q"]]
    return {"pairs": npair, "pairs_scored": len(cells), "certified": len(certified), "reps": reps,
            "familywise_false_certification": float((stats > 1.0).mean()), "nominal": nominal,
            "c_star": c_star, "relations_removed_at_c_star": removed}


def tables(root: Path, obs_list: list[str]) -> dict[tuple[str, str], list[str]]:
    lb = root / "merged_leaderboard.csv"
    if lb.exists():
        b = pd.read_csv(lb)
        return {(o, t): list(g.system) for (o, t), g in b.groupby(["obs", "tier"], sort=False) if o in obs_list}
    merge = MB.import_merge_windowbench()
    return {(o, t): list(v) for o in obs_list for t, v in merge.TIERS.items()}


def run(root: Path, obs_list: list[str], reps: int, seed: int = 0) -> list[dict]:
    C.bootstrap_env()
    merge = MB.import_merge_windowbench()
    from ccbench.merge_diagnostics import loco
    z, clusters = merge.load(root)
    out = []
    for (obs, tier), names in tables(root, obs_list).items():
        U, axes = merge.composite(z, obs, [n for n in names if n in set(z[z.obs == obs].system)])
        if U.empty or len(U) < 3:
            continue
        rec = {"obs": obs, "tier": tier, "systems": list(U.index), "axes": axes}
        rec.update(calibrate(merge, U, clusters, reps, seed))
        rec["leave_one_subfield_out"] = loco(U, clusters)
        out.append(rec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="board directory holding inputs/ (and merged_leaderboard.csv)")
    ap.add_argument("--obs", default="system_trunc,system")
    ap.add_argument("--reps", type=int, default=C.REGISTERED["aa_flips"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    res = run(root, [o for o in a.obs.split(",") if o], a.reps, a.seed)
    for r in res:
        lo = r["leave_one_subfield_out"]
        print(f"{r['tier']:16s} {r['obs']:13s} pairs {r['pairs']:3d} certified {r['certified']:3d} "
              f"family-wise {r['familywise_false_certification']:.3f} (nominal {r['nominal']}) "
              f"c* {r['c_star']:.3f} (drops {len(r['relations_removed_at_c_star'])}) "
              f"Jaccard {lo['jaccard_mean']:.3f}/{lo['jaccard_min']:.3f} flips {lo['winner_flips']}")
    p = Path(a.out) if a.out else root / "calibration.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"root": str(root), "result": res, "registry": C.registry()}, indent=1, default=str))
    print("wrote", p)


if __name__ == "__main__":
    main()
