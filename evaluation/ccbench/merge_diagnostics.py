"""Diagnostics of the certified tiers: leave-one-subfield-out stability and the per-pair
false-certification rate under a cluster sign-flip null. Usage: python -m ccbench.merge_diagnostics
--root <inputs root>."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccbench import paths
from ccbench.merge_windowbench import DELTA, TIERS, WB, cluster_t, composite, load


def loco(U: pd.DataFrame, clusters: dict) -> dict:
    def certify(cols):
        names = list(U.index)
        npair = len(names) * (len(names) - 1) // 2
        out = set()
        for a, b in itertools.combinations(names, 2):
            d = (U.loc[a, cols] - U.loc[b, cols]).dropna()
            if len(d) < 10:
                continue
            q, _, K = cluster_t(d.values, list(d.index), clusters, DELTA / npair)
            if abs(d.mean()) > q:
                out.add((a, b) if d.mean() > 0 else (b, a))
        return out
    base = certify(list(U.columns))
    keys = sorted({clusters.get(t, t) for t in U.columns})
    js, flips = [], 0
    for k in keys:
        cols = [t for t in U.columns if clusters.get(t, t) != k]
        s = certify(cols)
        inter = len(base & s); union = len(base | s)
        js.append(inter / union if union else 1.0)
        flips += sum(1 for a, b in s if (b, a) in base)
    return dict(folds=len(keys), jaccard_mean=float(np.mean(js)), jaccard_min=float(np.min(js)),
                winner_flips=flips, base_certified=len(base))


def aa(U: pd.DataFrame, clusters: dict, reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    names = list(U.index)
    npair = len(names) * (len(names) - 1) // 2
    keys = sorted({clusters.get(t, t) for t in U.columns})
    num = den = 0
    for _ in range(reps):
        sgn = {k: rng.choice([-1, 1]) for k in keys}
        for a, b in itertools.combinations(names, 2):
            d = (U.loc[a] - U.loc[b]).dropna()
            if len(d) < 10:
                continue
            d0 = d.values - d.values.mean()
            v = d0 * np.array([sgn[clusters.get(t, t)] for t in d.index])
            q, _, K = cluster_t(v, list(d.index), clusters, DELTA / npair)
            num += int(abs(v.mean()) > q); den += 1
    return dict(reps=reps, cells=den, false_certification_rate=num / den if den else float("nan"),
                nominal=DELTA / npair)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(WB))
    ap.add_argument("--obs", default="system_trunc")
    ap.add_argument("--reps", type=int, default=400)
    a = ap.parse_args(argv)
    z, clusters = load(Path(a.root))
    out = paths.out_dir("E21")

    diag = {}
    for tier, names in TIERS.items():
        U, axes = composite(z, a.obs, names)
        if U.empty:
            continue
        d = dict(tier=tier, axes=len(axes), loco=loco(U, clusters), aa=aa(U, clusters, a.reps, 3))
        diag[tier] = d
        print(f"\n{'='*70}\n{tier}")
        print(f"  leave one subfield out: {d['loco']['folds']} folds, Jaccard mean {d['loco']['jaccard_mean']:.3f} "
              f"min {d['loco']['jaccard_min']:.3f}, winner flips {d['loco']['winner_flips']}")
        print(f"  sign-flip null: per-pair false certification {d['aa']['false_certification_rate']:.4f} "
              f"(nominal {d['aa']['nominal']:.4f}) over {d['aa']['cells']} null cells")
    (out / f"diagnostics_{a.obs}.json").write_text(json.dumps(diag, indent=1))
    print(f"\nwrote {out}/diagnostics_{a.obs}.json")


if __name__ == "__main__":
    main()
