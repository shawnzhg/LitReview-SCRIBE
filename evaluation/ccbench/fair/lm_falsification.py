"""Tests of the module-sensitivity constant on the published pipelines: leave-one-pipeline-out
transfer and the count of locally expansive runs. Usage: python -m ccbench.fair.lm_falsification
[--window writing]."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ccbench import paths
from ccbench.config import prereg
from ccbench.fair.radius import fit_LM, paired_xy

EXCLUDE = tuple(prereg()["fair_first"]["lm_exclude"])


def loso(xy: pd.DataFrame, design: str, q_env: float) -> pd.DataFrame:
    rows = []
    for s in sorted(xy.system.unique()):
        tr, te = xy[xy.system != s], xy[xy.system == s]
        if len(tr) < 5 or te.empty:
            continue
        L, xi = fit_LM(tr.x.values, tr.y.values, design, q_env)
        if not np.isfinite(L):
            continue
        L_own, _ = fit_LM(te.x.values, te.y.values, design, q_env)
        rows.append({"held_out": s, "n_test": len(te), "L_M_train": L, "xi_train": xi, "L_M_own": L_own,
                     "coverage_heldout": float((te.y.values <= L * te.x.values + xi + 1e-9).mean())})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="writing")
    a = ap.parse_args(argv)
    q_env = float(prereg()["stats"]["envelope_quantile"])
    design = prereg()["fair_first"]["design"]["L_M"]
    dist = pd.read_parquet(paths.OUT / "E13" / "distances.parquet")
    xy = paired_xy(dist, a.window)
    xy = xy[~xy.system.isin(set(EXCLUDE))]
    if xy.empty:
        raise SystemExit("no paired (self-retrieving, reference-fed) exits at this window")
    out = paths.out_dir("E13")
    t1 = loso(xy, design, q_env)
    t1.to_csv(out / "lm_loso.csv", index=False)
    ratio = (xy.y / xy.x.where(xy.x > 1e-6)).dropna()
    print(f"window = {a.window}   n = {len(xy)} paired runs, {xy.system.nunique()} pipelines")
    print("(a) leave-one-pipeline-out:")
    print(t1.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"(b) locally expansive runs (y > x): {int((ratio > 1).sum())}/{len(ratio)}   max ratio y/x: {ratio.max():.3f}")


if __name__ == "__main__":
    main()
