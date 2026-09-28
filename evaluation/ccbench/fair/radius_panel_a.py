"""Fits the module-sensitivity constant on the published pipelines from their paired runs under the
two entries and compares the estimator designs, overall and per pipeline. Usage: python -m
ccbench.fair.radius_panel_a [--window writing]."""

from __future__ import annotations

import argparse

import pandas as pd

from ccbench import paths
from ccbench.config import prereg
from ccbench.fair.radius import fit_LM, paired_xy

DESIGNS = ("ratio_q95", "ratio_sup", "affine")


def designs_table(xy: pd.DataFrame, q_env: float) -> pd.DataFrame:
    rows = []
    for d in DESIGNS:
        L, xi = fit_LM(xy.x.values, xy.y.values, d, q_env)
        rows.append({"design": d, "L_M": L, "xi": xi, "n": len(xy)})
    for s, g in xy.groupby("system"):
        L, xi = fit_LM(g.x.values, g.y.values, DESIGNS[0], q_env)
        rows.append({"design": f"{DESIGNS[0]}:{s}", "L_M": L, "xi": xi, "n": len(g)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="writing")
    a = ap.parse_args(argv)
    ff = prereg()["fair_first"]
    q_env = float(prereg()["stats"]["envelope_quantile"])
    dist = pd.read_parquet(paths.OUT / "E13" / "distances.parquet")
    xy = paired_xy(dist, a.window)
    xy = xy[~xy.system.isin(set(ff["lm_exclude"]))]
    if xy.empty:
        raise SystemExit(f"no paired (self-retrieving, reference-fed) exits at the {a.window} window")
    t = designs_table(xy, q_env)
    t.to_csv(paths.out_dir("E13") / "lm_designs.csv", index=False)
    print(f"window = {a.window}   n = {len(xy)} paired runs, {xy.system.nunique()} pipelines, x in [{xy.x.min():.2f}, {xy.x.max():.2f}]")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
