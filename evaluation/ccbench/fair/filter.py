"""Pre-registered tolerance per readout (the median gap between human peer reviews) and the
admissibility test of the nuisance radius against it. Usage: python -m ccbench.fair.filter."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ccbench import paths
from ccbench.config import prereg

TOLERANCE = "human_median_gap"


def tolerances(hscores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    H = hscores.dropna(subset=["value"]).copy()
    H["is_self"] = H.source_review == H.task
    lo, hi = cfg["tolerance_admit"]["gt"], cfg["tolerance_admit"]["lt"]
    rows = []
    for (w, r), g in H.groupby(["window", "readout"]):
        gaps = []
        for t, gt in g.groupby("task"):
            own = gt[gt.is_self].value
            peers = gt[~gt.is_self].value
            if len(own) and len(peers):
                gaps += list((peers - float(own.iloc[0])).abs())
        v = float(np.median(gaps)) if gaps else np.nan
        rows.append({"window": w, "readout": r, "design": TOLERANCE, "eps_fair": v, "n_pairs": len(gaps), "admissible": bool(not np.isnan(v) and lo < v < hi)})
    return pd.DataFrame(rows)


def apply_filter(radii: pd.DataFrame, tol: pd.DataFrame) -> pd.DataFrame:
    t = tol[["window", "readout", "eps_fair", "admissible"]]
    c = radii.merge(t, on=["window", "readout"], how="left")
    regime, reason = [], []
    for r in c.itertuples():
        if not r.same_model:
            regime.append("unfair"), reason.append("cross_model")
            continue
        if r.status == "provenance_mismatch":
            regime.append("unfair"), reason.append("provenance_mismatch")
            continue
        if r.status != "ok" or np.isnan(r.eps_outside):
            regime.append("unfair"), reason.append("no_estimate")
            continue
        if (r.eps_in_a <= 1e-12) and (r.eps_in_b <= 1e-12):
            regime.append("exact"), reason.append("")
            continue
        if np.isnan(r.eps_fair):
            regime.append("unfair"), reason.append("no_tolerance")
            continue
        if not r.admissible:
            regime.append("unfair"), reason.append("humans_incomparable" if r.eps_fair >= 0.5 else "zero_tolerance")
            continue
        if r.eps_outside <= r.eps_fair:
            regime.append("tolerance"), reason.append("")
        else:
            regime.append("unfair"), reason.append("entry_mismatch")
    c["regime"] = regime
    c["reason"] = reason
    c["fair"] = c.regime.isin(["exact", "tolerance"])
    return c


def main(argv=None):
    argparse.ArgumentParser().parse_args(argv)
    cfg = prereg()["fair_first"]
    out = paths.out_dir("E13")
    ws = pd.read_parquet(out / "window_scores.parquet")
    radii = pd.read_parquet(out / "radii.parquet")
    tol = tolerances(ws[ws.panel == "H"], cfg)
    tol.to_csv(out / "tolerances.csv", index=False)
    cells = apply_filter(radii, tol)
    cells.to_parquet(out / "cells.parquet", index=False)
    pd.set_option("display.width", 240)
    print("tolerances, admissible readouts per window:")
    print(tol.groupby("window").admissible.agg(["sum", "size"]).to_string())
    print("\nfair cells per window:")
    print(cells.groupby("window").fair.agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    main()
