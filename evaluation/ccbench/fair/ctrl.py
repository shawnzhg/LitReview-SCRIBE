"""Computes the contextwise controllability radius per pair and window from the entry and exit
distances. Usage: python -m ccbench.fair.ctrl [--windows ...]."""

from __future__ import annotations

import argparse

import pandas as pd

from ccbench import paths
from ccbench.config import prereg


def ctrl_table(dist: pd.DataFrame, window: str, lambdas: list[float]) -> pd.DataFrame:
    ext = dist[(dist.kind == "exit") & (dist.window == window)].dropna(subset=["d"])
    ent = dist[(dist.kind == "entry") & (dist.window == window)].dropna(subset=["d"])
    e_of = {(str(r.a), r.task): float(r.d) for r in ent.itertuples()}
    rows = []
    for (a, b), g in ext.groupby([ext.a.astype(str), ext.b.astype(str)]):
        tasks = list(g.task)
        d_ex = list(g.d.astype(float))
        lo, hi = [], []
        for t in tasks:
            ea, eb = e_of.get((a, t)), e_of.get((b, t))
            if ea is None or eb is None:
                lo.append(0.0); hi.append(0.0)
            else:
                lo.append(abs(ea - eb)); hi.append(ea + eb)
        n = len(tasks)
        if n < 5:
            continue
        base = {"window": window, "a": a, "b": b, "n_tasks": n,
                "mean_exit": sum(d_ex) / n, "max_exit": max(d_ex),
                "mean_entry_lo": sum(lo) / n, "mean_entry_hi": sum(hi) / n}
        for lam in lambdas:
            m_lo = sum(lam * l + (1 - lam) * x for l, x in zip(lo, d_ex)) / n
            m_hi = sum(lam * h + (1 - lam) * x for h, x in zip(hi, d_ex)) / n
            base[f"ctrl_lo_l{lam}"] = m_lo
            base[f"ctrl_hi_l{lam}"] = m_hi
            base[f"ctrl_sup_l{lam}"] = max(lam * h + (1 - lam) * x for h, x in zip(hi, d_ex))
        rows.append(base)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="retrieval,writing")
    a = ap.parse_args(argv)
    lam = [float(prereg()["fair_first"]["ctrl_lambda_in"])]
    dist = pd.read_parquet(paths.OUT / "E13" / "distances.parquet")
    out = paths.out_dir("E13")
    allt = []
    for w in a.windows.split(","):
        t = ctrl_table(dist, w, lam)
        if t.empty:
            print(f"  {w:10} no pairs with >=5 common tasks")
            continue
        allt.append(t)
        col = f"ctrl_hi_l{lam[0]}"
        print(f"\n  window = {w}   pairs = {len(t)}   (lambda_in = {lam[0]})")
        s = t.sort_values(col)
        show = ["a", "b", "n_tasks", "mean_entry_lo", "mean_entry_hi", "mean_exit", col]
        print(s[show].head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if allt:
        df = pd.concat(allt, ignore_index=True)
        p = out / "ctrl_radius.csv"
        df.to_csv(p, index=False)
        print(f"\n  wrote {p} ({len(df)} pairs)")


if __name__ == "__main__":
    main()
