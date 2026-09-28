"""Certified leaderboard from per-task axis scores: equal-weight composite, paired cluster-robust t
test with Bonferroni correction over the table, and rank intervals per tier and observation
point. Usage: python -m ccbench.merge_windowbench --root <inputs root>."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import pandas as pd

from ccbench import paths
from ccbench.fair.composite import rank_intervals, tierable
from ccbench.fair.decide import cluster_t_radius as cluster_t

WB = Path(os.environ.get("WINDOWBENCH_ROOT", "/nonexistent/WINDOWBENCH_ROOT"))

TIERS = {
    "reference_fed": ["autosurvey.ref", "llmxmr.ref", "surveyg.ref", "sgi.ref", "lira"],
    "self_retrieving": ["autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr"],
}
DELTA = 0.05


def load(root: Path = WB) -> tuple[pd.DataFrame, dict]:
    z = pd.read_csv(root / "inputs" / "per_task_axis_z.csv")
    clusters = dict(pd.read_csv(root / "inputs" / "task_clusters.csv").values)
    return z, clusters


def task_composite(s: pd.DataFrame, axes: list[str]) -> pd.DataFrame:
    P = s[s.axis.isin(axes)].pivot_table(index=["system", "task"], columns="axis", values="z", aggfunc="mean").reindex(columns=axes)
    return P.mean(axis=1).where(P.notna().all(axis=1)).unstack("task")


def composite(z: pd.DataFrame, obs: str, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    s = z[(z.obs == obs) & z.system.isin(names)]
    if s.empty:
        return pd.DataFrame(), []
    per = {sys_: set(g.axis.unique()) for sys_, g in s.groupby("system")}
    axes = sorted(set.intersection(*per.values())) if per else []
    odd = {k: sorted(v - set(axes)) for k, v in per.items() if set(v) != set(axes)}
    if odd:
        raise ValueError(f"{obs}: axis sets differ across the tier, composite not comparable: {odd}")
    U = task_composite(s, axes)
    return U.reindex([n for n in names if n in U.index]), axes


def leaderboard(U: pd.DataFrame, clusters: dict):
    names = list(U.index)
    npair = len(names) * (len(names) - 1) // 2
    rel, rows = set(), []
    for a, b in itertools.combinations(names, 2):
        d = (U.loc[a] - U.loc[b]).dropna()
        if len(d) < 10:
            continue
        q, se, K = cluster_t(d.values, list(d.index), clusters, DELTA / npair)
        gap = float(d.mean())
        ok = bool(abs(gap) > q)
        if ok:
            rel.add((a, b) if gap > 0 else (b, a))
        rows.append(dict(a=a, b=b, gap=gap, q=q, se=se, n=len(d), K=K, certified=ok))
    iv, acyclic, naive_ok = rank_intervals(names, rel)
    full = U.mean(axis=1).sort_values(ascending=False)
    board = pd.DataFrame([
        dict(display_rank=i, system=s, composite=float(v), cert_lo=iv[s][0], cert_hi=iv[s][1])
        for i, (s, v) in enumerate(full.items(), 1)])
    meta = dict(certified=len(rel), pairs=npair, C_dec=len(rel) / npair if npair else float("nan"),
                acyclic=acyclic, naive_rank_formula_ok=naive_ok, tierable=tierable(names, rel),
                relations=sorted(f"{a}>{b}" for a, b in rel))
    return board, pd.DataFrame(rows), meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(WB))
    ap.add_argument("--obs", default="system_trunc,system")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if not (root / "inputs" / "per_task_axis_z.csv").exists():
        raise SystemExit(f"windowbench inputs not found under {root}")
    z, clusters = load(root)
    out = paths.out_dir("E19")
    boards, pairs, metas = [], [], []
    for obs in a.obs.split(","):
        for tier, names in TIERS.items():
            U, axes = composite(z, obs, names)
            if U.empty or len(U) < 3:
                continue
            b, p, m = leaderboard(U, clusters)
            b.insert(0, "tier", tier); b.insert(0, "obs", obs)
            p.insert(0, "tier", tier); p.insert(0, "obs", obs)
            m |= dict(obs=obs, tier=tier, n_axes=len(axes), axes=",".join(axes))
            boards.append(b); pairs.append(p); metas.append(m)
            tag = "PRIMARY" if obs == "system_trunc" else "secondary"
            print(f"\n{'='*74}\n{tier} / {obs}  ({tag})   {len(axes)} axes: {', '.join(axes)}")
            print(f"  certified {m['certified']}/{m['pairs']}  C_dec={m['C_dec']:.2f}   "
                  f"acyclic={m['acyclic']}  naive-rank-ok={m['naive_rank_formula_ok']}  "
                  f"tierable={m['tierable']}")
            print(f"  {'Display':>7}  {'Agent':20}{'Composite C':>12}{'Certified rank':>16}")
            for r in b.itertuples():
                cr = str(r.cert_lo) if r.cert_lo == r.cert_hi else f"{r.cert_lo}-{r.cert_hi}"
                print(f"  {r.display_rank:>7}  {r.system.replace('.ref','•'):20}"
                      f"{r.composite:12.3f}{cr:>16}")
            if m["relations"]:
                print("  " + ", ".join(x.replace(".ref", "•") for x in m["relations"]))
    pd.concat(boards).to_csv(out / "merged_leaderboard.csv", index=False)
    pd.concat(pairs).to_csv(out / "merged_pairs.csv", index=False)
    (out / "merged_meta.json").write_text(json.dumps(metas, indent=1))
    print(f"\nwrote {out}/merged_leaderboard.csv, merged_pairs.csv, merged_meta.json")


if __name__ == "__main__":
    main()
