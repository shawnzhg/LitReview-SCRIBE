"""Applies the conventional decision rules and then the certification rule, one requirement at a
time, to the pairs of the published pipelines. Usage: python -m ccbench.contrast_certified --root
<inputs root> [--radius <refusals json>] [--obs system_trunc]."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ccbench import paths
from ccbench.config import prereg
from ccbench.fair.units import same_model
from ccbench.merge_windowbench import WB, cluster_t, load, task_composite

PANEL = ["autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr", "lira", "drtulu"]
ENTRY = {s: "frozen pool" for s in PANEL} | {"lira": "review bibliography"}
DELTA = 0.05
MIN_TASKS = 10
BACKBONE_CHANGE = "no finite radius: backbone change outside the scaffold window"
ENTRY_UNMEASURED = "no finite radius: entry mismatch without a measured radius"
ENTRY_EXCEEDS = "eps_out > eps_fair: entry mismatch"


def load_radius(path) -> dict:
    d = json.loads(Path(path).read_text())
    return {frozenset((r["pipeline"], d["reference"])): r for r in d["table"]}


def nuisance(a: str, b: str, radius: dict | None = None) -> tuple[float, str]:
    if not same_model(a, b, prereg()["fair_first"]["same_model_groups"]):
        return math.inf, BACKBONE_CHANGE
    if ENTRY[a] == ENTRY[b]:
        return 0.0, ""
    r = (radius or {}).get(frozenset((a, b)))
    if r is None or not r.get("readouts"):
        return math.inf, ENTRY_UNMEASURED
    if r["refused"]:
        return float(r["max_eps_out"]), ENTRY_EXCEEDS
    return float(r["max_eps_out"]), ""


def admissible(a: str, b: str, radius: dict | None = None) -> tuple[bool, str]:
    why = nuisance(a, b, radius)[1]
    return not why, why


def contrast(obs: str = "system_trunc", root: Path = WB, radius: dict | None = None):
    z, clusters = load(root)
    s = z[(z.obs == obs) & z.system.isin(PANEL)]
    if s.empty:
        raise SystemExit(f"no rows at obs={obs}")
    axes_of = {k: set(g.axis.unique()) for k, g in s.groupby("system")}
    names = [n for n in PANEL if n in axes_of]
    own = pd.concat([task_composite(s[s.system == n], sorted(axes_of[n])) for n in names])
    pairs = list(itertools.combinations(names, 2))
    nz = {p: nuisance(*p, radius) for p in pairs}
    adm = [p for p in pairs if not nz[p][1]]

    rows = []
    for a, b in pairs:
        U = task_composite(s[s.system.isin([a, b])], sorted(axes_of[a] & axes_of[b]))
        d = (U.loc[a] - U.loc[b]).dropna()
        if len(d) < MIN_TASKS:
            continue
        tl = list(d.index)
        gap = float(d.mean())
        t_naive = gap / (d.std(ddof=1) / np.sqrt(len(d))) if d.std(ddof=1) > 0 else 0.0
        p_naive = 2 * (1 - stats.t.cdf(abs(t_naive), len(d) - 1))
        q_un, se, K = cluster_t(d.values, tl, clusters, DELTA)
        q_bon, _, _ = cluster_t(d.values, tl, clusters, DELTA / len(pairs))
        q_adm, _, _ = cluster_t(d.values, tl, clusters, DELTA / max(len(adm), 1))
        eps_out, why = nz[(a, b)]
        ok = not why
        rows.append(dict(a=a, b=b, gap=gap, n=len(d), K=K, admissible=ok, refusal=why,
                         R0=True, R1=bool(p_naive < DELTA), R2=bool(abs(gap) > q_un),
                         R3=bool(abs(gap) > q_bon),
                         R4=bool(ok and abs(gap) > q_adm),
                         R5=bool(ok and abs(gap) > q_adm + eps_out)))
    df = pd.DataFrame(rows)
    score = own.mean(axis=1).sort_values(ascending=False)
    return df, score, pairs, adm


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", default="system_trunc")
    ap.add_argument("--root", default=str(WB))
    ap.add_argument("--radius", default=None)
    a = ap.parse_args(argv)
    radius = load_radius(a.radius) if a.radius else None
    df, score, pairs, adm = contrast(a.obs, Path(a.root), radius)
    out = paths.out_dir("E20")

    print(f"THE CONVENTIONAL LEADERBOARD, obs = {a.obs}\n")
    print(f"  {'rank':>4}  {'system':14}{'score':>8}   comparable to")
    for i, (s, v) in enumerate(score.items(), 1):
        n_ok = sum(1 for x in score.index if x != s and admissible(s, x, radius)[0])
        flag = "  <-- comparable to NOTHING in this panel" if n_ok == 0 else ""
        print(f"  {i:>4}  {s:14}{v:8.3f}   {n_ok} of {len(score)-1} systems{flag}")

    RULES = [("R0", "sorted table (any difference)"), ("R1", "paired t, uncorrected"),
             ("R2", "cluster-robust t, uncorrected"), ("R3", "+ Bonferroni (careful leaderboard)"),
             ("R4", "+ fairness filter (refuse, not score)"), ("R5", "+ nuisance radius = CERTIFIED")]
    print(f"\n  {'rule':>4}  {'':38}{'decided':>9}{'of':>4}{'':>3}note")
    for key, label in RULES:
        n = int(df[key].sum())
        denom = len(pairs) if key in ("R0", "R1", "R2", "R3") else len(adm)
        note = "" if key in ("R0", "R1", "R2", "R3") else f"{len(pairs)-len(adm)} pairs refused first"
        print(f"  {key:>4}  {label:38}{n:>9}{denom:>4}{'':>3}{note}")

    ref = df[~df.admissible]
    print(f"\n  refusals ({len(ref)} of {len(pairs)} pairs), by reason:")
    for why, g in ref.groupby("refusal"):
        print(f"    {why:62} {len(g):>3} pairs   {sorted({x for p in g[['a','b']].values for x in p})}")

    cert = df[df.R5]
    print(f"\n  certified relations ({len(cert)}):")
    for r in cert.itertuples():
        hi, lo = (r.a, r.b) if r.gap > 0 else (r.b, r.a)
        print(f"    {hi:14} > {lo:14}  gap={abs(r.gap):.3f}  n={r.n}  K={r.K}")

    df.to_csv(out / f"contrast_{a.obs}.csv", index=False)
    score.rename("score").to_csv(out / f"conventional_order_{a.obs}.csv")
    summary = {"obs": a.obs, "n_pairs": len(pairs), "n_admissible": len(adm),
               "decided": {k: int(df[k].sum()) for k, _ in RULES},
               "conventional_top": score.index[0],
               "conventional_top_comparable_to": sum(
                   1 for x in score.index if x != score.index[0] and admissible(score.index[0], x, radius)[0])}
    (out / f"contrast_summary_{a.obs}.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}/contrast_{a.obs}.csv, conventional_order_{a.obs}.csv, "
          f"contrast_summary_{a.obs}.json")


if __name__ == "__main__":
    main()
