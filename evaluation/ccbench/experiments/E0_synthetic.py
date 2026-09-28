"""Checks the composition bound on synthetic four-stage chains built from the human graphs. Usage:
python -m ccbench.experiments.E0_synthetic [--tasks 20] [--R 6]."""

from __future__ import annotations

import argparse
import json

import numpy as np
import ot
import pandas as pd

from ccbench import paths
from ccbench.composition import bound as CB
from ccbench.config import prereg
from ccbench.ingest import gold
from ccbench.metrics import interfaces as M
from ccbench.synthetic.harness import Params, run_chain


def exit_distance(k: int, a, b) -> float:
    if k == 1:
        return 1 - M.jaccard(a.P, b.P)
    if k == 2:
        return M.d2_fgw(a.rollout.graph, b.rollout.graph) or 0.0
    if k == 3:
        d3 = M.d3_outline(a.rollout, b.rollout) or 0.0
        d2 = M.d2_fgw(a.rollout.graph, b.rollout.graph) or 0.0
        return 0.5 * (d2 + d3)
    return M.d4_report(a.rollout, b.rollout) or 0.0


def W(k: int, A: list, B: list) -> float:
    C = np.array([[exit_distance(k, a, b) for b in B] for a in A])
    return float(ot.emd2(np.full(len(A), 1 / len(A)), np.full(len(B), 1 / len(B)), C))


def entry_of(ch, k: int) -> dict:
    if k == 2:
        return {"P": ch.P}
    if k == 3:
        return {"claims": ch.claims}
    return {"claims": ch.claims, "sections": ch.sections}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--R", type=int, default=6)
    a = ap.parse_args(argv)
    cfg = prereg()
    syn = cfg["synthetic"]
    q = float(cfg["stats"]["envelope_quantile"])
    out = paths.out_dir("E0")
    tasks = gold.campaign50_tasks()[: a.tasks]
    R = a.R
    PA = Params(rho=syn["retrieval_recall"]["A"], kappa=syn["claim_keep"]["A"], pi=syn["section_perm"]["A"], name="A")
    PB = Params(rho=syn["retrieval_recall"]["B"], kappa=syn["claim_keep"]["B"], pi=syn["section_perm"]["B"], name="B")

    chains = {}
    for t in tasks:
        chains[t] = {"A": [run_chain(t, PA, 1000 + r) for r in range(R)], "B": [run_chain(t, PB, 2000 + r) for r in range(R)]}
        print(f"chains {t}", flush=True)

    eps = {k: [] for k in (2, 3, 4)}
    env_pts = {k: [] for k in (2, 3, 4)}
    for t in tasks:
        for k in (2, 3, 4):
            for src in ("A", "B"):
                for ch in chains[t][src][:3]:
                    z = entry_of(ch, k)
                    ea = [run_chain(t, PA, 5000 + r, z) for r in range(3)]
                    eb = [run_chain(t, PB, 6000 + r, z) for r in range(3)]
                    eps[k].append(W(k, ea, eb))
            muA = [entry_of(ch, k) for ch in chains[t]["A"][:3]]
            muB = [entry_of(ch, k) for ch in chains[t]["B"][:3]]
            x = W(k - 1, chains[t]["A"][:3], chains[t]["B"][:3])
            pa = [run_chain(t, PB, 7000 + i, z) for i, z in enumerate(muA)]
            pb = [run_chain(t, PB, 8000 + i, z) for i, z in enumerate(muB)]
            y = W(k, pa, pb)
            env_pts[k].append((x, y))
        print(f"constants {t}", flush=True)
    consts = {}
    for k in (2, 3, 4):
        env = CB.envelope_with_loo(env_pts[k], q)
        rl = CB.ratio_lipschitz(env_pts[k], q)
        consts[k] = {"eps_sup": max(eps[k]), "eps_q95": float(np.quantile(eps[k], 0.95)), "eps_mean": float(np.mean(eps[k])), "L": env.L, "xi": env.xi, "L_ratio_q95": rl["q"], "L_ratio_max": rl["max"], "loo_violation_rate": env.loo_violation_rate, "n_env_points": env.n}
    e1 = [W(1, chains[t]["A"], chains[t]["B"]) for t in tasks]
    consts[1] = {"eps_sup": max(e1), "eps_q95": float(np.quantile(e1, 0.95)), "eps_mean": float(np.mean(e1)), "L": 1.0, "xi": 0.0, "L_ratio_q95": 1.0}
    json.dump(consts, open(out / "constants.json", "w"), indent=1, default=float)

    rows = []
    for t in tasks:
        for m in (1, 2, 3, 4):
            obs = W(m, chains[t]["A"], chains[t]["B"])
            e = {k: consts[k]["eps_q95"] for k in range(1, m + 1)}
            xi = {k: consts[k]["xi"] for k in range(2, m + 1)}
            L = {k: consts[k]["L"] for k in range(2, m + 1)}
            bnd = CB.composed_bound(0.0, e, xi, L, m)
            Lr = {k: consts[k]["L_ratio_q95"] for k in range(2, m + 1)}
            bnd_ratio = CB.composed_bound(0.0, e, {k: 0.0 for k in Lr}, Lr, m)
            rows.append({"task": t, "m": m, "observed": obs, "bound_affine": bnd, "bound_ratio": bnd_ratio, "holds_affine": obs <= bnd + 1e-9, "holds_ratio": obs <= bnd_ratio + 1e-9, "nonvacuous_affine": bnd < 1.0, "nonvacuous_ratio": bnd_ratio < 1.0, "tightness": obs / bnd if bnd > 0 else np.nan})
    cb = pd.DataFrame(rows)
    cb.to_csv(out / "chain_bound.csv", index=False)

    pd.set_option("display.width", 220)
    print("\nconstants:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items() if isinstance(vv, float)} for k, v in consts.items()}, default=float))
    summ = cb.groupby("m").agg(observed=("observed", "mean"), bound_affine=("bound_affine", "mean"), bound_ratio=("bound_ratio", "mean"), holds=("holds_affine", "mean"), nonvacuous=("nonvacuous_affine", "mean"), tightness=("tightness", "median")).round(3)
    summ.to_csv(out / "chain_bound_summary.csv")
    print("\ncomposition bound per stage (mean observed vs bound; fraction of chains on which it holds):")
    print(summ.to_string())


if __name__ == "__main__":
    main()
