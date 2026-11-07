"""Computes the controllability distance on the logged systems' branch kernels and checks
contraction, the unique fixed point, the pseudometric properties and zero self-distance. Usage:
python -m ccbench.experiments.E3_distance [--systems ...]."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ccbench import paths
from ccbench.build import load_rollouts
from ccbench.config import prereg
from ccbench.distance.kernel import build_kernel
from ccbench.distance.operator import contraction_check, fixed_point, make_union, pseudometric_check, sup_over_visited, whole_system


def logged_systems() -> list[str]:
    return list(paths.BASELINE_ARMS)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=None, help="comma list of system keys; default: the published pipelines")
    a = ap.parse_args(argv)
    cfg = prereg()["controllability"]
    systems = a.systems.split(",") if a.systems else logged_systems()
    out = paths.out_dir("E3")

    kernels = []
    kinfo = {}
    by_sys: dict[str, list] = {s: [] for s in systems}
    for ro in load_rollouts(systems):
        key = ro.system if ro.mode in (None, "campaign", "native_chain") else f"{ro.system}.{ro.mode}"
        if key in by_sys:
            by_sys[key].append(ro)
    for s in systems:
        k = build_kernel(s, by_sys[s])
        kernels.append(k)
        kinfo[s] = {"n_rollouts": k.n_rollouts, "n_transitions": k.n_transitions, "n_states": len(k.states), "n_labels": len(k.labels), "mean_branching": float(np.mean([len(v) for v in k.trans.values()])) if k.trans else 0.0}
        print(f"kernel {s:20s} rollouts={k.n_rollouts:3d} transitions={k.n_transitions:7d} states={len(k.states):4d} labels={len(k.labels):3d}", flush=True)
    json.dump(kinfo, open(out / "kernels.json", "w"), indent=1)

    U = make_union(kernels)
    print(f"union: {U.n} states, {U.DL.shape[0]} labels; gamma={cfg['gamma']} beta={cfg['beta']} modulus bound={cfg['gamma']*(1-cfg['beta'])}", flush=True)

    cc = contraction_check(U, cfg["gamma"], cfg["beta"], n_trials=3)
    print("contraction ratios:", [round(r, 4) for r in cc["ratios"]], "bound", cc["bound"], "holds", cc["holds"], flush=True)

    print("fixed point from d0=0", flush=True)
    d0, i0 = fixed_point(U, np.zeros((U.n, U.n)), log=True)
    print("fixed point from d0=1", flush=True)
    d1, i1 = fixed_point(U, np.ones((U.n, U.n)), log=True)
    gap = float(np.max(np.abs(d0 - d1)))
    print(f"uniqueness: sup|d*_0 - d*_1| = {gap:.2e}", flush=True)

    pm = pseudometric_check(d0)
    self_distance = {s: float(max(np.abs(U.block(d0, i, i).diagonal()).max(), np.abs(U.block(d1, i, i).diagonal()).max())) for i, s in enumerate(systems)}
    C = whole_system(U, d0)
    Cinf = sup_over_visited(U, d0)
    pd.DataFrame(C, index=systems, columns=systems).round(4).to_csv(out / "C_matrix.csv")
    pd.DataFrame(Cinf, index=systems, columns=systems).round(4).to_csv(out / "Cinf_matrix.csv")
    np.save(out / "fixed_point_matrix.npy", d0)
    checks = {
        "contraction": cc,
        "convergence": {"from_zero": {k: v for k, v in i0.items() if k != "history"}, "from_one": {k: v for k, v in i1.items() if k != "history"}, "uniqueness_gap": gap},
        "pseudometric": pm,
        "self_distance_max": self_distance,
        "self_distance_zero": all(v <= float(cfg["tol"]) for v in self_distance.values()),
        "empirical_contraction_ratio_last": i0["history"][-1]["ratio"],
        "modulus_bound": cfg["gamma"] * (1 - cfg["beta"]),
    }
    json.dump(checks, open(out / "checks.json", "w"), indent=1, default=float)
    json.dump({"from_zero": i0["history"], "from_one": i1["history"]}, open(out / "fixed_point.json", "w"), indent=1)
    print("\nC(S_A,S_B;rho):")
    print(pd.DataFrame(C, index=systems, columns=systems).round(3).to_string())
    print("\nchecks:", json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk in ('holds', 'max_ratio', 'uniqueness_gap', 'triangle_violations', 'symmetry_max_abs', 'diag_max_abs')}) for k, v in checks.items()}, default=float))


if __name__ == "__main__":
    main()
