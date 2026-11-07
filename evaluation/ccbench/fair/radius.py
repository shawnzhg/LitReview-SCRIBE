"""Nuisance radius per comparison cell of the published pipelines from the attribution bound, with
the pre-registered estimator design: the module-sensitivity constant as the 95th-percentile ratio of
paired exit gaps to entry mismatches, score constants at their 95th percentile, and the mean entry
mismatch. Usage: python -m ccbench.fair.radius --out <dir>."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccbench import paths
from ccbench.composition import bound as CB
from ccbench.config import prereg
from ccbench.fair.units import same_model

REF_SUFFIX = ".ref"


def design() -> dict:
    return dict(prereg()["fair_first"]["design"])


def paired_xy(dist: pd.DataFrame, window: str) -> pd.DataFrame:
    ent = dist[(dist.kind == "entry") & (dist.window == window)].dropna(subset=["d"])
    ext = dist[(dist.kind == "exit") & (dist.window == window)].dropna(subset=["d"])
    pair_d: dict[tuple[str, str], float] = {}
    for r in ext.itertuples():
        a, b = str(r.a), str(r.b)
        if a + REF_SUFFIX == b:
            pair_d[(a, r.task)] = float(r.d)
        elif b + REF_SUFFIX == a:
            pair_d[(b, r.task)] = float(r.d)
    ent_d = {(str(r.a), r.task): float(r.d) for r in ent.itertuples()}
    rows = [{"system": s, "task": t, "x": ent_d[(s, t)], "y": y} for (s, t), y in pair_d.items() if (s, t) in ent_d]
    return pd.DataFrame(rows, columns=["system", "task", "x", "y"])


def fit_LM(x: np.ndarray, y: np.ndarray, design: str, q: float) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), 0.0
    if design == "affine":
        L, xi = CB.fit_envelope(x, y, q)
        return float(L), float(xi)
    m = x > 1e-6
    if not m.any():
        return float("nan"), 0.0
    r = y[m] / x[m]
    return (float(np.quantile(r, 0.95)) if design == "ratio_q95" else float(r.max())), 0.0


def ls_rows(scores: pd.DataFrame, dist: pd.DataFrame, window: str, readout: str, systems: list[str]) -> pd.DataFrame:
    s = scores[(scores.window == window) & (scores.readout == readout) & scores.system.isin(systems)].dropna(subset=["value"])
    val = {(r.system, r.task): float(r.value) for r in s.itertuples()}
    x = dist[(dist.kind == "exit") & (dist.window == window) & dist.a.isin(systems) & dist.b.isin(systems)].dropna(subset=["d"])
    out = []
    for r in x.itertuples():
        if r.d > 1e-6 and (r.a, r.task) in val and (r.b, r.task) in val:
            out.append((r.task, abs(val[(r.a, r.task)] - val[(r.b, r.task)]) / r.d))
    return pd.DataFrame(out, columns=["task", "ratio"])


def ls_const(r: np.ndarray, design: str) -> float:
    if not len(r):
        return float("nan")
    return float(np.quantile(r, 0.95)) if design == "q95" else float(r.max())


def agg(v: np.ndarray, design: str) -> float:
    if not len(v):
        return float("nan")
    return {"mean": float(np.mean(v)), "q95": float(np.quantile(v, 0.95)), "sup": float(np.max(v))}[design]


def radii(scores: pd.DataFrame, dist: pd.DataFrame, des: dict, cfg: dict) -> pd.DataFrame:
    q_env = float(prereg()["stats"]["envelope_quantile"])
    groups = cfg["same_model_groups"]
    exclude = set(cfg["lm_exclude"])
    LM = {}
    for w in ("writing",):
        xy = paired_xy(dist, w)
        xy = xy[~xy.system.isin(exclude)]
        LM[w] = fit_LM(xy.x.values, xy.y.values, des["L_M"], q_env)
    systems = sorted(scores[scores.panel == "A"].system.unique())
    rows = []
    for w in ("retrieval", "writing", "system"):
        common = w in ("retrieval", "system")
        sc = scores[(scores.window == w) & scores.system.isin(systems)]
        ent = dist[(dist.kind == "entry") & (dist.window == w) & dist.a.isin(systems)]
        ein = {s: ent[ent.a == s].dropna(subset=["d"]).set_index("task").d for s in systems}
        enote = {s: (ent[ent.a == s].note.mode().iloc[0] if len(ent[ent.a == s]) else "") for s in systems}
        L_pt, xi_pt = (0.0, 0.0) if common else LM.get(w, (float("nan"), 0.0))
        for readout in sorted(sc.readout.unique()):
            Ls_pt = 0.0 if common else ls_const(ls_rows(scores, dist, w, readout, systems).ratio.values, des["L_s"])
            for a, b in itertools.combinations(systems, 2):
                sm = same_model(a, b, groups)
                ea, eb = (0.0, 0.0) if common else (agg(ein[a].values, des["eps_in"]), agg(ein[b].values, des["eps_in"]))
                base = {"panel": "A", "window": w, "a": a, "b": b, "readout": readout, "eps_in_a": ea, "eps_in_b": eb, "L_M": L_pt, "xi": xi_pt, "L_s": Ls_pt, "same_model": sm, "entry_note_a": enote[a], "entry_note_b": enote[b], "n_tasks_a": int(len(ein[a])), "n_tasks_b": int(len(ein[b]))}
                if common and enote[a].split("|")[-1] != enote[b].split("|")[-1]:
                    rows.append({**base, "eps_outside": np.nan, "status": "provenance_mismatch"})
                    continue
                if any(np.isnan(v) for v in (ea, eb, L_pt, Ls_pt)):
                    rows.append({**base, "eps_outside": np.nan, "status": "no_estimate"})
                    continue
                if not sm:
                    rows.append({**base, "eps_outside": np.nan, "status": "cross_model"})
                    continue
                core = 0.0 if common else Ls_pt * (L_pt * (ea + eb) + xi_pt * (int(ea > 0) + int(eb > 0)))
                rows.append({**base, "eps_outside": core, "status": "ok"})
        print(f"radii {w}: {sc.readout.nunique()} readouts", flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for radii.parquet and eps_design_selected.json")
    a = ap.parse_args(argv)
    cfg = prereg()["fair_first"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(paths.OUT / "E13" / "window_scores.parquet")
    scores = scores[scores.panel == "A"]
    dist = pd.read_parquet(paths.OUT / "E13" / "distances.parquet")
    des = design()
    json.dump(des, open(out / "eps_design_selected.json", "w"), indent=1)
    R = radii(scores, dist, des, cfg)
    R.to_parquet(out / "radii.parquet", index=False)
    ok = R[R.status == "ok"]
    print(f"design: {des}")
    print(f"radii: {len(R)} cells, {len(ok)} with an estimate; median eps_outside by window:")
    print(ok.groupby("window").eps_outside.median().round(3).to_string())


if __name__ == "__main__":
    main()
