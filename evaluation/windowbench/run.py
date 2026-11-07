"""Builds the per-window comparison cells of one arm against its opponents, each with its fairness
regime and verdict. Usage: python -m windowbench.run --ours <arm> --opponents <set> --out <dir>."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import decide as DEC
from . import fairness as F
from . import roster as R
from .load import Data


def _family_series(M: pd.DataFrame, members: list[str], readout: str, tasks: list[str], bot: dict,
                   window: str) -> pd.Series | None:
    present = [m for m in members if m in M.index.get_level_values(0)]
    if not present or readout not in M.columns:
        return None
    cols = []
    for m in present:
        s = M.loc[m][readout].reindex(tasks) if m in M.index.get_level_values(0) else pd.Series(np.nan, index=tasks)
        cols.append(DEC.drop_failed(s, bot.get((m, window), set())))
    return pd.concat(cols, axis=1).mean(axis=1, skipna=True)


def build_cells(D: Data, ours: list[str], opponents: list[str], windows: list[str], *, alpha: float | None = None) -> pd.DataFrame:
    alpha = alpha if alpha is not None else C.REGISTERED["alpha"]
    rows = []
    for w in windows:
        systems = list(dict.fromkeys(ours + opponents))
        Z = D.z_matrix(w, systems)
        X = D.raw_matrix(w, systems)
        readouts = [(r, "graph") for r in D.readouts_at(w)] + [(r, "extra") for r in D.extra_readouts_at(w)]
        d_eff = alpha / DEC.table_pairs(len(opponents) + 1)
        for r, kind in readouts:
            M = Z if (kind == "graph" and not Z.empty and r in Z.columns) else (X if (not X.empty and r in X.columns) else None)
            if M is None:
                continue
            flags = D.flags(r, w)
            group = D.group(r)
            for o in opponents:
                reg = F.family_regime(ours, o, w, r, D.radii, D.eps_fair)
                a = _family_series(M, ours, r, D.tasks, D.bot, w)
                base = {"window": w, "readout": r, "group": group, "flags": ",".join(flags), "kind": kind,
                        "ours_family": "+".join(ours), "opponent": o, **{k: reg[k] for k in ("regime", "reason", "same_model", "entry_a", "entry_b")},
                        "d_eff": d_eff, "eps": reg["eps_outside"], "opp_finished": D.n_finished(o, w),
                        "ours_finished": int(np.mean([D.n_finished(m, w) for m in ours]))}
                if reg["regime"] == "not_observable" or a is None or o not in M.index.get_level_values(0):
                    opp_v = float(M.loc[o][r].mean()) if o in M.index.get_level_values(0) else np.nan
                    st = "not_observable" if reg["regime"] == "not_observable" else ("ours_absent" if a is None else "opponent_absent")
                    rows.append({**base, "n": 0, "K": 0, "ours": float(a.mean()) if a is not None else np.nan, "opp": opp_v,
                                 "diff": np.nan, "se": np.nan, "q": np.nan, "t": np.nan, "p": np.nan, "certified": False,
                                 "sampling_decided": False, "winner": None, "status": st, "n_missing_a": 0, "n_missing_b": 0})
                    continue
                b = DEC.drop_failed(M.loc[o][r].reindex(D.tasks), D.bot.get((o, w), set()))
                res = DEC.contrast(a, b, D.clusters, d_eff, eps=reg["eps_outside"])
                if reg["regime"] != "exact" and reg["regime"] != "tolerance":
                    res["certified"] = False
                    res["status"] = "descriptive" if res["status"] != "too_few_tasks" else res["status"]
                rows.append({**base, **res})
    cells = pd.DataFrame(rows)
    if cells.empty:
        return cells
    present = set(D.systems_present)
    cells.loc[~cells.opponent.isin(present), "status"] = "not_scored"
    return cells


def resources(D: Data, systems: list[str]) -> pd.DataFrame:
    p = C.SOURCES["conformance"]
    if not p.exists():
        return pd.DataFrame()
    conf = pd.read_csv(p)
    conf = conf[conf.system.isin(systems)]
    if conf.empty:
        return pd.DataFrame()
    agg = conf.groupby("system").agg(tasks=("task", "nunique"), usd_mean=("usd", "mean"), n_llm_mean=("n_llm", "mean"),
                                     n_search_mean=("n_search", "mean"), wall_s_mean=("wall_s", "mean"), words_mean=("words", "mean")).reset_index()
    side = C.OUT_DEFAULT / "draft" / "draft_resources.csv"
    if side.exists():
        dr = pd.read_csv(side)
        dr = dr[(dr.status == "ok") & dr.system.str.endswith(".draft")]
        if len(dr):
            d = dr.assign(base=dr.system.str[: -len(".draft")]).groupby("base").agg(
                n_llm_to_draft_mean=("n_llm_to_draft", "mean"), draft_words_mean=("words", "mean")).reset_index()
            agg = agg.merge(d, left_on="system", right_on="base", how="left").drop(columns=["base"])
    for col in ("backbone", "retrieval_interface", "temperature", "provenance", "model_group"):
        agg[col] = agg.system.map(lambda k: R.SYSTEMS.get(k, {}).get(col))
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True, help="family name or comma-separated system keys")
    ap.add_argument("--opponents", required=True, help="dataset name (fixed_input, same_pool) or comma-separated keys")
    ap.add_argument("--windows", default="all")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    ours = R.resolve_arms(a.ours)
    opps = R.resolve_arms(a.opponents)
    windows = C.REGISTERED["windows"] if a.windows == "all" else a.windows.split(",")
    D = Data()
    cells = build_cells(D, ours, opps, windows, alpha=a.alpha)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / "cells.csv", index=False)
    reg = C.registry({"run": {"ours": ours, "opponents": opps, "windows": windows, "args": vars(a),
                              "n_cells": int(len(cells)), "systems_present": D.systems_present}})
    (out / "registry.json").write_text(json.dumps(reg, indent=1, default=str))
    print(f"wrote {out}/cells.csv ({len(cells)} cells) and registry.json in {time.time()-t0:.1f}s")
    return cells


if __name__ == "__main__":
    main()
