"""Builds the capability-axis tables: per-axis certified contrasts, the equal-axis comparison, rank
intervals and refinement effects. Usage: python -m windowbench.axes table --dataset
fixed_input|same_pool --with-ours <arm> [--out <dir>]."""

from __future__ import annotations

import argparse
import itertools
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
from .run import resources

AXES = C.REGISTERED["axes"]


def members(system: str) -> list[str]:
    if system in R.FAMILIES:
        return list(R.FAMILIES[system])
    if system in R.SYSTEMS:
        return [system]
    raise KeyError(f"unknown system or family {system!r}")


def representative(system: str) -> str:
    return members(system)[0]


def label(system: str) -> str:
    m = members(system)
    if len(m) > 1:
        return f"{system} ({len(m)} seeds)"
    return R.SYSTEMS[m[0]]["label"]


def axis_readouts(D: Data, axis: str, obs: str) -> list[str]:
    groups = set(C.REGISTERED["axis_groups"][axis])
    excl = set(C.REGISTERED["axis_exclude_flags"])
    out = []
    for r in D.readouts_at(obs):
        if r == "completion" or D.group(r) not in groups:
            continue
        if excl & set(D.flags(r, obs)):
            continue
        out.append(r)
    return out


def axis_values(D: Data, systems: list[str], axis: str, obs: str, reads: list[str]) -> pd.DataFrame:
    keys = [k for s in systems for k in members(s)]
    Z = D.z_matrix(obs, keys)
    if Z.empty:
        return pd.DataFrame()
    rows = {}
    for s in systems:
        m = [k for k in members(s) if k in Z.index.get_level_values(0)]
        if not m:
            continue
        if axis == "retrieval" and any(R.SYSTEMS[k]["entries"]["retrieval"] is None for k in m):
            continue
        per_seed = pd.concat([Z.loc[k].reindex(columns=reads).mean(axis=1).rename(k) for k in m], axis=1)
        v = per_seed.mean(axis=1).reindex(D.tasks)
        if v.notna().any():
            rows[s] = v
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def bot_tasks(D: Data, system: str, obs: str) -> set:
    sets = [D.bot.get((k, obs), set()) for k in members(system)]
    return set.intersection(*sets) if sets else set()


def axis_pairs(D: Data, systems: list[str], obs: str) -> list[tuple[str, str, dict]]:
    out = []
    for a, b in itertools.combinations(systems, 2):
        reg = F.family_regime(members(a), representative(b), obs, None, D.radii, D.eps_fair)
        out.append((a, b, reg))
    return out


def axis_confirmatory(D: Data, systems: list[str], obs: str, axes: list[str] | None = None,
                      alpha: float | None = None) -> pd.DataFrame:
    alpha = alpha if alpha is not None else C.REGISTERED["alpha"]
    axes = axes or AXES
    pairs = axis_pairs(D, systems, obs)
    n_pairs = DEC.table_pairs(len(systems))
    reads_of = {ax: axis_readouts(D, ax, obs) for ax in axes}
    values = {ax: axis_values(D, systems, ax, obs, reads_of[ax]) for ax in axes if reads_of[ax]}
    live_axes = [ax for ax in axes if ax in values and len(values[ax].index) >= 2]
    n_axes = max(1, len(live_axes))
    rows = []
    for ax in live_axes:
        reads, V = reads_of[ax], values[ax]
        for a, b, reg in pairs:
            row = {"obs": obs, "axis": ax, "a": a, "b": b, "regime": reg["regime"], "reason": reg["reason"],
                   "cross_model": bool(reg.get("cross_model", False)), "n_readouts": len(reads),
                   "n_pairs_in_family": n_pairs, "n_axes_in_family": n_axes,
                   "n_axes_with_rows": None, "d_eff": alpha / (n_pairs * n_axes)}
            if a not in V.index or b not in V.index:
                rows.append({**row, "status": "not_scored", "certified": False, "winner": None})
                continue
            sa = DEC.drop_failed(V.loc[a], bot_tasks(D, a, obs))
            sb = DEC.drop_failed(V.loc[b], bot_tasks(D, b, obs))
            eps = reg["eps_outside"] if reg["regime"] in ("exact", "tolerance") else np.nan
            res = DEC.contrast(sa, sb, D.clusters, alpha / (n_pairs * n_axes), eps=eps)
            if reg["regime"] not in ("exact", "tolerance"):
                res["certified"] = False
                res["status"] = "descriptive"
                res["winner"] = None
            rows.append({**row, **res})
    out = pd.DataFrame(rows)
    if len(out):
        out["n_axes_with_rows"] = out.axis.nunique()
    return out


def axis_z_rows(D: Data, systems: list[str], obs: str) -> pd.DataFrame:
    rows = []
    for ax in AXES:
        reads = axis_readouts(D, ax, obs)
        if not reads:
            continue
        V = axis_values(D, systems, ax, obs, reads)
        for s_ in V.index:
            v = DEC.drop_failed(V.loc[s_], bot_tasks(D, s_, obs)).dropna()
            rows += [{"obs": obs, "axis": ax, "system": s_, "task": t, "z": float(z)} for t, z in v.items()]
    return pd.DataFrame(rows, columns=["obs", "axis", "system", "task", "z"])


def composite(D: Data, z: pd.DataFrame, obs: str, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    C.bootstrap_env()
    from ccbench.merge_windowbench import composite as ccb_composite
    U, axes = ccb_composite(z, obs, [s_ for s_ in names if s_ in set(z.system)])
    return (U.reindex(columns=D.tasks) if not U.empty else U), axes


def primary_confirmatory(D: Data, systems: list[str], obs: str, alpha: float | None = None) -> pd.DataFrame:
    alpha = alpha if alpha is not None else C.REGISTERED["alpha"]
    pairs = axis_pairs(D, systems, obs)
    adm = [(a, b, r) for a, b, r in pairs if r["regime"] in ("exact", "tolerance")]
    n_pairs = DEC.table_pairs(len(systems))
    z = axis_z_rows(D, systems, obs)
    if z.empty:
        return pd.DataFrame()
    rankable = all(r["regime"] == "exact" for _a, _b, r in adm)
    rows = []
    is_conf = obs == C.REGISTERED["axis_primary_observation"]
    for a, b, reg in adm:
        base = {"role": "confirmatory" if is_conf else "secondary_descriptive",
                "obs": obs, "family": "primary_equal_axes",
                "a": a, "b": b, "regime": reg["regime"], "cross_model": bool(reg.get("cross_model", False)),
                "n_pairs_in_family": n_pairs, "d_eff": alpha / n_pairs, "rankable_readouts": rankable}
        try:
            U, axes = composite(D, z, obs, [a, b])
        except ValueError:
            rows.append({**base, "status": "axis_sets_differ", "certified": False, "winner": None})
            continue
        if a not in U.index or b not in U.index:
            continue
        sa, sb = U.loc[a], U.loc[b]
        res = DEC.contrast(sa, sb, D.clusters, alpha / n_pairs, eps=reg["eps_outside"])
        rows.append({**base, "n_readouts": sum(len(axis_readouts(D, ax, obs)) for ax in axes), "n_axes": len(axes), **res})
    return pd.DataFrame(rows)


def rank_intervals(conf: pd.DataFrame, systems: list[str]) -> pd.DataFrame:
    C.bootstrap_env()
    from ccbench.fair.composite import rank_intervals as closure_intervals
    rows = []
    for (obs, ax), g in conf.groupby(["obs", "axis"], sort=False):
        present = [s for s in systems if (s in set(g.a) | set(g.b))]
        n = len(present)
        rel = set()
        for r in g.itertuples():
            if r.certified == True and r.winner in ("a", "b"):
                rel.add((r.a, r.b) if r.winner == "a" else (r.b, r.a))
        iv, _acyclic, _naive = closure_intervals(present, rel)
        for s in present:
            above = below = xm = 0
            mean_z = np.nan
            for r in g.itertuples():
                cert = r.certified == True
                if r.a == s:
                    mean_z = r.ours if r.ours == r.ours else mean_z
                    if cert:
                        below += r.winner == "a"
                        above += r.winner == "b"
                        xm += bool(getattr(r, "cross_model", False))
                elif r.b == s:
                    mean_z = r.opp if r.opp == r.opp else mean_z
                    if cert:
                        below += r.winner == "b"
                        above += r.winner == "a"
                        xm += bool(getattr(r, "cross_model", False))
            adm = int(((g.a == s) | (g.b == s)) [g.regime.isin(["exact", "tolerance"])].sum())
            lo, hi = iv[s]
            rows.append({"obs": obs, "axis": ax, "system": s, "mean_z": mean_z, "lo": lo, "hi": hi,
                         "wins": below, "losses": above, "admissible_pairs": adm, "n_systems": n,
                         "certified_position": lo == hi, "cross_model_certs": xm})
    return pd.DataFrame(rows)


def big_table(D: Data, systems: list[str], obs_windows: list[str]) -> dict:
    draft_axes = C.REGISTERED["draft_exit_axes"]
    conf = pd.concat([axis_confirmatory(D, systems, o, axes=draft_axes if o in DRAFT_OF.values() else None)
                      for o in obs_windows], ignore_index=True)
    conf = conf[conf.status != "not_scored"] if not conf.empty else conf
    ri = rank_intervals(conf, systems) if not conf.empty else pd.DataFrame()
    prim = [primary_confirmatory(D, systems, o) for o in obs_windows]
    prim = pd.concat([p for p in prim if not p.empty], ignore_index=True) if any(not p.empty for p in prim) else pd.DataFrame()
    return {"conf": conf, "ranks": ri, "primary": prim}


DRAFT_OF = {"system_trunc": "draft_trunc", "system": "draft"}


def refinement_effect(D: Data, systems: list[str], axes: list[str] | None = None, alpha: float | None = None,
                      obs_final: str = "system_trunc") -> pd.DataFrame:
    alpha = alpha if alpha is not None else C.REGISTERED["alpha"]
    obs_draft = DRAFT_OF[obs_final]
    axes = axes or list(C.REGISTERED["draft_exit_axes"])
    with_draft = [s for s in systems if any(R.SYSTEMS[k]["entries"].get(obs_draft) for k in members(s))]
    refiners = [s for s in with_draft if not all(R.SYSTEMS[k].get("draft_is_final") for k in members(s))]
    reads_of = {ax: sorted(set(axis_readouts(D, ax, obs_final)) & set(axis_readouts(D, ax, obs_draft))) for ax in axes}
    n_fam = max(1, len(refiners) * sum(1 for ax in axes if reads_of[ax]))
    rows = []
    for ax in axes:
        reads = reads_of[ax]
        if not reads:
            continue
        Vd = axis_values(D, with_draft, ax, obs_draft, reads)
        Vf = axis_values(D, with_draft, ax, obs_final, reads)
        for s in with_draft:
            row = {"obs_final": obs_final, "obs_draft": obs_draft, "axis": ax, "system": s, "n_readouts": len(reads),
                   "n_in_family": n_fam, "d_eff": alpha / n_fam,
                   "draft_is_final": s not in refiners}
            if row["draft_is_final"]:
                rows.append({**row, "status": "no_refinement_stage", "diff": 0.0, "certified": False, "winner": None,
                             "draft_z": float(Vf.loc[s].mean()) if s in Vf.index else np.nan, "final_z": float(Vf.loc[s].mean()) if s in Vf.index else np.nan})
                continue
            if s not in Vd.index or s not in Vf.index:
                rows.append({**row, "status": "not_scored", "diff": np.nan, "certified": False, "winner": None})
                continue
            failed = bot_tasks(D, s, obs_final) | bot_tasks(D, s, obs_draft)
            res = DEC.contrast(DEC.drop_failed(Vf.loc[s], failed), DEC.drop_failed(Vd.loc[s], failed),
                               D.clusters, alpha / n_fam, eps=0.0)
            rows.append({**row, **res, "draft_z": res["opp"], "final_z": res["ours"],
                         "effect": ("refinement_helps" if res["diff"] > 0 else "refinement_hurts") if res["certified"] else "undecided"})
    return pd.DataFrame(rows)


def draft_ladder(conf: pd.DataFrame, obs_final: str = "system_trunc") -> pd.DataFrame:
    obs_draft = DRAFT_OF[obs_final]
    f = conf[(conf.obs == obs_final) & conf.axis.isin(C.REGISTERED["draft_exit_axes"]) & conf.regime.isin(["exact", "tolerance"])]
    d = conf[(conf.obs == obs_draft) & conf.regime.isin(["exact", "tolerance"])].set_index(["axis", "a", "b"])
    rows = []
    for r in f.itertuples():
        key = (r.axis, r.a, r.b)
        if key not in d.index:
            continue
        x = d.loc[key]
        sig_f = (r.winner if r.certified else None)
        sig_d = (x.winner if bool(x.certified) else None)
        if sig_f is None and sig_d is None:
            att = "none"
        elif sig_f is not None and sig_d == sig_f:
            att = "present_at_draft_persists"
        elif sig_f is not None and sig_d is None:
            att = "appears_after_refinement"
        elif sig_f is None and sig_d is not None:
            att = "present_at_draft_removed_by_refinement"
        else:
            att = "sign_flips_in_refinement"
        rows.append({"axis": r.axis, "a": r.a, "b": r.b, "draft_diff": float(x["diff"]), "draft_q": float(x.q) if x.q == x.q else np.nan,
                     "draft_certified": bool(x.certified), "draft_winner": sig_d, "final_diff": float(r.diff), "final_q": float(r.q) if r.q == r.q else np.nan,
                     "final_certified": bool(r.certified), "final_winner": sig_f, "attribution": att})
    return pd.DataFrame(rows)


def _systems_for(dataset: str, with_ours: str | None) -> list[str]:
    base = list(R.DATASETS[dataset]) if dataset in R.DATASETS else R.resolve_arms(dataset)
    if with_ours:
        for s in with_ours.split(","):
            s = s.strip()
            if s and s not in base:
                base.append(s)
    return base


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--dataset", default="fixed_input", help="fixed_input | same_pool | comma-separated keys")
    t.add_argument("--with-ours", default=None, help="our family or keys to add")
    t.add_argument("--obs", default="system_trunc,system,retrieval,planning")
    t.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    D = Data()
    obs = a.obs.split(",")
    systems = _systems_for(a.dataset, a.with_ours)
    res = big_table(D, systems, obs)
    out = Path(a.out) if a.out else C.OUT_DEFAULT / (f"axes_{a.dataset}" + (f"_with_{a.with_ours.replace(',', '+')}" if a.with_ours else ""))
    out.mkdir(parents=True, exist_ok=True)
    res["conf"].to_csv(out / "axis_confirmatory.csv", index=False)
    res["ranks"].to_csv(out / "axis_ranks.csv", index=False)
    if not res["primary"].empty:
        res["primary"].to_csv(out / "primary_confirmatory.csv", index=False)
    resources(D, [k for s in systems for k in members(s)]).to_csv(out / "resources.csv", index=False)
    primary = C.REGISTERED["axis_primary_observation"] if C.REGISTERED["axis_primary_observation"] in obs else obs[0]
    if DRAFT_OF.get(primary) in obs:
        refinement_effect(D, systems, obs_final=primary).to_csv(out / "refinement_effect.csv", index=False)
        draft_ladder(res["conf"], obs_final=primary).to_csv(out / "draft_ladder.csv", index=False)
    (out / "registry.json").write_text(json.dumps(C.registry({"systems": systems, "obs": obs, "args": vars(a)}), indent=1, default=str))
    print(f"wrote {out}/ in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
