"""Entry-conditioned synthesis readouts at the draft and final exits: build writes the conditioned
tables and their peer band at one view, table normalises and certifies them. Usage: python -m
windowbench.conditioned_synthesis build|table [--view system|system_trunc] [--dataset ...]."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import decide as DEC
from . import roster as R

TIERS = {"required": lambda c: c.tier == "required",
         "grounded": lambda c: c.tier in ("required", "admissible") and (c.groundedness or 0) >= 0.9,
         "all": lambda c: c.tier in ("required", "admissible")}
READS = [f"synP_{t}_cov" for t in TIERS] + [f"synP_{t}_reach" for t in TIERS]


VIEWS = {"system": lambda key, task: C.CCB_OUT / "E10/induced" / key / f"{task}.json",
         "system_trunc": lambda key, task: C.CCB_OUT / "E13/induced/system_trunc" / key / f"{task}.json"}


def _induced(key: str, task: str, view: str = "system") -> set | None:
    p = VIEWS[view](key, task)
    if not p.exists():
        return None
    return set(json.load(open(p)).get("claim_match", {}).keys())


def _suffix(view: str) -> str:
    return "" if view == "system" else "_trunc"


def _rows_for(key: str, task: str, papers: set, g, matched: set | None) -> list[dict]:
    rows = []
    for tier, sel in TIERS.items():
        claims = [c for c in g.claims if sel(c)]
        reach = [c for c in claims if set(map(str, c.pmids)) & papers]
        n_t, n_r = len(claims), len(reach)
        hit = sum(1 for c in reach if c.id in matched) if matched is not None else None
        rows.append({"system": key, "task": task, "readout": f"synP_{tier}_cov", "value": (hit / n_r) if (matched is not None and n_r) else np.nan,
                     "n_units": n_r, "note": "" if n_r else "no_reachable_claims"})
        rows.append({"system": key, "task": task, "readout": f"synP_{tier}_reach", "value": (n_r / n_t) if n_t else np.nan, "n_units": n_t, "note": ""})
    return rows


def build(systems: list[str], out: Path, view: str = "system") -> None:
    C.bootstrap_env()
    from ccbench.gt import graphs, peers
    from ccbench.ingest import gold
    from ccbench.model import Rollout
    tasks = gold.campaign50_tasks()
    out.mkdir(parents=True, exist_ok=True)
    G = {}
    rows = []
    for key in systems:
        rdir = C.CCB_OUT / "rollouts" / key
        for t in tasks:
            fp = rdir / f"{t}.json"
            if not fp.exists():
                continue
            ro = Rollout.from_json(fp)
            if ro.is_bot:
                for r in READS:
                    rows.append({"system": key, "task": t, "readout": r, "value": np.nan, "n_units": 0, "note": f"bot:{ro.bot_reason}"})
                continue
            g = G.setdefault(t, graphs.load(t))
            P = set(map(str, ro.papers))
            rows += _rows_for(key, t, P, g, _induced(key, t, view))
        print(f"{key}: {sum(1 for r in rows if r['system'] == key) // len(READS)} tasks", flush=True)
    df = pd.DataFrame(rows)
    df["window"] = np.where(df.system.str.endswith(".draft"), "draft", "system")
    df["system"] = df.system.str.replace(r"\.draft$", "", regex=True)
    dup = df[(df.window == "system") & df.system.map(lambda k: bool(R.SYSTEMS.get(k, {}).get("draft_is_final")))].copy()
    dup["window"] = "draft"
    df = pd.concat([df, dup], ignore_index=True)
    df["view"] = view
    suf = _suffix(view)
    df.to_parquet(out / f"conditioned{suf}.parquet", index=False)
    hrows = []
    for t in tasks:
        g = G.setdefault(t, graphs.load(t))
        for p in [t] + peers.peers(t):
            m = _induced(f"human:{p}", t, view)
            if m is None:
                continue
            gp = graphs.load(p)
            for r in _rows_for(f"human:{p}", t, set(map(str, gp.pmids)), g, m):
                hrows.append({**r, "source_review": p})
    H = pd.DataFrame(hrows)
    H.to_parquet(out / f"conditioned_human{suf}.parquet", index=False)
    print(f"wrote {out}/conditioned{suf}.parquet ({df.system.nunique()} systems) and conditioned_human{suf}.parquet "
          f"({H.source_review.nunique() if len(H) else 0} reviews)")


def load(out: Path, view: str = "system"):
    suf = _suffix(view)
    df = pd.read_parquet(out / f"conditioned{suf}.parquet")
    hp = out / f"conditioned_human{suf}.parquet"
    H = pd.read_parquet(hp) if hp.exists() else pd.DataFrame()
    return df, H


def calibrate(df: pd.DataFrame, H: pd.DataFrame) -> pd.DataFrame:
    C.bootstrap_env()
    from ccbench.rankability.normalise import calibration, normalise
    cal = calibration(H[["task", "readout", "value", "source_review"]])
    d = df.copy()
    d["direction"] = "quality"
    return normalise(d, cal)


def table(dataset: str, with_ours: str | None, out: Path, alpha: float | None = None, view: str = "system") -> dict:
    from .axes import _systems_for, bot_tasks, members, label
    from . import fairness as F
    from .load import Data
    D = Data()
    systems = _systems_for(dataset, with_ours)
    df, H = load(out, view)
    if H.empty:
        raise SystemExit(f"the peer band of view {view} is missing in {out}: run `build --view {view}`")
    Z = calibrate(df, H)
    alpha = alpha or C.REGISTERED["alpha"]
    res = {}
    for window in ("draft", "system"):
        zz = Z[Z.window == window]
        piv = zz.pivot_table(index=["system", "task"], columns="readout", values="z", aggfunc="mean")
        vals = {}
        for s in systems:
            m = [k for k in members(s) if k in piv.index.get_level_values(0)]
            if not m:
                continue
            vals[s] = pd.concat([piv.loc[k].reindex(D.tasks) for k in m], axis=1, keys=m).T.groupby(level=1).mean().T if len(m) > 1 else piv.loc[m[0]].reindex(D.tasks)
        pairs = [(a, b) for a, b in itertools.combinations(systems, 2) if a in vals and b in vals]
        cov = [x for x in READS if x.endswith("_cov")]
        n_fam = DEC.table_pairs(len(systems)) * len(cov)
        rows = []
        for r in cov:
            for a, b in pairs:
                reg = F.family_regime(members(a), members(b)[0], "draft" if window == "draft" else "system", None, D.radii, D.eps_fair)
                fair = reg["regime"] in ("exact", "tolerance")
                res_ = DEC.contrast(DEC.drop_failed(vals[a][r], bot_tasks(D, a, window)),
                                    DEC.drop_failed(vals[b][r], bot_tasks(D, b, window)), D.clusters, alpha / n_fam,
                                    eps=reg["eps_outside"] if fair else np.nan)
                if not fair:
                    res_["certified"], res_["winner"], res_["status"] = False, None, "descriptive"
                rows.append({"window": window, "readout": r, "a": a, "b": b, "tier_regime": reg["regime"], "reason": reg["reason"],
                             "conditioned_on": "reach(P)", "n_in_family": n_fam,
                             **{k: res_[k] for k in ("n", "K", "ours", "opp", "diff", "q", "certified", "winner", "status")}})
        means = pd.DataFrame({s: v.mean() for s, v in vals.items()}).T
        means["n_tasks"] = pd.Series({s: int(v[READS[0]].notna().sum()) for s, v in vals.items()})
        res[window] = {"pairs": pd.DataFrame(rows), "means": means}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--systems", default=None, help="rollout keys; default = every key with an E10 induced dir")
    b.add_argument("--out", default=str(C.OUT_DEFAULT / "conditioned"))
    b.add_argument("--view", default="system", choices=list(VIEWS))
    t = sub.add_parser("table")
    t.add_argument("--view", default="system", choices=list(VIEWS))
    t.add_argument("--dataset", default="same_pool")
    t.add_argument("--with-ours", default=None)
    t.add_argument("--out", default=str(C.OUT_DEFAULT / "conditioned"))
    a = ap.parse_args(argv)
    C.bootstrap_env()
    out = Path(a.out)
    if a.cmd == "build":
        if a.systems:
            systems = a.systems.split(",")
        else:
            root = C.CCB_OUT / "E10/induced" if a.view == "system" else C.CCB_OUT / "E13/induced/system_trunc"
            systems = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith("human:"))
        build(systems, out, a.view)
    else:
        res = table(a.dataset, a.with_ours, out, view=a.view)
        tag = a.dataset + (f"_with_{a.with_ours}" if a.with_ours else "") + _suffix(a.view)
        for w, r in res.items():
            r["pairs"].to_csv(out / f"conditioned_pairs_{tag}_{w}.csv", index=False)
            r["means"].to_csv(out / f"conditioned_means_{tag}_{w}.csv")
            print(f"{w}: {int(r['pairs'].certified.sum()) if len(r['pairs']) else 0} certified of {len(r['pairs'])} cells")
        print(f"wrote {out}/conditioned_pairs_{tag}_*.csv and conditioned_means_{tag}_*.csv")


if __name__ == "__main__":
    main()
