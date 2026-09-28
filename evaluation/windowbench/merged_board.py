"""Exports per-task axis scores, task clusters and readout admission from the windowbench tables and
runs ccbench.merge_windowbench per observation point and tier; board/build_board.py drives it."""

from __future__ import annotations

import hashlib
import importlib
import json
import time
from pathlib import Path

import pandas as pd

from . import axes as A
from . import config as C
from . import decide as DEC
from .load import Data

EXPORT_SYSTEMS = ["autosurvey.ref", "surveyg.ref", "llmxmr.ref", "sgi.ref", "lira",
                  "autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr", "drtulu"]
EXPORT_OBS = ["system", "system_trunc", "draft", "draft_trunc", "retrieval", "planning"]
ADMISSION_OBS = ["system", "system_trunc", "retrieval", "planning"]

FLAG_NOTE = {"guardrail": "every human at 1; pass-through, not a quality axis",
             "no_topical_signal": "wrong-review control scores >= human peers",
             "saturated": "peer q10 == 1.000"}
AXIS_OF_GROUP = {g: ax for ax, gs in C.REGISTERED["axis_groups"].items() for g in gs}


EXCLUSION_PRECEDENCE = ("channel_defect", "guardrail", "no_topical_signal", "saturated", "non_discriminating")


def _exclusion(flags: list[str], hard: list[str]) -> list[str]:
    for f in EXCLUSION_PRECEDENCE:
        if f in hard:
            if f == "non_discriminating" and "humans_incomparable" in flags:
                return ["humans_incomparable", "non_discriminating"]
            return [f]
    return sorted(hard)


def export_readout_admission(D: Data, obs_list: list[str]) -> pd.DataFrame:
    rows = []
    for obs in obs_list:
        for r in sorted(D.readouts_at(obs)):
            flags = D.flags(r, obs)
            grp = D.group(r)
            axis = AXIS_OF_GROUP.get(grp)
            hard = sorted(f for f in flags if f in set(C.REGISTERED["axis_exclude_flags"]))
            admitted = bool(axis) and not hard and r != "completion"
            excl = _exclusion(flags, hard) if not admitted else []
            note = C.REGISTERED["channel_defects"].get(r, "") if "channel_defect" in excl else ""
            if not note:
                note = next((FLAG_NOTE[f] for f in excl if f in FLAG_NOTE), "")
            rows.append({"obs": obs, "readout": r, "group": grp, "axis_used": axis if admitted else None,
                         "admitted": admitted, "exclusion": ",".join(excl), "note": note})
    return pd.DataFrame(rows)


def export_per_task_axis_z(D: Data, systems: list[str], obs_list: list[str], clusters: dict) -> pd.DataFrame:
    rows = []
    for obs in obs_list:
        for ax in A.AXES:
            reads = A.axis_readouts(D, ax, obs)
            if not reads:
                continue
            V = A.axis_values(D, systems, ax, obs, reads)
            if V.empty:
                continue
            for s in V.index:
                v = DEC.drop_failed(V.loc[s], A.bot_tasks(D, s, obs)).dropna()
                for t, z in v.items():
                    rows.append({"obs": obs, "axis": ax, "system": s, "task": t, "z": float(z),
                                 "cluster": clusters.get(t, t), "n_readouts": len(reads)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_o"] = pd.Categorical(df.obs, categories=[o for o in obs_list], ordered=True)
    df["_a"] = pd.Categorical(df.axis, categories=list(A.AXES), ordered=True)
    df = df.sort_values(["_o", "_a", "system", "task"]).drop(columns=["_o", "_a"])
    return df.reset_index(drop=True)


def export_per_readout_mean_z(D: Data, systems: list[str], obs_list: list[str]) -> pd.DataFrame:
    rows = []
    for obs in obs_list:
        keys = sorted({k for s in systems for k in A.members(s)})
        reads = sorted(D.readouts_at(obs))
        Z = D.z_matrix(obs, keys)
        if Z.empty:
            continue
        for k in keys:
            if k not in Z.index.get_level_values(0):
                continue
            M = Z.loc[k].reindex(columns=reads)
            for r in reads:
                n = int(M[r].notna().sum())
                if not n:
                    continue
                rows.append({"obs": obs, "system": k, "readout": r, "group": D.group(r),
                             "mean_z": float(M[r].mean()), "n_tasks": n})
    return pd.DataFrame(rows)


def write_inputs(D: Data, systems: list[str], obs_list: list[str], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    clusters = dict(D.clusters)
    tc = pd.DataFrame(sorted(clusters.items()), columns=["task", "cluster"])
    z = export_per_task_axis_z(D, systems, obs_list, clusters)
    adm = export_readout_admission(D, [o for o in ADMISSION_OBS if o in obs_list] or ADMISSION_OBS)
    mz = export_per_readout_mean_z(D, systems, obs_list)
    files = {"per_task_axis_z.csv": z.round(6), "task_clusters.csv": tc,
             "readout_admission.csv": adm, "per_readout_mean_z.csv": mz.round(5)}
    meta = {"builder": "windowbench/merged_board.py::write_inputs", "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "systems": systems, "obs": obs_list, "n_tasks": len(clusters), "n_clusters": len(set(clusters.values())),
            "files": {}}
    for name, df in files.items():
        fp = out / name
        df.to_csv(fp, index=False)
        meta["files"][name] = {"rows": int(len(df)), "columns": list(df.columns), "sha256_16": _sha16(fp)}
        print(f"  wrote {name:26s} {len(df):7d} rows  sha256_16={meta['files'][name]['sha256_16']}")
    meta["registry"] = C.registry({"systems": systems, "obs": obs_list})
    (out / "inputs.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def import_merge_windowbench():
    C.bootstrap_env()
    mod = importlib.import_module("ccbench.merge_windowbench")
    if Path(mod.__file__).resolve().parent != C.CCB:
        raise ImportError(f"ccbench.merge_windowbench resolved to {mod.__file__}, not the repo package {C.CCB}")
    return mod


def run_board(merge, z: pd.DataFrame, clusters: dict, tiers: dict, obs_list: list[str], highlight: list[str],
              quiet: bool = False, bots: dict | None = None):
    boards, pairs, metas, lines = [], [], [], []
    bots = bots or {}
    for obs in obs_list:
        for tier, names in tiers.items():
            have = [n for n in names if n in set(z[z.obs == obs].system)]
            missing = [n for n in names if n not in have]
            if len(have) < 3:
                lines.append(f"\n{obs}/{tier}: only {len(have)} systems present, no board")
                continue
            try:
                U, axes = merge.composite(z, obs, have)
            except ValueError as e:
                lines.append(f"\n{obs}/{tier}: NOT COMPARABLE -- {e}")
                continue
            b, p, m = merge.leaderboard(U, clusters)
            n_tasks = U.notna().sum(axis=1).to_dict()
            b["n_tasks"] = b.system.map(n_tasks).astype("Int64")
            b["n_bot"] = b.system.map(lambda x: bots.get((obs, x))).astype("Int64")
            b.insert(0, "tier", tier); b.insert(0, "obs", obs)
            p.insert(0, "tier", tier); p.insert(0, "obs", obs)
            m |= dict(obs=obs, tier=tier, n_axes=len(axes), axes=",".join(axes), missing=",".join(missing),
                      n_tasks={k: int(v) for k, v in U.notna().sum(axis=1).to_dict().items()},
                      n_bot={k: bots.get((obs, k)) for k in U.index})
            boards.append(b); pairs.append(p); metas.append(m)
            tag_s = "PRIMARY" if obs == C.REGISTERED["axis_primary_observation"] else "secondary"
            lines.append(f"\n{'='*78}\n{tier} / {obs}  ({tag_s})   {len(axes)} axes: {', '.join(axes)}"
                         + (f"   [absent: {', '.join(missing)}]" if missing else ""))
            lines.append(f"  certified {m['certified']}/{m['pairs']}  C_dec={m['C_dec']:.2f}   acyclic={m['acyclic']}  "
                         f"naive-rank-ok={m['naive_rank_formula_ok']}  tierable={m['tierable']}")
            lines.append(f"  {'Display':>7}  {'Agent':24}{'Composite C':>12}{'Certified rank':>16}"
                         f"{'n_tasks':>9}{'failed':>8}")
            for r in b.itertuples():
                cr = str(r.cert_lo) if r.cert_lo == r.cert_hi else f"{r.cert_lo}-{r.cert_hi}"
                star = " *" if r.system in highlight else "  "
                nb = "-" if pd.isna(r.n_bot) else str(int(r.n_bot))
                lines.append(f"  {r.display_rank:>7}  {r.system.replace('.ref','•'):24}{r.composite:12.3f}{cr:>16}"
                             f"{int(r.n_tasks):>9}{nb:>8}{star}")
            if m["relations"]:
                lines.append("  " + ", ".join(x.replace(".ref", "•") for x in m["relations"]))
    if not quiet:
        print("\n".join(lines))
    return boards, pairs, metas, "\n".join(lines)


def _sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
