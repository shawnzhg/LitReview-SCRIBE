"""Writes the refusal table: the outside-bound radius of LiRA against each self-retrieving pipeline,
compared with each readout's tolerance. Usage: python refusals.py --radius-dir <dir> [--json
<out.json>]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
_loaded = sys.modules.get("ccbench")
if _loaded is not None and Path(getattr(_loaded, "__file__", "") or "").resolve().parent.parent != EVAL:
    raise SystemExit(f"ccbench already imported from {_loaded.__file__}, not from {EVAL}")
if str(EVAL) in sys.path:
    sys.path.remove(str(EVAL))
sys.path.insert(0, str(EVAL))

import numpy as np
import pandas as pd

from ccbench import paths
from ccbench.config import prereg
from ccbench.fair import lm_falsification as LF
from ccbench.fair import radius as RAD
from ccbench.fair.filter import TOLERANCE
from ccbench.fair.radius_panel_a import paired_xy
from ccbench.fair.units import same_model

REF = ".ref"


def load(radius_dir: Path) -> dict:
    d = Path(radius_dir)
    return {"dist": pd.read_parquet(d / "distances.parquet"),
            "scores": pd.read_parquet(d / "window_scores.parquet",
                                      columns=["system", "task", "panel", "window", "readout", "value"]),
            "radii": pd.read_parquet(d / "radii.parquet", columns=["panel", "window", "readout", "L_s", "L_M", "xi"]),
            "tol": pd.read_csv(d / "tolerances.csv"),
            "design": json.loads((d / "eps_design_selected.json").read_text())}


def entry_mismatch(dist: pd.DataFrame, window: str, how: str) -> dict:
    ent = dist[(dist.kind == "entry") & (dist.window == window)].dropna(subset=["d"])
    return {str(s): RAD.agg(g.d.values, how) for s, g in ent.groupby("a")}


def pipelines(eps_in: dict, reference: str, groups: dict) -> list:
    return [s for s in paths.PANEL_A if s in eps_in and s != reference and not s.endswith(REF)
            and same_model(s, reference, groups)]


def module_constant(dist: pd.DataFrame, window: str, design: str, q: float, exclude) -> tuple:
    xy = paired_xy(dist, window)
    if xy.empty:
        raise SystemExit(f"no paired (self-retrieving, reference-fed) exits at the {window} window: L_M is not measured")
    xy = xy[~xy.system.isin(set(exclude))].reset_index(drop=True)
    L, xi = RAD.fit_LM(xy.x.values, xy.y.values, design, q)
    return L, xi, xy


def score_constants(scores: pd.DataFrame, dist: pd.DataFrame, window: str, design: str, readouts) -> pd.Series:
    systems = sorted(scores[scores.panel == "A"].system.unique())
    out = {}
    for r in readouts:
        rows = RAD.ls_rows(scores, dist, window, r, systems)
        out[r] = RAD.ls_const(rows.ratio.values, design)
    return pd.Series(out, dtype=float)


def tolerance_rows(tol: pd.DataFrame, window: str, design: str, lo: float, hi: float) -> tuple:
    tw = tol[(tol.window == window) & (tol.design == design)]
    adm = tw[(tw.eps_fair > lo) & (tw.eps_fair < hi)]
    return tw, adm.set_index("readout").eps_fair


def slack_count(*eps_in: float) -> int:
    return sum(1 for e in eps_in if e > 0)


def refusal_table(eps_in: dict, reference: str, pipes: list, L: float, xi: float, Ls: pd.Series,
                  eps_fair: pd.Series) -> pd.DataFrame:
    rows = []
    for p in pipes:
        e = eps_in[p] + eps_in[reference]
        slack = xi * slack_count(eps_in[p], eps_in[reference])
        eo = Ls.reindex(eps_fair.index) * (L * e + slack)
        margin = eo - eps_fair
        rows.append({"pair": f"{p} -- {reference}", "pipeline": p, "eps_in_P": eps_in[p],
                     "eps_in_ref": eps_in[reference], "median_eps_out": float(eo.median()),
                     "max_eps_out": float(eo.max()), "median_eps_fair": float(eps_fair.median()),
                     "refused": int((eo > eps_fair).sum()),
                     "readouts": int(eo.notna().sum()), "min_margin": float(margin.min()),
                     "admit_threshold_L_M": float(((eps_fair / Ls.reindex(eps_fair.index) - slack) / e).max())})
    return pd.DataFrame(rows)


def all_refused(eps_in: dict, reference: str, pipes: list, L: float, xi: float, Ls: pd.Series,
                eps_fair: pd.Series) -> bool:
    t = refusal_table(eps_in, reference, pipes, L, xi, Ls, eps_fair)
    return bool((t.refused == t.readouts).all())


def compute(radius_dir: Path, reference: str = "lira", window: str = "writing", exclude=LF.EXCLUDE) -> dict:
    D = load(radius_dir)
    pre = prereg()
    ff = pre["fair_first"]
    q = float(pre["stats"]["envelope_quantile"])
    design = D["design"]
    eps_in = entry_mismatch(D["dist"], window, design["eps_in"])
    pipes = pipelines(eps_in, reference, ff["same_model_groups"])
    L, xi, xy = module_constant(D["dist"], window, design["L_M"], q, exclude)
    tw, eps_fair = tolerance_rows(D["tol"], window, TOLERANCE, float(ff["tolerance_admit"]["gt"]),
                                  float(ff["tolerance_admit"]["lt"]))
    Ls = score_constants(D["scores"], D["dist"], window, design["L_s"], list(eps_fair.index))
    table = refusal_table(eps_in, reference, pipes, L, xi, Ls, eps_fair)
    loso = LF.loso(xy, design["L_M"], q)
    loso_verdicts = {r.held_out: {"L_M": float(r.L_M_train), "all_refused": all_refused(
        eps_in, reference, pipes, float(r.L_M_train), float(r.xi_train), Ls, eps_fair)} for r in loso.itertuples()}
    rad = D["radii"][(D["radii"].panel == "A") & (D["radii"].window == window)].drop_duplicates("readout")
    rad = rad.set_index("readout")
    ratio = xy.y / xy.x.where(xy.x > 1e-6)
    return {"table": table, "n_tolerance_readouts": int(len(tw)), "n_admissible": int(len(eps_fair)),
            "median_eps_fair": float(eps_fair.median()), "median_L_s": float(Ls.median()),
            "L_M": float(L), "xi": float(xi), "n_paired_runs": int(len(xy)),
            "min_ratio": float(ratio.min()), "min_margin": float(table.min_margin.min()),
            "admit_threshold_L_M": float(table.admit_threshold_L_M.max()), "loso": loso_verdicts,
            "radii_L_M": sorted(float(v) for v in rad.L_M.dropna().unique()),
            "L_s_matches_radii": bool(np.allclose(Ls.values, rad.L_s.reindex(Ls.index).values, rtol=0, atol=1e-12)),
            "design": design, "reference": reference, "window": window, "tol_design": TOLERANCE,
            "pipelines": pipes}


def report(res: dict) -> str:
    t = res["table"]
    lines = [f"{'pair':<28}{'eps_in,P':>10}{'eps_in,ref':>12}{'median eps_out':>16}{'median eps_fair':>17}"
             f"{'refused':>10}"]
    for r in t.itertuples():
        lines.append(f"{r.pair:<28}{r.eps_in_P:>10.3f}{r.eps_in_ref:>12.3f}{r.median_eps_out:>16.3f}"
                     f"{r.median_eps_fair:>17.3f}{f'{r.refused}/{r.readouts}':>10}")
    loso = res["loso"]
    Ls = [v["L_M"] for v in loso.values()]
    lines += [f"tolerance readouts ({res['window']}, {res['tol_design']}): {res['n_tolerance_readouts']}; "
              f"admissible: {res['n_admissible']}; median eps_fair {res['median_eps_fair']:.4f}",
              f"L_M ({res['design']['L_M']}, {res['n_paired_runs']} paired runs): {res['L_M']:.4f} (xi {res['xi']:.4f}); "
              f"radius table L_M {', '.join(f'{v:.4f}' for v in res['radii_L_M'])}",
              f"median L_s over admissible readouts: {res['median_L_s']:.4f}; equals radius table: "
              f"{res['L_s_matches_radii']}",
              f"smallest margin eps_out - eps_fair: {res['min_margin']:+.4f}",
              f"largest L_M admitting any readout: {res['admit_threshold_L_M']:.4f}; smallest observed ratio y/x: "
              f"{res['min_ratio']:.4f}",
              f"leave-one-pipeline-out L_M {min(Ls):.4f}-{max(Ls):.4f}: every pair refused on every readout under "
              f"each: {all(v['all_refused'] for v in loso.values())}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-dir", required=True)
    ap.add_argument("--window", default="writing")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    res = compute(Path(a.radius_dir), window=a.window)
    print(report(res))
    if a.json:
        out = dict(res, table=res["table"].to_dict(orient="records"))
        Path(a.json).write_text(json.dumps(out, indent=1, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
