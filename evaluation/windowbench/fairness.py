"""Decides the fairness regime of a system pair at a window: exact, tolerance, refused or not
observable."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import roster as R


def _radius_lookup(radii: pd.DataFrame | None, window: str, a: str, b: str, readout: str):
    if radii is None:
        return None
    w = "system" if window in ("system_trunc", "draft", "draft_trunc") else window
    r = radii[(radii.window == w) & (radii.readout == readout) & (((radii.a == a) & (radii.b == b)) | ((radii.a == b) & (radii.b == a)))]
    if r.empty:
        return None
    row = r.iloc[0]
    return {"status": row.status, "eps_outside": float(row.eps_outside) if not pd.isna(row.eps_outside) else np.nan,
            "L_M": float(row.L_M)}


def admissible(eps_out, eps_fair) -> tuple[bool, str]:
    adm = C.REGISTERED["tolerance_admit"]
    if eps_fair is None or not np.isfinite(eps_fair) or not (adm["gt"] < eps_fair < adm["lt"]):
        return False, "no_admissible_tolerance"
    if eps_out is None or not np.isfinite(eps_out):
        return False, "no_finite_radius"
    if eps_out > eps_fair:
        return False, "eps_out_exceeds_eps_fair"
    return True, ""


def regime(a: str, b: str, window: str, readout: str | None = None, radii: pd.DataFrame | None = None,
           eps_fair: dict | None = None, scaffold: bool = False) -> dict:
    A, B = R.SYSTEMS[a], R.SYSTEMS[b]
    ea, eb = A["entries"].get(window), B["entries"].get(window)
    out = {"same_model": A["model_group"] == B["model_group"], "entry_a": ea, "entry_b": eb,
           "provenance_a": A["provenance"], "provenance_b": B["provenance"], "eps_outside": np.nan}
    if ea is None or eb is None:
        return {**out, "regime": "not_observable", "reason": "no_exit_object:" + ("both" if (ea is None and eb is None) else ("a" if ea is None else "b"))}
    out["cross_model"] = not out["same_model"]
    if scaffold and out["cross_model"]:
        return {**out, "regime": "unfair", "reason": "backbone_outside_scaffold_window", "eps_outside": np.inf}
    if ea == eb and A["provenance"] == B["provenance"]:
        return {**out, "regime": "exact", "reason": "", "eps_outside": 0.0}
    look = _radius_lookup(radii, window, a, b, readout) if readout else None
    eps_out = look["eps_outside"] if (look and look["status"] == "ok") else np.nan
    ef = (eps_fair or {}).get(readout) if readout else None
    ok, why = admissible(eps_out, ef)
    if ok:
        return {**out, "regime": "tolerance", "reason": "", "eps_outside": float(eps_out)}
    return {**out, "regime": "unfair", "reason": why, "eps_outside": eps_out}


def family_regime(ours: list[str], opp: str, window: str, readout: str | None, radii, eps_fair,
                  scaffold: bool = False) -> dict:
    regs = [regime(m, opp, window, readout, radii, eps_fair, scaffold) for m in ours]
    kinds = {(r["regime"], r["reason"]) for r in regs}
    if len(kinds) > 1:
        raise ValueError(f"family {ours} is not homogeneous vs {opp} at {window}: {kinds}")
    return regs[0]
