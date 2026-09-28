"""Normalises readouts to the fraction-of-human scale: quality between the peer band and the target
review, style as distance from the peer band."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration(H: pd.DataFrame, lo: float = 0.10, hi: float = 0.90) -> pd.DataFrame:
    H = H.copy()
    H["is_self"] = H.source_review == H.task
    rows = []
    for (t, r), g in H.groupby(["task", "readout"]):
        own = g[g.is_self].value.dropna()
        peers = g[~g.is_self].value.dropna()
        rows.append({"task": t, "readout": r, "own": float(own.iloc[0]) if len(own) else np.nan, "q_lo": float(peers.quantile(lo)) if len(peers) else np.nan, "q_hi": float(peers.quantile(hi)) if len(peers) else np.nan, "median": float(peers.median()) if len(peers) else np.nan, "n_peers": int(len(peers)), "peer_sd": float(peers.std(ddof=0)) if len(peers) > 1 else np.nan})
    return pd.DataFrame(rows)


def normalise(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(cal, on=["task", "readout"], how="left")
    z = np.full(len(m), np.nan)
    note = np.array([""] * len(m), dtype=object)
    for i, r in enumerate(m.itertuples()):
        s = r.value
        if s is None or (isinstance(s, float) and np.isnan(s)):
            note[i] = "bot"
            continue
        own, qlo, qhi = r.own, r.q_lo, r.q_hi
        if getattr(r, "family", "") == "task" or r.readout == "completion":
            z[i] = float(s)
            note[i] = "pass_through"
            continue
        if r.direction == "quality" and not np.isnan(own) and own >= 0.999 and not np.isnan(qlo) and qlo >= 0.999:
            z[i] = min(1.0, max(0.0, float(s)))
            note[i] = "guardrail"
            continue
        if r.direction == "quality":
            if np.isnan(qlo):
                if np.isnan(own) or own <= 0:
                    note[i] = "no_calibration"
                    continue
                z[i] = min(1.0, max(0.0, s / own))
                note[i] = "no_peers"
                continue
            ceil = own if not np.isnan(own) else qhi
            if np.isnan(ceil) or ceil <= qlo:
                ceil = max(ceil if not np.isnan(ceil) else -1, qhi)
            if ceil <= qlo:
                note[i] = "non_discriminating_task"
                continue
            z[i] = min(1.0, max(0.0, (s - qlo) / (ceil - qlo)))
        else:
            if np.isnan(qlo) or np.isnan(qhi):
                if np.isnan(own):
                    note[i] = "no_calibration"
                    continue
                z[i] = 1.0 if abs(s - own) < 1e-9 else max(0.0, 1 - abs(s - own))
                note[i] = "no_peers"
                continue
            width = max(qhi - qlo, 0.05)
            d = 0.0 if qlo <= s <= qhi else (qlo - s if s < qlo else s - qhi)
            z[i] = 1 - min(1.0, d / width)
    m["z"] = z
    m["z_note"] = note
    return m


def discriminability(cal: pd.DataFrame, H: pd.DataFrame, S: pd.DataFrame | None = None, direction: dict | None = None, eps_fair_q: float = 0.5, eps_ge: float = 0.5, sd_lt: float = 0.02) -> pd.DataFrame:
    H = H.copy()
    H["is_self"] = H.source_review == H.task
    gaps = []
    for (t, r), g in H.groupby(["task", "readout"]):
        own = g[g.is_self].value.dropna()
        if own.empty:
            continue
        for v in g[~g.is_self].value.dropna():
            gaps.append({"readout": r, "gap": abs(v - float(own.iloc[0]))})
    gaps = pd.DataFrame(gaps)
    ef = gaps.groupby("readout").gap.quantile(eps_fair_q).rename("eps_fair") if len(gaps) else pd.Series(dtype=float, name="eps_fair")
    sd = cal.groupby("readout").peer_sd.mean().rename("peer_sd_mean")
    nt = cal.groupby("readout").n_peers.mean().rename("n_peers_mean")
    own_mean = cal.groupby("readout").own.mean().rename("own_mean")
    parts = [ef, sd, nt, own_mean]
    if S is not None and len(S):
        sys_sd = S.groupby(["readout", "system"]).value.mean().groupby("readout").std(ddof=0).rename("system_sd")
        parts.append(sys_sd)
    out = pd.concat(parts, axis=1).reset_index()
    if "system_sd" not in out:
        out["system_sd"] = np.nan
    if direction:
        out["direction"] = out.readout.map(direction)
    guardrail = (out.own_mean >= 0.999) & (out.peer_sd_mean.fillna(0) < sd_lt)
    out["guardrail"] = guardrail | (out.readout == "completion")
    out["non_discriminating"] = ((out.peer_sd_mean < sd_lt) | out.eps_fair.isna() | (out.system_sd.fillna(1.0) < sd_lt)) & ~out["guardrail"]
    out["humans_incomparable"] = out.eps_fair >= eps_ge
    out["reason"] = np.where(out["guardrail"], "guardrail_pass_through", np.where(out.eps_fair.isna(), "no_human_gap", np.where(out.peer_sd_mean < sd_lt, "no_human_variance", "")))
    return out
