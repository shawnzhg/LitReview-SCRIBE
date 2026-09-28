"""The pairwise decision rule: paired contrast on the common tasks with the scorer's cluster-robust
radius at the Bonferroni level, certified when the gap exceeds the radius plus the nuisance
radius."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def cluster_t_radius(d, tasks, clusters, d_eff):
    C.bootstrap_env()
    from ccbench.fair.decide import cluster_t_radius as ccb_cluster_t_radius
    return ccb_cluster_t_radius(d, tasks, clusters, d_eff)


def table_pairs(n_systems: int) -> int:
    return max(1, n_systems * (n_systems - 1) // 2)


def drop_failed(s: pd.Series, failed: set) -> pd.Series:
    s = s.astype(float).copy()
    idx = [t for t in failed if t in s.index]
    s.loc[idx] = np.nan
    return s


def contrast(a: pd.Series, b: pd.Series, clusters: dict, d_eff: float, eps: float = 0.0,
             min_tasks: int | None = None) -> dict:
    min_tasks = C.REGISTERED["min_tasks"] if min_tasks is None else min_tasks
    a = a.astype(float)
    b = b.astype(float)
    ok = a.notna() & b.notna()
    va, vb = a[ok].values, b[ok].values
    n_missing_a, n_missing_b = int((a.isna() & b.notna()).sum()), int((b.isna() & a.notna()).sum())
    tasks = [t for t, k in zip(a.index, ok.values) if k]
    n = len(tasks)
    base = {"n": n, "n_missing_a": n_missing_a, "n_missing_b": n_missing_b, "eps": eps}
    if n < min_tasks:
        return {**base, "K": 0, "ours": np.nan, "opp": np.nan, "diff": np.nan, "se": np.nan, "q": np.nan, "t": np.nan,
                "certified": False, "sampling_decided": False, "winner": None, "status": "too_few_tasks"}
    d = va - vb
    q, se, K = cluster_t_radius(d, tasks, clusters, d_eff)
    diff = float(d.mean())
    if np.isnan(q):
        return {**base, "K": K, "ours": float(va.mean()), "opp": float(vb.mean()), "diff": diff, "se": np.nan,
                "q": np.nan, "t": np.nan, "p": np.nan, "certified": False, "sampling_decided": False,
                "winner": None, "mde": np.nan, "status": "too_few_clusters"}
    if np.allclose(va, vb, rtol=0, atol=0):
        return {**base, "K": K, "ours": float(va.mean()), "opp": float(vb.mean()), "diff": 0.0, "se": 0.0,
                "q": q, "t": np.nan, "p": np.nan, "certified": False, "sampling_decided": False,
                "winner": None, "mde": np.nan, "status": "tie_identical"}
    eps_ok = not (eps is None or (isinstance(eps, float) and np.isnan(eps)))
    samp = bool(abs(diff) > q)
    cert = bool(eps_ok and abs(diff) > q + eps)
    tval = float(diff / se) if (se and se > 0) else np.nan
    pval = float(2 * (1 - stats.t.cdf(abs(tval), K - 1))) if (tval == tval and K > 1) else np.nan
    return {**base, "K": K, "ours": float(va.mean()), "opp": float(vb.mean()), "diff": diff, "se": se, "q": q,
            "t": tval, "p": pval, "certified": cert, "sampling_decided": samp,
            "winner": ("a" if diff > 0 else "b") if (cert or samp) else None,
            "mde": (q + eps) if eps_ok else np.nan,
            "status": "decided" if cert else ("undecided" if eps_ok else "descriptive")}
