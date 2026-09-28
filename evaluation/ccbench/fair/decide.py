"""Cluster-robust t sampling radius of a paired difference at a given error level."""

from __future__ import annotations

import numpy as np
from scipy import stats


def cluster_t_radius(d: np.ndarray, tasks: list[str], clusters: dict[str, str], d_eff: float) -> tuple[float, float, int]:
    d = np.asarray(d, float)
    N = len(d)
    Dm = d.mean()
    groups: dict[str, list[int]] = {}
    for j, t in enumerate(tasks):
        groups.setdefault(clusters.get(t, t), []).append(j)
    K = len(groups)
    if K < 2:
        return float("nan"), float("nan"), K
    ssq = sum(float((d[idx] - Dm).sum()) ** 2 for idx in groups.values())
    se = float(np.sqrt(K / (K - 1) * ssq) / N)
    tq = float(stats.t.ppf(1 - d_eff / 2, K - 1))
    return tq * se, se, K
