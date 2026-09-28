"""Estimators of the stage and composition bounds: same-entry discrepancies, module envelopes (slope
and slack) and the composed multi-stage bound."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ccbench.metrics import interfaces as M
from ccbench.model import Rollout

def stage_exit_distance(k: int, a: Rollout, b: Rollout) -> float | None:
    if k == 1:
        return M.d1_evidence(a, b) if (a.papers or b.papers) else None
    if k == 2:
        return M.d2_fgw(a.graph, b.graph)
    if k == 3:
        return M.d3_outline(a, b)
    if k == 4:
        return M.d4_report(a, b)
    raise ValueError(k)


@dataclass
class Envelope:
    L: float
    xi: float
    quantile: float
    n: int
    loo_violation_rate: float
    points: list[tuple[float, float]]


def fit_envelope(x: np.ndarray, y: np.ndarray, q: float) -> tuple[float, float]:
    import statsmodels.api as sm

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 4 or np.allclose(x, x[0]):
        return 0.0, float(np.quantile(y, q))
    X = sm.add_constant(x)
    try:
        res = sm.QuantReg(y, X).fit(q=q, max_iter=2000)
        L = float(res.params[1])
    except Exception:
        L = 0.0
    L = max(0.0, L)
    xi = float(np.quantile(y - L * x, q))
    return L, max(0.0, xi)


def envelope_with_loo(points: list[tuple[float, float]], q: float) -> Envelope:
    pts = [(float(a), float(b)) for a, b in points if a is not None and b is not None and not (np.isnan(a) or np.isnan(b))]
    if len(pts) < 4:
        return Envelope(0.0, float("nan"), q, len(pts), float("nan"), pts)
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    L, xi = fit_envelope(x, y, q)
    viol = 0
    for i in range(len(pts)):
        m = np.ones(len(pts), bool)
        m[i] = False
        Li, xii = fit_envelope(x[m], y[m], q)
        viol += int(y[i] > Li * x[i] + xii + 1e-9)
    return Envelope(L, xi, q, len(pts), viol / len(pts), pts)


def ratio_lipschitz(points: list[tuple[float, float]], q: float) -> dict:
    r = [b / a for a, b in points if a is not None and b is not None and a > 1e-6]
    if not r:
        return {"q": float("nan"), "max": float("nan"), "n": 0}
    return {"q": float(np.quantile(r, q)), "max": float(max(r)), "median": float(np.median(r)), "n": len(r)}


def composed_bound(eps0: float, eps: dict[int, float], xi: dict[int, float], L: dict[int, float], m: int) -> float:
    tot = eps0 * float(np.prod([L.get(j, 1.0) for j in range(1, m + 1)]))
    for k in range(1, m + 1):
        tot += (eps.get(k, 0.0) + xi.get(k, 0.0)) * float(np.prod([L.get(j, 1.0) for j in range(k + 1, m + 1)]))
    return float(tot)
