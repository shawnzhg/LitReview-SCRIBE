"""The controllability operator on branch kernels with exact optimal transport, its fixed point, and
the contraction and pseudometric checks."""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

import numpy as np
import ot

from ccbench.config import prereg
from ccbench.distance.kernel import Kernel, d_Z_states
from ccbench.labels import d_L


@dataclass
class Union:

    kernels: list[Kernel]
    offset: list[int]
    n: int
    states: list[tuple]
    sys_of: np.ndarray
    DZ: np.ndarray
    DL: np.ndarray
    lab_gid: list[dict[int, int]]
    supports: list[tuple[np.ndarray, np.ndarray, np.ndarray]]

    def block(self, d: np.ndarray, a: int, b: int) -> np.ndarray:
        ra = slice(self.offset[a], self.offset[a] + len(self.kernels[a].states))
        rb = slice(self.offset[b], self.offset[b] + len(self.kernels[b].states))
        return d[ra, rb]

    def gid(self, k: int, local: int) -> int:
        return self.offset[k] + local


def make_union(kernels: list[Kernel]) -> Union:
    cfg = prereg()["controllability"]
    bins = cfg["state_bins"]
    offset, n = [], 0
    for k in kernels:
        offset.append(n)
        n += len(k.states)
    states = [s for k in kernels for s in k.states]
    sys_of = np.array([i for i, k in enumerate(kernels) for _ in k.states])
    DZ = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            DZ[i, j] = DZ[j, i] = d_Z_states(states[i], states[j], bins)
    glabels: dict[tuple, int] = {}
    gobjs = []
    lab_gid = []
    for k in kernels:
        m = {}
        for li, t in enumerate(k.labels):
            if t not in glabels:
                glabels[t] = len(glabels)
                gobjs.append(k.label_objs[li])
            m[li] = glabels[t]
        lab_gid.append(m)
    L = len(glabels)
    DL = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            DL[i, j] = DL[j, i] = d_L(gobjs[i], gobjs[j], cfg["label_weights"])
    supports = []
    for ki, k in enumerate(kernels):
        for xi in range(len(k.states)):
            sup = k.support(xi)
            supports.append((np.array([lab_gid[ki][l] for l, _, _ in sup]), np.array([offset[ki] + nx for _, nx, _ in sup]), np.array([p for _, _, p in sup])))
    return Union(kernels, offset, n, states, sys_of, DZ, DL, lab_gid, supports)


def apply_T(U: Union, d: np.ndarray, gamma: float, beta: float, pairs: list[tuple[int, int]] | None = None) -> np.ndarray:
    n = U.n
    out = np.array(d, copy=True)
    it = pairs if pairs is not None else [(i, j) for i in range(n) for j in range(i, n)]
    for i, j in it:
        la, xa, pa = U.supports[i]
        lb, xb, pb = U.supports[j]
        C = beta * U.DL[np.ix_(la, lb)] + (1 - beta) * d[np.ix_(xa, xb)]
        if len(pa) == 1 or len(pb) == 1:
            w = float((pa[:, None] * pb[None, :] * C).sum())
        else:
            w = float(ot.emd2(pa, pb, C))
        v = (1 - gamma) * U.DZ[i, j] + gamma * w
        out[i, j] = out[j, i] = v
    return out


def fixed_point(U: Union, d0: np.ndarray | None = None, gamma: float | None = None, beta: float | None = None, max_iter: int | None = None, tol: float | None = None, log: bool = False) -> tuple[np.ndarray, dict]:
    cfg = prereg()["controllability"]
    gamma = cfg["gamma"] if gamma is None else gamma
    beta = cfg["beta"] if beta is None else beta
    max_iter = cfg["max_iter"] if max_iter is None else max_iter
    tol = cfg["tol"] if tol is None else tol
    d = np.zeros((U.n, U.n)) if d0 is None else np.array(d0, dtype=float)
    hist = []
    t0 = time.time()
    prev_delta = None
    for it in range(max_iter):
        d1 = apply_T(U, d, gamma, beta)
        delta = float(np.max(np.abs(d1 - d)))
        ratio = (delta / prev_delta) if prev_delta else None
        hist.append({"iter": it, "sup_change": delta, "ratio": ratio})
        if log:
            print(f"  iter {it:3d} sup|Td-d|={delta:.3e} ratio={ratio if ratio is None else round(ratio,4)} ({time.time()-t0:.1f}s)", flush=True)
        d = d1
        prev_delta = delta
        if delta < tol:
            break
    info = {"iterations": len(hist), "converged": bool(hist and hist[-1]["sup_change"] < tol), "history": hist, "gamma": gamma, "beta": beta, "modulus_bound": gamma * (1 - beta), "seconds": time.time() - t0, "n_states": U.n}
    return d, info


def contraction_check(U: Union, gamma: float, beta: float, n_trials: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_trials):
        d1 = rng.random((U.n, U.n))
        d1 = (d1 + d1.T) / 2
        d2 = rng.random((U.n, U.n))
        d2 = (d2 + d2.T) / 2
        num = np.max(np.abs(apply_T(U, d1, gamma, beta) - apply_T(U, d2, gamma, beta)))
        den = np.max(np.abs(d1 - d2))
        ratios.append(float(num / den) if den > 0 else 0.0)
    return {"ratios": ratios, "max_ratio": max(ratios), "bound": gamma * (1 - beta), "holds": max(ratios) <= gamma * (1 - beta) + 1e-9}


def pseudometric_check(d: np.ndarray, n_samples: int = 20000, seed: int = 0) -> dict:
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    sym = float(np.max(np.abs(d - d.T)))
    nonneg = float(d.min())
    diag = float(np.max(np.abs(np.diag(d))))
    i = rng.integers(0, n, n_samples)
    j = rng.integers(0, n, n_samples)
    k = rng.integers(0, n, n_samples)
    viol = d[i, k] - d[i, j] - d[j, k]
    return {"symmetry_max_abs": sym, "min_value": nonneg, "diag_max_abs": diag, "triangle_max_violation": float(viol.max()), "triangle_violations": int((viol > 1e-9).sum()), "triples": n_samples}


def whole_system(U: Union, d: np.ndarray) -> np.ndarray:
    m = len(U.kernels)
    C = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            C[a, b] = d[U.gid(a, U.kernels[a].x0), U.gid(b, U.kernels[b].x0)]
    return C


def sup_over_visited(U: Union, d: np.ndarray) -> np.ndarray:
    m = len(U.kernels)
    C = np.zeros((m, m))
    for a, b in itertools.product(range(m), range(m)):
        C[a, b] = float(U.block(d, a, b).max())
    return C
