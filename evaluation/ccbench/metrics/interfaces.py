"""Interface distances on the stage exits (evidence sets, claim graphs, outlines, reports)."""

from __future__ import annotations

import math

import numpy as np
import ot

from ccbench.model import Rollout
from ccbench.readouts.outline import _norm, match_titles


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def weighted_jaccard(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    num = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    den = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return num / den if den else 0.0


def d1_evidence(a: Rollout, b: Rollout) -> float:
    if a.evidence_ids and b.evidence_ids:
        return 1 - jaccard(a.evidence_ids, b.evidence_ids)
    return 1 - jaccard(a.papers, b.papers)


def graph_arrays(g: dict, types: list[str] | None = None) -> tuple[list[dict], np.ndarray]:
    nodes = g.get("nodes") or []
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    for e in g.get("edges") or []:
        i, j = idx.get(e.get("src")), idx.get(e.get("tgt"))
        if i is None or j is None:
            continue
        A[i, j] = A[j, i] = 1.0
    return nodes, A


def d2_fgw(ga: dict | None, gb: dict | None, lam: float = 0.5, node_cost: str = "pmid", emb_a: np.ndarray | None = None, emb_b: np.ndarray | None = None, max_nodes: int = 400) -> float | None:
    if not ga or not gb or not ga.get("nodes") or not gb.get("nodes"):
        return None
    na, A = graph_arrays(ga)
    nb, B = graph_arrays(gb)
    na, nb = na[:max_nodes], nb[:max_nodes]
    A, B = A[: len(na), : len(na)], B[: len(nb), : len(nb)]
    if emb_a is not None and emb_b is not None:
        M = 1 - np.clip(emb_a[: len(na)] @ emb_b[: len(nb)].T, -1, 1)
        M = M / 2.0
    else:
        M = np.zeros((len(na), len(nb)))
        for i, x in enumerate(na):
            for j, y in enumerate(nb):
                M[i, j] = 1 - jaccard(x.get("pmids") or [], y.get("pmids") or [])
    p = np.full(len(na), 1 / len(na))
    q = np.full(len(nb), 1 / len(nb))
    try:
        val = ot.gromov.fused_gromov_wasserstein2(M, A, B, p, q, loss_fun="square_loss", alpha=lam, symmetric=True, max_iter=200)
    except Exception:
        return None
    return float(min(1.0, max(0.0, val)))


def d3_outline(a: Rollout, b: Rollout, threshold: int = 80) -> float | None:
    ta = [t for t in (_norm(o.title) for o in a.outline) if t]
    tb = [t for t in (_norm(o.title) for o in b.outline) if t]
    if not ta or not tb:
        return None
    n, pairs = match_titles(ta, tb, threshold)
    match = n / max(len(ta), len(tb))
    size = min(1.0, abs(math.log(len(ta) / len(tb))))
    order = 0.0
    if len(pairs) >= 2:
        pa = [ta.index(s) for _, s, _ in pairs]
        pb = [tb.index(h) for h, _, _ in pairs]
        conc = disc = 0
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                s1 = (pa[i] - pa[j]) * (pb[i] - pb[j])
                if s1 > 0:
                    conc += 1
                elif s1 < 0:
                    disc += 1
        order = disc / (conc + disc) if (conc + disc) else 0.0
    return float(min(1.0, 0.5 * (1 - match) + 0.2 * size + 0.3 * order))


def citation_profile(ro: Rollout) -> dict[str, float]:
    prof: dict[str, float] = {}
    if not ro.report:
        return prof
    for s in ro.report.sentences:
        for p in s.cites:
            prof[p] = prof.get(p, 0) + 1
    tot = sum(prof.values()) or 1
    return {k: v / tot for k, v in prof.items()}


def d4_report(a: Rollout, b: Rollout) -> float | None:
    if not a.report or not b.report:
        return None
    ja = 1 - jaccard(a.report.bibliography, b.report.bibliography)
    pa, pb = citation_profile(a), citation_profile(b)
    wj = 1 - weighted_jaccard(pa, pb)
    la, lb = max(1, a.report.words), max(1, b.report.words)
    size = min(1.0, abs(math.log(la / lb)))
    return float(min(1.0, 0.4 * ja + 0.4 * wj + 0.2 * size))
