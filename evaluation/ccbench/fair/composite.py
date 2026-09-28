"""Certified rank intervals over the transitive closure of the certified relations."""

from __future__ import annotations

import itertools

import numpy as np


def transitive_closure(names: list[str], rel: set[tuple[str, str]]) -> tuple[np.ndarray, bool]:
    ix = {s: i for i, s in enumerate(names)}
    R = np.zeros((len(names), len(names)), bool)
    for a, b in rel:
        R[ix[a], ix[b]] = True
    for k in range(len(names)):
        for i in range(len(names)):
            if R[i, k]:
                R[i] |= R[k]
    return R, bool(any(R[i, i] for i in range(len(names))))


def rank_intervals(names: list[str], rel: set[tuple[str, str]]) -> tuple[dict, bool, bool]:
    R, cyclic = transitive_closure(names, rel)
    ix = {s: i for i, s in enumerate(names)}
    iv = {s: (1 + int(R[:, ix[s]].sum()), len(names) - int(R[ix[s], :].sum())) for s in names}
    naive = {s: (1 + sum(1 for x, y in rel if y == s),
                 len(names) - sum(1 for x, y in rel if x == s)) for s in names}
    return iv, not cyclic, naive == iv


def tierable(names: list[str], rel: set[tuple[str, str]]) -> bool:
    R, _ = transitive_closure(names, rel)
    ix = {s: i for i, s in enumerate(names)}
    un = lambda x, y: not R[ix[x], ix[y]] and not R[ix[y], ix[x]]
    for a, b, c in itertools.permutations(names, 3):
        if un(a, b) and un(b, c) and not un(a, c):
            return False
    return True
