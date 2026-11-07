"""Builds the empirical branch kernel of each logged system on a coarse history state."""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ccbench import labels
from ccbench.config import prereg
from ccbench.model import Label, Rollout

TERMINAL = ("terminal", 0, 0, 0)
START_STAGE = "acquisition"


def _bin(x: int, edges: list[int]) -> int:
    return bisect.bisect_right(edges, x) - 1


@dataclass
class Kernel:
    name: str
    states: list[tuple]
    index: dict[tuple, int]
    labels: list[tuple]
    label_objs: list[Label]
    trans: dict[int, list[tuple[int, int, float]]] = field(default_factory=dict)
    x0: int = 0
    n_rollouts: int = 0
    n_transitions: int = 0

    def support(self, x: int) -> list[tuple[int, int, float]]:
        return self.trans.get(x, [(self.index_label_terminal, self.index[TERMINAL], 1.0)])

    @property
    def index_label_terminal(self) -> int:
        return len(self.labels) - 1


def abstract_states(ro: Rollout, bins: dict) -> list[tuple[tuple, Label]]:
    n_search = n_llm = n_papers = 0
    seen: set[str] = set()
    seq = []
    stage = START_STAGE
    for e in ro.events:
        lab = labels.coarsen(e.label)
        x = (stage, _bin(n_search, bins["n_search"]), _bin(n_llm, bins["n_llm"]), _bin(n_papers, bins["n_papers"]))
        seq.append((x, lab))
        if e.kind == "search":
            n_search += 1
        elif e.kind in ("llm", "failure", "retry", "validate"):
            n_llm += 1
        for p in e.returned_ids:
            if p not in seen:
                seen.add(p)
                n_papers += 1
        stage = lab.stage
    return seq


def build_kernel(name: str, rollouts: list[Rollout]) -> Kernel:
    bins = prereg()["controllability"]["state_bins"]
    counts: dict[tuple, dict[tuple[tuple, tuple], int]] = defaultdict(lambda: defaultdict(int))
    states: dict[tuple, int] = {}
    labs: dict[tuple, int] = {}
    lab_objs: dict[tuple, Label] = {}
    term_label = Label("terminal", "failure", "none", "0", "aborted")

    def sid(x):
        if x not in states:
            states[x] = len(states)
        return states[x]

    def lid(l: Label):
        t = l.as_tuple()
        if t not in labs:
            labs[t] = len(labs)
            lab_objs[t] = l
        return labs[t]

    x0 = (START_STAGE, 0, 0, 0)
    sid(x0)
    n_tr = 0
    n_ro = 0
    for ro in rollouts:
        seq = abstract_states(ro, bins)
        if not seq and ro.is_bot:
            continue
        n_ro += 1
        for i, (x, lab) in enumerate(seq):
            nxt = seq[i + 1][0] if i + 1 < len(seq) else TERMINAL
            counts[x][(lab.as_tuple(), nxt)] += 1
            sid(x)
            sid(nxt)
            lid(lab)
            n_tr += 1
        if not seq:
            counts[x0][(term_label.as_tuple(), TERMINAL)] += 1
            n_tr += 1
    sid(TERMINAL)
    lid(term_label)
    state_list = [None] * len(states)
    for s, i in states.items():
        state_list[i] = s
    lab_list = [None] * len(labs)
    lobj = [None] * len(labs)
    for t, i in labs.items():
        lab_list[i] = t
        lobj[i] = lab_objs[t]
    trans: dict[int, list[tuple[int, int, float]]] = {}
    for x, d in counts.items():
        tot = sum(d.values())
        trans[states[x]] = [(labs[l], states[nx], c / tot) for (l, nx), c in d.items()]
    return Kernel(name=name, states=state_list, index=states, labels=lab_list, label_objs=lobj, trans=trans, x0=states[x0], n_rollouts=n_ro, n_transitions=n_tr)


def d_Z_states(x: tuple, y: tuple, bins: dict) -> float:
    if x == y:
        return 0.0
    if x[0] != y[0]:
        return 1.0
    scale = [len(bins["n_search"]) - 1, len(bins["n_llm"]) - 1, len(bins["n_papers"]) - 1]
    return float(np.mean([abs(x[i + 1] - y[i + 1]) / max(1, scale[i]) for i in range(3)]))
