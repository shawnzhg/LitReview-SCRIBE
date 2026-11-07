"""Synthetic four-stage chains derived from a human graph with known retrieval, claim-retention and
section-order constants."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ccbench.gt import graphs, peers
from ccbench.ingest import gold
from ccbench.model import Context, OutlineNode, Report, Rollout, Sentence


@dataclass
class Params:
    rho: float = 0.8
    rho_d: float = 0.1
    kappa: float = 0.85
    pi: float = 0.1
    name: str = "S"


@dataclass
class Chain:
    task: str
    params: Params
    seed: int
    P: list[str] = field(default_factory=list)
    claims: list = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    rollout: Rollout | None = None


def _ctx(task: str, g: graphs.GStar) -> Context:
    try:
        return gold.context_for(task)
    except Exception:
        return Context(task=task, topic=g.topic or "", cutoff=g.cutoff or 9999)


def stage1_retrieval(g: graphs.GStar, p: Params, rng: np.random.Generator, task: str) -> list[str]:
    gold_p = sorted(g.pmids)
    keep = [x for x in gold_p if rng.random() < p.rho]
    distract = []
    if p.rho_d > 0:
        for pid in peers.peers(task)[:2]:
            gp = graphs.load(pid)
            distract += [x for x in sorted(gp.pmids - g.pmids) if rng.random() < p.rho_d]
    out = keep + distract
    rng.shuffle(out)
    return out


def stage2_synthesis(g: graphs.GStar, P: list[str], p: Params, rng: np.random.Generator) -> list:
    Pset = set(P)
    out = []
    for c in g.claims:
        if c.pmids and not set(c.pmids) <= Pset:
            continue
        if rng.random() < p.kappa:
            out.append(c)
    return out


def stage3_organisation(g: graphs.GStar, claims: list, p: Params, rng: np.random.Generator) -> list[str]:
    secs = [s.id for s in g.sections if s.section_type != "abstract"] or sorted({c.section_id for c in claims if c.section_id})
    alive = {c.section_id for c in claims}
    order = [s for s in secs if s in alive] or secs
    n_perm = int(round(p.pi * len(order)))
    idx = list(range(len(order)))
    for _ in range(n_perm):
        i, j = rng.integers(0, len(order), 2)
        idx[i], idx[j] = idx[j], idx[i]
    return [order[i] for i in idx]


def stage4_writing(g: graphs.GStar, sections: list[str], claims: list, P: list[str], p: Params, ctx: Context, name: str) -> Rollout:
    title = {s.id: s.title for s in g.sections}
    by_sec: dict[str, list] = {}
    for c in claims:
        by_sec.setdefault(c.section_id, []).append(c)
    sentences, secs, bib = [], [], {}
    words = 0
    marks = 0
    Pstar = sorted(g.pmids)
    for k, sid in enumerate(sections):
        secs.append({"id": sid, "title": title.get(sid, sid), "level": 1, "text": ""})
        for j, c in enumerate(by_sec.get(sid, [])):
            cites = list(c.pmids)
            sentences.append(Sentence(sid=f"{sid}#{j}", section=sid, text=c.text, cites=cites))
            words += len(c.text.split())
            marks += len(cites)
            for x in cites:
                bib.setdefault(x, None)
    outline = [OutlineNode(id=s, title=title.get(s, s), level=1) for s in sections]
    rep = Report(sections=secs, sentences=sentences, bibliography=list(bib), words=words, total_citation_marks=marks)
    graph = {"nodes": [{"id": c.id, "pmids": c.pmids, "text": c.text} for c in claims], "edges": [{"src": e.src, "tgt": e.tgt, "type": e.type} for e in g.edges if e.src in {c.id for c in claims} and e.tgt in {c.id for c in claims}]}
    return Rollout(system=name, task=ctx.task, panel="S", context=ctx, papers=list(P), outline=outline, report=rep, graph=graph, mode="synthetic")


def run_chain(task: str, p: Params, seed: int, entry: dict | None = None) -> Chain:
    g = graphs.load(task)
    rng = np.random.default_rng(seed)
    ctx = _ctx(task, g)
    P = entry["P"] if entry and "P" in entry else stage1_retrieval(g, p, rng, task)
    claims = entry["claims"] if entry and "claims" in entry else stage2_synthesis(g, P, p, rng)
    secs = entry["sections"] if entry and "sections" in entry else stage3_organisation(g, claims, p, rng)
    ro = stage4_writing(g, secs, claims, P, p, ctx, p.name)
    return Chain(task=task, params=p, seed=seed, P=P, claims=claims, sections=secs, rollout=ro)
