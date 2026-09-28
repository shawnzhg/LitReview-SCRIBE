"""Builds the scoring units of a human review graph, with strata comparable across tasks."""

from __future__ import annotations

import functools
import json
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ccbench import paths
from ccbench.config import prereg
from ccbench.gt import graphs
from ccbench.gt import years as gt_years


@dataclass
class Units:
    task: str
    hubs: set[str]
    tail: set[str]
    recent: set[str]
    classic: set[str]
    communities: list[set[str]]
    subfields: dict[str, set[str]]
    claims_by_tier: dict[str, list[str]]
    grounded_hi: list[str]
    grounded_lo: list[str]
    central: list[str]
    multi_source: dict[str, set[str]]
    reused_papers: dict[str, set[str]]
    edges_by_type: dict[str, list[tuple[str, str]]]
    edges_by_distance: dict[str, list[tuple[str, str]]]
    hard_negatives: list[tuple[str, str]]
    motifs: dict[str, list[tuple[str, ...]]]
    top_sections: list[dict]
    section_of_claim: dict[str, str | None]
    paragraph_of_claim: dict[str, str | None]
    meta: dict = field(default_factory=dict)


def _paragraphs(task: str) -> tuple[dict[str, dict], dict[str, str]]:
    p = paths.resolve_opt(f"biolit-bench/results_2000/benchmark_claims/{task}/review_structure_llm.json")
    if p is None:
        return {}, {}
    rs = json.load(open(p))
    paras = {x["paragraph_id"]: x for x in rs.get("paragraphs", [])}
    central: dict[str, str] = {}
    for pid, x in paras.items():
        cs = (x.get("central_sentence") or "").strip()
        if cs:
            central[pid] = cs
    return paras, central


@functools.lru_cache(maxsize=512)
def build(task: str) -> Units:
    cfg = prereg()["units"]
    g = graphs.load(task)
    w = g.pmid_weights()
    pm = sorted(g.pmids)
    if pm:
        ws = [w.get(p, 1) for p in pm]
        thr = float(np.quantile(ws, cfg["hub_weight_quantile"]))
        med = float(np.median(ws))
        hubs = {p for p in pm if w.get(p, 1) >= thr and w.get(p, 1) > med}
    else:
        hubs = set()
    tail = set(pm) - hubs
    cut = g.cutoff or 9999
    recent, classic = set(), set()
    for p in pm:
        y = gt_years.year_of(p)
        if y is None:
            continue
        (recent if y >= cut - cfg["recent_years_before_cutoff"] else classic).add(p)
    sf = gt_years.subfields()
    subfields: dict[str, set[str]] = defaultdict(set)
    for p in pm:
        if p in sf:
            subfields[sf[p]].add(p)
    G = nx.Graph()
    G.add_nodes_from(pm)
    for c in g.claims:
        ps = sorted(set(c.pmids))
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                G.add_edge(ps[i], ps[j])
    communities = [set(cc) for cc in nx.connected_components(G) if len(cc) >= cfg["community_min_size"]]
    by_tier: dict[str, list[str]] = defaultdict(list)
    for c in g.claims:
        by_tier[c.tier].append(c.id)
    ghi = [c.id for c in g.claims if c.groundedness is not None and c.groundedness >= 0.5]
    glo = [c.id for c in g.claims if c.groundedness is not None and c.groundedness < 0.5]
    paras, central_text = _paragraphs(task)
    central = []
    for c in g.claims:
        cs = central_text.get(c.paragraph_id or "")
        if cs and c.text and (c.text[:60] == cs[:60] or cs[:60] in c.text or c.text[:60] in cs):
            central.append(c.id)
    if not central:
        by_para: dict[str, list] = defaultdict(list)
        for c in g.claims:
            if c.paragraph_id:
                by_para[c.paragraph_id].append(c)
        for pid, cl in by_para.items():
            if len(cl) >= 2:
                central.append(max(cl, key=lambda c: (c.weight, -cl.index(c))).id)
    multi = {c.id: set(c.pmids) for c in g.claims if len(set(c.pmids)) >= cfg["multi_source_min_refs"]}
    reuse_cnt: dict[str, set[str]] = defaultdict(set)
    for c in g.claims:
        for p in set(c.pmids):
            reuse_cnt[p].add(c.id)
    reused = {p: s for p, s in reuse_cnt.items() if len(s) >= cfg["reused_paper_min_claims"]}
    top_of = {s.id: g.top_section(s.id) for s in g.sections}
    body = [s for s in g.sections if s.section_type != "abstract" and (s.level == 1 or s.parent is None)]
    body = [s for s in body if s.title.lower().strip() not in ("references", "bibliography", "acknowledgements")]
    sec_of_claim = {c.id: top_of.get(c.section_id) for c in g.claims}
    top_secs = []
    for k, s in enumerate(body):
        role = "introduction" if k == 0 else ("conclusion" if (k == len(body) - 1 and len(body) > 1) or s.section_type == "discussion_or_conclusion" else "body")
        cl = [c.id for c in g.claims if sec_of_claim.get(c.id) == s.id]
        ps = {p for c in g.claims if sec_of_claim.get(c.id) == s.id for p in c.pmids}
        top_secs.append({"id": s.id, "title": s.title, "role": role, "order": k, "claims": cl, "papers": sorted(ps)})
    order_of = {s["id"]: s["order"] for s in top_secs}
    by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_dist: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for e in g.edges:
        by_type[e.type].append((e.src, e.tgt))
        a, b = sec_of_claim.get(e.src), sec_of_claim.get(e.tgt)
        if a is None or b is None:
            d = "unknown"
        elif a == b:
            d = "within"
        elif abs(order_of.get(a, 0) - order_of.get(b, 0)) == 1:
            d = "adjacent"
        else:
            d = "distant"
        by_dist[d].append((e.src, e.tgt))
    hard = [(e.src, e.tgt) for e in g.hard_negatives]
    D = nx.DiGraph()
    for e in g.edges:
        if e.type in ("elaboration", "mechanism", "evidence"):
            D.add_edge(e.src, e.tgt, type=e.type)
    motifs: dict[str, list[tuple[str, ...]]] = {"chain": [], "mechanism_path": [], "contrast_condition": []}
    L = cfg["motif_chain_min_len"]
    for n0 in D.nodes:
        for path in _paths_from(D, n0, L):
            types = {D.edges[path[i], path[i + 1]]["type"] for i in range(len(path) - 1)}
            if types == {"elaboration"}:
                motifs["chain"].append(tuple(path))
            elif "mechanism" in types:
                motifs["mechanism_path"].append(tuple(path))
    contrast = {frozenset(p) for p in by_type.get("contrast", [])}
    condition = {frozenset(p) for p in by_type.get("condition", [])}
    for c1 in contrast:
        for c2 in condition:
            if c1 & c2:
                motifs["contrast_condition"].append(tuple(sorted(c1 | c2)))
    for k in motifs:
        motifs[k] = sorted(set(motifs[k]))[:200]
    return Units(task=task, hubs=hubs, tail=tail, recent=recent, classic=classic, communities=communities, subfields=dict(subfields), claims_by_tier=dict(by_tier), grounded_hi=ghi, grounded_lo=glo, central=central, multi_source=multi, reused_papers=reused, edges_by_type=dict(by_type), edges_by_distance=dict(by_dist), hard_negatives=hard, motifs=motifs, top_sections=top_secs, section_of_claim=sec_of_claim, paragraph_of_claim={c.id: c.paragraph_id for c in g.claims}, meta={"n_papers": len(pm), "n_claims": len(g.claims), "n_edges": len(g.edges), "n_top_sections": len(top_secs), "n_communities": len(communities), "n_central": len(central)})


def _paths_from(D: nx.DiGraph, start: str, length: int, max_paths: int = 50) -> list[list[str]]:
    out: list[list[str]] = []
    stack = [[start]]
    while stack and len(out) < max_paths:
        p = stack.pop()
        if len(p) == length:
            out.append(p)
            continue
        for nxt in D.successors(p[-1]):
            if nxt not in p:
                stack.append(p + [nxt])
    return out
