"""Loads the human review graph of a task: sections, claims, papers, typed relations, hard negatives
and claim tiers."""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field

import numpy as np

from ccbench import paths
from ccbench.config import prereg
from ccbench.gt import years as gt_years


@dataclass
class Claim:
    id: str
    text: str
    section_id: str | None
    section_type: str | None
    paragraph_id: str | None
    ref_nos: list[int]
    pmids: list[str]
    groundedness: float | None
    contradict: float | None
    weight: int
    tier: str


@dataclass
class Edge:
    src: str
    tgt: str
    type: str
    confidence: float | None
    direction: str | None
    cosine: float | None
    hard_negative: bool = False


@dataclass
class Section:
    id: str
    title: str
    level: int
    parent: str | None
    section_type: str | None


@dataclass
class GStar:
    task: str
    topic: str | None
    claims: list[Claim]
    edges: list[Edge]
    hard_negatives: list[Edge]
    sections: list[Section]
    refs: dict[int, dict]
    reference_weights: dict[int, int]
    cutoff: int | None
    w_req: float
    meta: dict = field(default_factory=dict)

    @property
    def pmids(self) -> set[str]:
        return {str(r["pmid"]) for r in self.refs.values() if r.get("pmid")}

    def pmid_weights(self) -> dict[str, int]:
        w: dict[str, int] = {}
        for no, r in self.refs.items():
            p = r.get("pmid")
            if p:
                w[str(p)] = w.get(str(p), 0) + int(self.reference_weights.get(no, 1))
        return w

    def reachable_pmids(self, cutoff: int | None = None) -> set[str]:
        c = cutoff if cutoff is not None else self.cutoff
        return {p for p in self.pmids if c is None or gt_years.before_cutoff(p, c)}

    def claims_by_tier(self, tier: str) -> list[Claim]:
        return [c for c in self.claims if c.tier == tier]

    def section_titles(self, min_level: int = 1) -> list[str]:
        return [s.title for s in self.sections if s.level >= min_level and s.title and s.section_type != "abstract"]

    def section_of_claim(self) -> dict[str, str | None]:
        return {c.id: c.section_id for c in self.claims}

    def top_section(self, section_id: str | None) -> str | None:
        by_id = {s.id: s for s in self.sections}
        cur = by_id.get(section_id) if section_id else None
        while cur is not None and cur.parent and cur.parent in by_id:
            cur = by_id[cur.parent]
        return cur.id if cur else section_id


def _grounding_path(task: str):
    return paths.resolve_opt(f"data/reference_corpus/train_2000/grounding/{task}.json")


def _load_sections(task: str) -> list[Section]:
    p = paths.resolve_opt(f"biolit-bench/results_2000/benchmark_claims/{task}/review_structure_llm.json")
    if p is None:
        return []
    with open(p) as f:
        rs = json.load(f)
    out = []
    for s in rs.get("sections", []):
        out.append(
            Section(
                id=s["section_id"],
                title=(s.get("title") or "").strip(),
                level=int(s.get("heading_level") or 1),
                parent=s.get("parent_section_id"),
                section_type=s.get("section_type"),
            )
        )
    return out


@functools.lru_cache(maxsize=4096)
def load(task: str, cutoff: int | None = None) -> GStar:
    cfg = prereg()["tiers"]
    with open(paths.gt_graph_path(task)) as f:
        g = json.load(f)
    refs: dict[int, dict] = {}
    rp = paths.gt_refs_path(task)
    if rp.exists():
        with open(rp) as f:
            for k, v in json.load(f).items():
                try:
                    refs[int(k)] = v
                except ValueError:
                    continue
    rw = {int(k): int(v) for k, v in (g.get("reference_weights") or {}).items() if str(k).lstrip("-").isdigit()}
    wvals = np.array(list(rw.values()) or [1])
    w_req = float(np.quantile(wvals, cfg["w_req_quantile"]))

    contra: dict[str, float] = {}
    gp = _grounding_path(task)
    if gp is not None:
        with open(gp) as f:
            for cid, v in json.load(f).items():
                if isinstance(v, dict) and v.get("contradict") is not None:
                    contra[cid] = float(v["contradict"])

    claims: list[Claim] = []
    for c in g.get("claims", []):
        nos = [int(x) for x in (c.get("reference_ids") or []) if str(x).lstrip("-").isdigit()]
        pm = [str(refs[n]["pmid"]) for n in nos if n in refs and refs[n].get("pmid")]
        gr = c.get("groundedness")
        cd = contra.get(c["claim_id"])
        wt = int(c.get("max_ref_weight") or 0)
        entailed = gr is not None and float(gr) >= 0.5
        if gr is None:
            tier = "unscorable"
        elif cd is not None and float(cd) >= 0.5 and not entailed:
            tier = "forbidden"
        elif entailed and wt >= w_req:
            tier = "required"
        else:
            tier = "admissible"
        claims.append(
            Claim(
                id=c["claim_id"],
                text=c.get("sentence") or "",
                section_id=c.get("section_id"),
                section_type=c.get("section_type"),
                paragraph_id=c.get("paragraph_id"),
                ref_nos=nos,
                pmids=pm,
                groundedness=float(gr) if gr is not None else None,
                contradict=cd,
                weight=wt,
                tier=tier,
            )
        )

    def _edge(e: dict, hn: bool) -> Edge:
        return Edge(
            src=e["source_id"],
            tgt=e["target_id"],
            type=e.get("relation_type") or e.get("rel") or "",
            confidence=e.get("rel_confidence"),
            direction=e.get("direction"),
            cosine=e.get("cosine"),
            hard_negative=hn,
        )

    edges = [_edge(e, False) for e in g.get("review_level_edges", [])]
    hard = [_edge(e, True) for e in g.get("hard_negatives", [])]
    sections = _load_sections(task)

    topic = None
    try:
        from ccbench.gt import peers

        topic = peers.topic_of(task)
    except Exception:
        pass
    if cutoff is None:
        tp = paths.taskspec_path(task)
        with open(tp) as f:
            cutoff = int(json.load(f)["publication_cutoff"])
    return GStar(
        task=task,
        topic=topic,
        claims=claims,
        edges=edges,
        hard_negatives=hard,
        sections=sections,
        refs=refs,
        reference_weights=rw,
        cutoff=cutoff,
        w_req=w_req,
        meta={"n_claims": g.get("n_claims"), "has_sections": bool(sections), "has_contradict": bool(contra)},
    )


def exists(task: str) -> bool:
    return paths.gt_graph_path(task).exists()
