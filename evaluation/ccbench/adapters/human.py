"""Renders a human review as a rollout through the same sentence splitting and citation resolution
as system outputs, so that peer reviews pass through the same readouts."""

from __future__ import annotations

import json
import re

from ccbench import paths
from ccbench.adapters.common import numeric_resolver, section_sentences
from ccbench.gt import graphs
from ccbench.ingest import gold
from ccbench.model import Context, OutlineNode, Report, Rollout, bot_rollout

CITE = re.compile(r"\[([\d,\s;–-]+)\]")


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", CITE.sub("", text or "")).strip().lower()


def adapt(task: str, target_task: str | None = None) -> Rollout:
    if not graphs.exists(task):
        return bot_rollout(f"human:{task}", target_task or task, "H", Context(task=target_task or task, topic="", cutoff=9999), "no_graph")
    g = graphs.load(task)
    ctx = gold.context_for(target_task or task)
    rs_path = paths.resolve_opt(f"biolit-bench/results_2000/benchmark_claims/{task}/review_structure_llm.json")
    refmap = {str(no): [str(r["pmid"])] for no, r in g.refs.items() if r.get("pmid")}
    resolve = numeric_resolver(refmap)
    claim_pmids: dict[str, list[str]] = {}
    for c in g.claims:
        k = _key(c.text)
        if k and c.pmids:
            claim_pmids.setdefault(k, list(c.pmids))
            claim_pmids.setdefault(k[:60], list(c.pmids))
    outline: list[OutlineNode] = []
    sections: list[dict] = []
    bib: dict[str, None] = {}
    if rs_path is not None:
        with open(rs_path) as f:
            rs = json.load(f)
        meta = {s["section_id"]: s for s in rs.get("sections", [])}
        for s in rs.get("sections", []):
            if s.get("section_type") != "abstract":
                outline.append(OutlineNode(id=s["section_id"], title=(s.get("title") or "").strip(), level=int(s.get("heading_level") or 1), parent=s.get("parent_section_id")))
        texts: dict[str, list[str]] = {}
        for para in rs.get("paragraphs", []):
            texts.setdefault(para.get("section_id"), []).append(para.get("text") or "")
        for sid, paras in texts.items():
            m = meta.get(sid) or {}
            sections.append({"id": sid, "title": (m.get("title") or "").strip(), "level": int(m.get("heading_level") or 1), "text": "\n\n".join(paras)})
    sentences = []
    words = marks = unresolved = 0
    for sec in sections:
        sents, w, mk, un = section_sentences(sec["id"], sec["text"], CITE, resolve, bib)
        for s in sents:
            k = _key(s.text)
            extra = claim_pmids.get(k) or claim_pmids.get(k[:60]) or []
            for p in extra:
                if p not in s.cites:
                    s.cites.append(p)
                bib.setdefault(p, None)
        sentences += sents
        words += w
        marks += mk
        unresolved += un
    report = Report(sections=sections, sentences=sentences, bibliography=list(bib), words=words, unresolved_citations=unresolved, total_citation_marks=marks)
    weights = g.pmid_weights()
    papers = sorted(g.pmids, key=lambda p: -weights.get(p, 0))
    return Rollout(system=f"human:{task}", task=ctx.task, panel="H", context=ctx, status="ok", mode="human", papers=papers, outline=outline, report=report, graph={"nodes": [{"id": c.id, "text": c.text, "pmids": c.pmids, "type": c.tier} for c in g.claims], "edges": [{"src": e.src, "tgt": e.tgt, "type": e.type} for e in g.edges], "role": "human"}, resources={}, meta={"source_review": task, "has_body": rs_path is not None})
