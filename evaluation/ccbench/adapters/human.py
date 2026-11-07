"""Renders a human review as a rollout: its paragraphs give the body and sections, every claim appears
with its exact text and the pmids of its references, and its reference list is the retrieval exit."""

from __future__ import annotations

import re

from ccbench import paths
from ccbench.adapters.common import split_sentences, word_count
from ccbench.gt import graphs
from ccbench.ingest import gold
from ccbench.model import Context, OutlineNode, Report, Rollout, Sentence, bot_rollout


def adapt(task: str, target_task: str | None = None) -> Rollout:
    if not graphs.exists(task):
        return bot_rollout(f"human:{task}", target_task or task, "H", Context(task=target_task or task, topic="", cutoff=9999), "no_graph")
    g = graphs.load(task)
    ctx = gold.context_for(target_task or task)
    rs_path = paths.resolve_opt(f"biolit-bench/results_2000/benchmark_claims/{task}/review_structure_llm.json")
    if rs_path is None:
        raise paths.MissingCompanionError(task, [f"biolit-bench/results_2000/benchmark_claims/{task}/review_structure_llm.json"])
    rs = paths.read_json(rs_path)
    claim_by_id = {c.id: c for c in g.claims}
    sentences: list[Sentence] = []
    sections: list[dict] = []
    outline: list[OutlineNode] = []
    words = 0
    marks = 0
    sec_title = {s["section_id"]: s for s in rs.get("sections", [])}
    for s in rs.get("sections", []):
        if s.get("section_type") == "abstract":
            continue
        outline.append(OutlineNode(id=s["section_id"], title=(s.get("title") or "").strip(), level=int(s.get("heading_level") or 1), parent=s.get("parent_section_id")))
    placed: set[str] = set()
    for pk, para in enumerate(rs.get("paragraphs", [])):
        sid = para.get("section_id")
        pid = para.get("paragraph_id") or f"p{pk}"
        text = para.get("text") or ""
        words += word_count(text)
        if not any(x["id"] == sid for x in sections):
            sections.append({"id": sid, "title": (sec_title.get(sid) or {}).get("title") or "", "level": int((sec_title.get(sid) or {}).get("heading_level") or 1), "text": text})
        else:
            for x in sections:
                if x["id"] == sid:
                    x["text"] = (x["text"] + "\n\n" + text).strip()
        cited = [claim_by_id[c] for c in (para.get("claim_ids") or []) if c in claim_by_id]
        j = 0
        claim_keys = set()
        for c in cited:
            key = re.sub(r"\s+", " ", c.text).strip()
            claim_keys.add(key[:60])
            sentences.append(Sentence(sid=f"{sid}/{pid}#{j}", section=sid, text=key, cites=list(c.pmids)))
            marks += len(c.ref_nos)
            placed.add(c.id)
            j += 1
        for sent in split_sentences(text):
            key = re.sub(r"\s+", " ", sent).strip()
            if key[:60] in claim_keys or any(k in key or key[:60] in k for k in claim_keys):
                continue
            sentences.append(Sentence(sid=f"{sid}/{pid}#{j}", section=sid, text=key, cites=[]))
            j += 1
    for c in g.claims:
        if c.id not in placed and c.text:
            sid = c.section_id or "s0"
            sentences.append(Sentence(sid=f"{sid}#x{len(sentences)}", section=sid, text=re.sub(r"\s+", " ", c.text).strip(), cites=list(c.pmids)))
            marks += len(c.ref_nos)
    bib: dict[str, None] = {}
    for s in sentences:
        for p in s.cites:
            bib.setdefault(p, None)
    report = Report(sections=sections, sentences=sentences, bibliography=list(bib), words=words, unresolved_citations=0, total_citation_marks=marks)
    weights = g.pmid_weights()
    papers = sorted(g.pmids, key=lambda p: -weights.get(p, 0))
    return Rollout(system=f"human:{task}", task=ctx.task, panel="H", context=ctx, status="ok", mode="human", papers=papers, outline=outline, report=report, graph={"nodes": [{"id": c.id, "text": c.text, "pmids": c.pmids, "type": c.tier} for c in g.claims], "edges": [{"src": e.src, "tgt": e.tgt, "type": e.type} for e in g.edges], "role": "human"}, resources={}, meta={"source_review": task, "has_body": True})
