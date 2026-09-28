"""Builds the synthesis and planning views of a rollout and its length-matched report, so that each
readout is read at the window where it is observable."""

from __future__ import annotations

import re
from dataclasses import replace

from ccbench import paths
from ccbench.adapters.common import WORD_RE, split_sentences, word_count
from ccbench.gt import graphs
from ccbench.gt.units import Units
from ccbench.ingest import agent_runs
from ccbench.model import OutlineNode, Report, Rollout, Sentence
from ccbench.readouts import channel, subgraph

WINDOW_READOUT_PREFIXES = {
    "retrieval": ("ret_",),
    "synthesis": ("syn_", "rel_", "motif_", "form_link_density", "form_claims_per_paper", "wr_evidence_agreement"),
    "planning": ("org_",),
    "writing": ("",),
    "system": ("",),
    "system_trunc": ("",),
}


def readouts_observable_at(window: str, name: str) -> bool:
    if name == "completion":
        return True
    return any(name.startswith(p) for p in WINDOW_READOUT_PREFIXES[window])


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def synthesis_view(ro: Rollout, claims: list[dict], relations: list[dict]) -> tuple[Rollout, set[str]]:
    sents = []
    bib: dict[str, None] = {}
    for c in claims:
        t = _norm(c.get("text"))
        if not t:
            continue
        pm = [str(p) for p in (c.get("pmids") or [])]
        sents.append(Sentence(sid=str(c["id"]), section="graph", text=t, cites=pm))
        for p in pm:
            bib.setdefault(p, None)
    links: set[str] = set()
    ids = {s.sid for s in sents}
    for r in relations:
        a, b = str(r.get("src")), str(r.get("tgt"))
        if a in ids and b in ids:
            links.add(f"{a}|{b}")
            links.add(f"{b}|{a}")
    rep = Report(sections=[{"id": "graph", "title": "", "level": 1, "text": ""}], sentences=sents, bibliography=list(bib), words=sum(word_count(s.text) for s in sents))
    view = replace(ro, report=rep, outline=[], meta={**ro.meta, "view": "synthesis"})
    return view, links


def planning_view(ro: Rollout, sections: list[dict], claim_text: dict[str, str], claim_pmids: dict[str, list[str]]) -> Rollout:
    secs, sents, outline = [], [], []
    bib: dict[str, None] = {}
    for k, s in enumerate(sections):
        sid = str(s.get("id") or f"s{k}")
        title = _norm(s.get("title"))
        level = 1 if not s.get("parent") else 2
        outline.append(OutlineNode(id=sid, title=title, level=level, parent=s.get("parent")))
        texts = []
        for j, cid in enumerate(s.get("claim_ids") or []):
            t = _norm(claim_text.get(str(cid)))
            if not t:
                continue
            pm = [str(p) for p in claim_pmids.get(str(cid), [])]
            sents.append(Sentence(sid=f"{sid}#{j}", section=sid, text=t, cites=pm))
            for p in pm:
                bib.setdefault(p, None)
            texts.append(t)
        secs.append({"id": sid, "title": title, "level": level, "text": "\n\n".join(texts)})
    rep = Report(sections=secs, sentences=sents, bibliography=list(bib), words=sum(word_count(s.text) for s in sents))
    return replace(ro, report=rep, outline=outline, meta={**ro.meta, "view": "planning"})


def _agent_claims(sg: dict | None) -> tuple[list[dict], list[dict]]:
    if not sg:
        return [], []
    claims = [{"id": c["claim_id"], "text": c.get("text") or "", "pmids": [str(p) for p in (c.get("cited_paper_ids") or [])] or [str(e).replace("e_", "").split("_")[0] for e in (c.get("evidence_ids") or [])]} for c in sg.get("claims", [])]
    rels = [{"src": r["source"], "tgt": r["target"]} for r in sg.get("relations", [])]
    return claims, rels


def agent_view(ro: Rollout, window: str) -> tuple[Rollout | None, set[str]]:
    if ro.is_bot and window in ("writing", "system"):
        return None, set()
    run = agent_runs.load_run(paths.agent_run_dir(ro.system, ro.task, ro.mode))
    if window in ("writing", "system"):
        return ro, set()
    if window == "retrieval":
        return ro, set()
    if window == "synthesis":
        claims, rels = _agent_claims(run["synthesis_graph"])
        if not claims:
            return None, set()
        return synthesis_view(ro, claims, rels)
    if window == "planning":
        op = run["outline_plan"]
        if not op:
            return None, set()
        src = None
        for key in ("entry_synthesis_graph", "synthesis_graph"):
            claims, _ = _agent_claims(run.get(key))
            ids = {c["id"] for c in claims}
            if any(cid in ids for s in op.get("sections", []) for cid in (s.get("claim_ids") or [])):
                src = claims
                break
        if src is None:
            src, _ = _agent_claims(run.get("entry_synthesis_graph") or run.get("synthesis_graph"))
        ct = {c["id"]: c["text"] for c in src}
        cp = {c["id"]: c["pmids"] for c in src}
        secs = [{"id": s["section_id"], "title": s.get("title") or "", "parent": s.get("parent_id"), "claim_ids": s.get("claim_ids") or []} for s in op.get("sections", [])]
        return planning_view(ro, secs, ct, cp), set()
    raise ValueError(window)


def human_view(hro: Rollout, window: str) -> tuple[Rollout | None, set[str]]:
    src = hro.meta.get("source_review")
    if not src or not graphs.exists(src):
        return None, set()
    g = graphs.load(src)
    if window in ("writing", "system", "retrieval"):
        return hro, set()
    if window == "system_trunc":
        return truncated_view(hro), set()
    if window == "synthesis":
        claims = [{"id": c.id, "text": c.text, "pmids": c.pmids} for c in g.claims]
        rels = [{"src": e.src, "tgt": e.tgt} for e in g.edges]
        return synthesis_view(hro, claims, rels)
    if window == "planning":
        top = {s.id: g.top_section(s.id) for s in g.sections}
        body = [s for s in g.sections if s.section_type != "abstract" and (s.level == 1 or s.parent is None)]
        by_sec: dict[str, list[str]] = {}
        for c in g.claims:
            by_sec.setdefault(top.get(c.section_id) or c.section_id or "s0", []).append(c.id)
        secs = [{"id": s.id, "title": s.title, "parent": None, "claim_ids": by_sec.get(s.id, [])} for s in body]
        return planning_view(hro, secs, {c.id: c.text for c in g.claims}, {c.id: list(c.pmids) for c in g.claims}), set()
    raise ValueError(window)


def _cut(text: str, k: int) -> str:
    if k <= 0:
        return ""
    ms = list(WORD_RE.finditer(text))
    return text if len(ms) <= k else text[: ms[k - 1].end()]


def _cut_section_text(text: str, kept: list[Sentence], n_complete: int) -> str:
    paras, n = [], 0
    for p in re.split(r"\n\s*\n", text):
        if not p.strip():
            continue
        c = len(split_sentences(re.sub(r"\[[^\]]*\]", "", p)))
        if n + c > n_complete:
            break
        paras.append(p)
        n += c
    rest = " ".join(s.text for s in kept[n:])
    return "\n\n".join(paras + ([rest] if rest else []))


def truncated_view(ro: Rollout) -> Rollout:
    tw = ro.context.target_words or 0
    if ro.is_bot or not ro.report or not tw or ro.report.words <= tw:
        return ro
    sents, total, cut = [], 0, False
    for s in ro.report.sentences:
        n = word_count(s.text)
        if total + n <= tw:
            sents.append(s)
            total += n
            continue
        head = _cut(s.text, tw - total)
        if head:
            sents.append(replace(s, text=head))
            total, cut = tw, True
        break
    last = sents[-1].section if sents else None
    kept_last = [s for s in sents if s.section == last]
    n_full = sum(1 for s in ro.report.sentences if s.section == last)
    secs = []
    for sec in ro.report.sections:
        if sec["id"] == last:
            if cut or len(kept_last) < n_full:
                sec = {**sec, "text": _cut_section_text(sec.get("text") or "", kept_last, len(kept_last) - int(cut))}
            secs.append(sec)
            break
        secs.append(sec)
    bib: dict[str, None] = {}
    for s in sents:
        for c in s.cites:
            bib.setdefault(c, None)
    rep = Report(sections=secs, sentences=sents, bibliography=list(bib), words=total, unresolved_citations=ro.report.unresolved_citations, total_citation_marks=sum(len(s.cites) for s in sents))
    return replace(ro, report=rep, meta={**ro.meta, "view": "truncated", "truncated_from_words": ro.report.words})


def induced_path(system_key: str, task: str, window: str):
    if window in ("writing", "system"):
        return channel.induced_path(system_key, task)
    if window == "system_trunc":
        return paths.out_dir("E13", "induced", "system_trunc", system_key) / f"{task}.json"
    return paths.out_dir("E13", "induced", window, system_key) / f"{task}.json"


def score_view(view: Rollout, links: set[str], window: str, g: graphs.GStar, u: Units, key: str, cache: bool = True, tau_floor: float | None = None) -> list[dict]:
    p = induced_path(key, view.task, window)
    if cache and p.exists() and tau_floor is None:
        ind = channel.Induced.from_json(p)
    else:
        ind = channel.induce(view, g, u, tau_floor=tau_floor, window=window)
        if window == "synthesis":
            ind.sent_paragraph = {s.sid: s.sid for s in view.report.sentences}
            ind.sys_links = links
        if tau_floor is None:
            ind.to_json(p)
    rows = []
    for s in subgraph.all_unit_readouts(view, ind, g, u):
        if window not in ("writing", "system", "system_trunc") and not readouts_observable_at(window, s.name):
            continue
        if window == "retrieval" and s.name == "completion":
            continue
        rows.append({"system": key, "task": view.task, "panel": view.panel, "mode": view.mode, "window": window, "readout": s.name, "stage": s.stage, "family": s.family, "stratum": s.stratum, "criterion": s.criterion, "direction": s.direction, "value": s.value, "is_bot": s.value is None, "n_units": s.n_units, "note": s.note, "source_review": view.meta.get("source_review")})
    return rows
