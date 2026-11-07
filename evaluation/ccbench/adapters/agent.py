"""Adapter for staged agent runs: maps the artifact chain (evidence bundle, claim graph, outline,
report, trace) onto a rollout."""

from __future__ import annotations

from pathlib import Path

from ccbench import labels, paths
from ccbench.adapters.common import split_sentences, word_count
from ccbench.ingest import agent_runs, gold
from ccbench.model import BOT, Event, OutlineNode, Report, Rollout, Sentence, bot_rollout

KIND_MAP = {"llm": "llm", "retrieval_search": "search", "retrieval_open": "open", "validate": "validate", "retry": "retry", "failure": "failure"}


def _events(run_dir: Path) -> tuple[list[Event], dict]:
    events: list[Event] = []
    tot = {"in_tokens": 0, "out_tokens": 0, "usd": 0.0, "wall_ms": 0, "n_llm": 0, "n_llm_err": 0, "n_search": 0, "n_open": 0}
    for i, r in enumerate(agent_runs.iter_trace(run_dir)):
        res = r.get("resources") or {}
        kind = KIND_MAP.get(r.get("kind"), r.get("kind") or "llm")
        ret = r.get("retrieval") or {}
        returned = [str(x.get("paper_id")) for x in (ret.get("returned") or []) if isinstance(x, dict) and x.get("paper_id")]
        llm = r.get("llm") or {}
        events.append(
            Event(
                seq=int(r.get("seq", i)),
                t_start=r["_t0"] or 0.0,
                t_end=r["_t1"] or r["_t0"] or 0.0,
                kind=kind,
                label=labels.label_from_trace(r),
                resources={"in_tokens": res.get("in_tokens") or 0, "out_tokens": res.get("out_tokens") or 0, "wall_ms": res.get("wall_ms") or 0, "usd": res.get("usd") or 0.0},
                route=ret.get("route"),
                n_returned=len(returned) if returned else ret.get("k_requested"),
                returned_ids=returned,
                query=ret.get("query"),
                finish_reason=llm.get("finish_reason"),
                prompt_chars=llm.get("n_prompt_tokens") or 0,
                completion_chars=llm.get("n_completion_tokens") or 0,
            )
        )
        tot["in_tokens"] += res.get("in_tokens") or 0
        tot["out_tokens"] += res.get("out_tokens") or 0
        tot["usd"] += res.get("usd") or 0.0
        tot["wall_ms"] += res.get("wall_ms") or 0
        if kind == "llm":
            tot["n_llm"] += 1
        elif kind == "search":
            tot["n_search"] += 1
        elif kind == "open":
            tot["n_open"] += 1
        elif kind == "failure":
            tot["n_llm_err"] += 1
    return events, tot


def _outline(op: dict | None) -> list[OutlineNode]:
    if not op:
        return []
    out = []
    for s in op.get("sections", []):
        out.append(OutlineNode(id=s["section_id"], title=s.get("title") or "", level=1 if not s.get("parent_id") else 2, parent=s.get("parent_id")))
    return out


def _has_stored_sentences(ra: dict) -> bool:
    secs = ra.get("sections") or []
    if not secs:
        return False
    for sec in secs:
        ents = sec.get("sentences")
        if not isinstance(ents, list):
            return False
        if not ents and str(sec.get("text") or "").strip():
            return False
    return True


NESTED_LEVEL_PLANS = ("entry_outline_plan",)


def section_title_map(run: dict, report_sections: list[dict]) -> tuple[dict[str, tuple[str, int]], str]:
    rep_ids = [str(sec.get("section_id") or "") for sec in report_sections]
    order = ["outline_plan", "entry_outline_plan"]
    best_name, best_cov, best_plan = None, 0.0, None
    for name in order:
        plan = run.get(name)
        if not plan or not plan.get("sections"):
            continue
        ids = {s["section_id"] for s in plan["sections"]}
        cov = (sum(r in ids for r in rep_ids) / len(rep_ids)) if rep_ids else 0.0
        if cov > best_cov:
            best_name, best_cov, best_plan = name, cov, plan
    out: dict[str, tuple[str, int]] = {}
    if best_plan is not None:
        nested = best_name in NESTED_LEVEL_PLANS
        for s in best_plan["sections"]:
            out[s["section_id"]] = (s.get("title") or "", 2 if (nested and s.get("parent_id")) else 1)
    source = best_name or "none"
    n_stored = 0
    for sec in report_sections:
        sid = str(sec.get("section_id") or "")
        if sid not in out and sec.get("title"):
            out[sid] = (str(sec["title"]), 1)
            n_stored += 1
    if n_stored and best_plan is None:
        source = "stored"
    elif n_stored:
        source = f"{source}+stored"
    return out, source


def _report_ex(ra: dict | None, outline_titles: list[str] | None = None, title_map: dict[str, tuple[str, int]] | None = None) -> tuple[Report | None, dict]:
    if not ra:
        return None, {"report_sentence_source": None, "report_title_source": None}
    cmap: dict[str, list[str]] = {}
    for c in ra.get("citation_evidence_map", []):
        cmap.setdefault(c["sentence_id"], []).append(str(c["paper_id"]))
    sentences: list[Sentence] = []
    sections = []
    words = 0
    stored = _has_stored_sentences(ra)
    n_global = 0
    ids_in_sec: dict[str, list[int]] = {}
    for row in list(ra.get("citation_evidence_map", [])) + list(ra.get("sentence_claim_map", [])):
        s_id, _, k_ = str(row.get("sentence_id", "")).rpartition("#")
        if s_id and k_.isdigit():
            ids_in_sec.setdefault(s_id, []).append(int(k_))
    max_prev, n_snaps = -1, 0
    for k, sec in enumerate(ra.get("sections", [])):
        sid = sec.get("section_id") or f"s{len(sections)}"
        text = sec.get("text") or ""
        if title_map and sid in title_map:
            title, level = title_map[sid]
        else:
            title = sec.get("title") or (outline_titles[k] if outline_titles and k < len(outline_titles) else "")
            level = 1
        sections.append({"id": sid, "title": title, "level": level, "text": text})
        words += word_count(text)
        if not stored:
            seen = ids_in_sec.get(sid, [])
            if n_global < max_prev + 1:
                n_global, n_snaps = max_prev + 1, n_snaps + 1
            if seen and n_global > min(seen):
                n_global, n_snaps = min(seen), n_snaps + 1
            if seen:
                max_prev = max(max_prev, max(seen))
        if stored:
            for ent in sec["sentences"]:
                if not isinstance(ent, dict):
                    continue
                stext = str(ent.get("text") or "").strip()
                if not stext:
                    continue
                key = str(ent.get("sentence_id") or f"{sid}#{n_global}")
                n_global += 1
                cites = cmap.get(key) or [str(p) for p in (ent.get("citations") or [])]
                sentences.append(Sentence(sid=key, section=sid, text=stext, cites=list(cites)))
        else:
            for s in split_sentences(text):
                key = f"{sid}#{n_global}"
                n_global += 1
                sentences.append(Sentence(sid=key, section=sid, text=s, cites=cmap.get(key, [])))
    bib = [str(p) for p in ra.get("bibliography", [])]
    ta = ra.get("terminal_audit") or {}
    known = {s.sid for s in sentences}
    n_rows = len(ra.get("citation_evidence_map", []))
    n_hit = sum(1 for c in ra.get("citation_evidence_map", []) if c["sentence_id"] in known)
    meta = {
        "report_sentence_source": "stored" if stored else "resplit_global",
        "report_citation_rows_aligned": n_hit,
        "report_citation_rows": n_rows,
        "report_citation_alignment": (n_hit / n_rows) if n_rows else None,
        "report_resplit_anchor_snaps": None if stored else n_snaps,
    }
    rep = Report(sections=sections, sentences=sentences, bibliography=bib, words=int(ta.get("n_words") or words), unresolved_citations=len(ta.get("fabricated_citations") or []), total_citation_marks=int(ta.get("n_citations") or sum(len(c) for c in cmap.values())))
    return rep, meta


def _graph(sg: dict | None) -> dict | None:
    if not sg:
        return None
    nodes = [{"id": c["claim_id"], "text": c.get("text") or "", "pmids": [str(p) for p in (c.get("cited_paper_ids") or [])] or [str(e).replace("e_", "").split("_")[0] for e in (c.get("evidence_ids") or [])], "type": c.get("type")} for c in sg.get("claims", [])]
    edges = [{"src": r["source"], "tgt": r["target"], "type": r.get("fine_type") or r.get("type")} for r in sg.get("relations", [])]
    return {"nodes": nodes, "edges": edges, "role": sg.get("role")}


def adapt(system: str, task: str, mode: str) -> Rollout:
    ctx = gold.context_for(task)
    try:
        run_dir = paths.agent_run_dir(system, task, mode)
    except FileNotFoundError:
        return bot_rollout(system, task, "B", ctx, "run_dir_missing", mode=mode)
    run = agent_runs.load_run(run_dir)
    ws = agent_runs.window_status(run)
    events, tot = _events(run_dir)
    eb = run["evidence_bundle"] or run["entry_evidence_bundle"] or {}
    papers = [str(p.get("paper_id")) for p in sorted(eb.get("papers", []), key=lambda p: p.get("rank") or 0) if p.get("paper_id")]
    evidence_ids = [e.get("evidence_id") for e in eb.get("evidence", []) if e.get("evidence_id")]
    outline_nodes = _outline(run["outline_plan"])
    rep_secs = (run["report_artifact"] or {}).get("sections", [])
    n_sec = len(rep_secs)
    top = [o.title for o in outline_nodes if not o.parent]
    title_map, title_src = section_title_map(run, rep_secs)
    report, rmeta = _report_ex(run["report_artifact"], top if len(top) == n_sec else [o.title for o in outline_nodes], title_map)
    failed = [w for w, s in ws.items() if s.get("status") not in (None, "ok")]
    ok = report is not None and not failed
    ro = Rollout(
        system=system,
        task=task,
        panel="B",
        context=ctx,
        status="ok" if ok else BOT,
        bot_reason=None if ok else ("no_report" if report is None else f"window_failed:{','.join(failed)}"),
        mode=mode,
        events=events,
        papers=papers,
        evidence_ids=evidence_ids,
        outline=outline_nodes,
        report=report,
        graph=_graph(run["synthesis_graph"]),
        resources={**tot, **{k: v for k, v in (run["resources"] or {}).items() if k in ("wall_ms", "gpu_seconds")}},
        window_hashes={w: (s["entry_hash"], s["exit_hash"]) for w, s in ws.items()},
        meta={
            "run_dir": str(run_dir),
            "provenance_tier": eb.get("provenance_tier"),
            "entry_artifacts": {k: bool(run.get(k)) for k in agent_runs.ENTRY_ARTIFACTS},
            "report_title_source": title_src,
            "report_sections_titled": sum(1 for sec in rep_secs if str(sec.get("section_id") or "") in title_map),
            **rmeta,
        },
    )
    return ro
