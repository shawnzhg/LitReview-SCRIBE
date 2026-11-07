"""Window extractor for SurveyGen-I task directories: rebuilds the outline, per-subsection evidence,
first complete draft and refinement passes from the call log and writing files."""

from __future__ import annotations

import json
import re
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

ROLE_OF = OrderedDict()
SIGNATURES = [
    ("topic_clarify",        "other",           "helping to rephrase and clarify a user's research input"),
    ("keywords",             "other",           "prepare high-quality keyword search strategies"),
    ("screen_global",        "evidence_select", "Assess how relevant a research paper is to a given survey"),
    ("screen_subsection",    "evidence_select", "evaluating the relevance of a paper to a research topic"),
    ("subsection_clarify",   "other",           "part of a system that is writing an academic survey"),
    ("citation_trace",       "evidence_select", "assisting in academic survey writing. We are currently writing a subsection titled"),
    ("outline_gen",          "outline_gen",     "design of conceptually rich and future-facing survey structures"),
    ("outline_refine",       "outline_revise",  "helping refine the structure of a survey paper outline"),
    ("plan",                 "other",           "research planner helping coordinate the efficient writing"),
    ("dependency",           "other",           "survey planner. Your task is to determine which prior subsections each subsection depends on"),
    ("restructure",          "outline_revise",  "refining the structure of a technical survey paper"),
    ("skeleton",             "digest",          "conceptually structured writing skeleton"),
    ("write",                "draft",           "writing a **full subsection** of a technical academic survey paper"),
    ("select_best",          "select_best",     "selecting the best-written version among multiple candidates"),
    ("refine1_structure",    "refine_coherence","refining a **technical academic survey subsection** written according to a pre-defined structure"),
    ("refine2_citation",     "refine_citation", "performing a **second-stage refinement** of a technical survey subsection"),
    ("refine3_readability",  "refine_coherence","improve its readability, clarity, and paragraph structure"),
    ("terminology_memory",   "digest",          "extract important technical terms from the following academic subsection"),
    ("consistency_review",   "review",          "identify any logical conflicts or inconsistencies"),
    ("style_review",         "review",          "helping improve the writing quality of an academic survey paper"),
    ("resolve_conflict",     "refine_coherence","assisting in resolving logical inconsistencies in a technical survey paper"),
    ("style_rewrite",        "refine_coherence","improving the academic writing style of a survey paper subsection"),
    ("abstract",             "final_assembly",  "generate a concise and informative abstract"),
    ("intro_overview",       "final_assembly",  "writing the final subsection of the **introduction section**"),
    ("glossary",             "final_assembly",  "building a glossary of technical terms from an academic survey section"),
    ("table_focus",          "final_assembly",  "identify the single most important aspect that defines how to compare or categorize papers"),
    ("table_aspects",        "final_assembly",  "identify and prioritize the key comparison dimensions for a set of research papers"),
    ("table_categories",     "final_assembly",  "scholarly assistant for academic table construction"),
    ("table_categorize",     "final_assembly",  "tasked with categorizing a research paper based on its technical content"),
    ("table_caption",        "final_assembly",  "expert academic writer for EMNLP"),
    ("table_paragraph",      "final_assembly",  "write a short academic-style LaTeX paragraph that summarizes the following table"),
    ("table_qa",             "final_assembly",  "specialized in extracting concise, factual information from research papers"),
    ("table_cell",           "final_assembly",  "concise table-cell summarizer"),
    ("fig_overview",         "final_assembly",  "constructing a **concise, hierarchical conceptual overview**"),
    ("fig_node_bibkeys",     "final_assembly",  "helping to build a structured conceptual overview of a survey"),
    ("fig_sections",         "final_assembly",  "assisting in designing structure figures for an academic survey paper"),
    ("fig_section_tree",     "final_assembly",  "constructing a conceptual structure for **one section**"),
    ("latex_format",         "final_assembly",  "expert LaTeX formatter specializing in preparing scientific text for ACM"),
    ("bib_title_parse",      "other",           "expert bibliographic parser"),
]
for _t, _r, _s in SIGNATURES:
    ROLE_OF[_t] = _r
SIG_OF = {t: s for t, _r, s in SIGNATURES}

WS = re.compile(r"\s+")
CITE = re.compile(r"\[((?:pmid\d+\s*[;,]?\s*)+)\]")
WORD = re.compile(r"\b\w+\b")
PMID_IN_EVIDENCE = re.compile(r"\*\*Original Bibkey\*\*: (pmid\d+)")
TRACED = re.compile(r"Traced BIBKEY:?\*?\*?\s*(pmid\d+)")
VARIANT_BLOCK = re.compile(r"### Variant (\d+):\s*\n(.*?)(?=\n### Variant \d+:|\n---\s*\n\s*### Output Format)", re.S)

TITLE_RX = {
    "skeleton":            re.compile(r"Subsection title: (.+)"),
    "write":               re.compile(r"### Title\s*\n\s*`([^`\n]+)`"),
    "refine1_structure":   re.compile(r"\*\*Subsection Title\*\*: `([^`\n]+)`"),
    "refine2_citation":    re.compile(r"\*\*Subsection Title\*\*: `([^`\n]+)`"),
    "refine3_readability": re.compile(r"## Subsection Title\s*\n\s*`([^`\n]+)`"),
    "terminology_memory":  re.compile(r"### Subsection Title:\s*\n(.+)"),
    "citation_trace":      re.compile(r'subsection titled: "([^"\n]+)"'),
    "subsection_clarify":  re.compile(r"Subsection Title: (.+)"),
    "dependency":          re.compile(r"Current subsection:\s*\nTitle: (.+)"),
    "style_rewrite":       re.compile(r"## Subsection Location\s*\n(.+)"),
    "table_focus":         re.compile(r"- Subsection: (.+)"),
}


def collapse(s: str) -> str:
    return WS.sub(" ", s or "").strip()


def haystack(system: str, user: str) -> str:
    return collapse(system[:400]) + " || " + collapse(user[:3000]) + " ~~ " + collapse(user[-2000:])


def classify(hay: str) -> tuple[str, list[str]]:
    hits = [t for t, s in SIG_OF.items() if s in hay]
    if len(hits) == 1:
        return hits[0], hits
    if not hits:
        return "unmatched", hits
    return "ambiguous", hits


def _content(m) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return c or ""


def _json_loads_loose(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(t[i:j + 1])
            except Exception:
                return None
    return None


def word_count(text: str) -> int:
    return len(WORD.findall(CITE.sub(" ", text or "")))


def cited(text: str) -> list[str]:
    out = []
    for m in CITE.findall(text or ""):
        out.extend(re.findall(r"pmid(\d+)", m))
    return out


def outline_nodes(o: dict) -> list[dict]:
    nodes = []
    for i, sec in enumerate(o.get("sections") or [], 1):
        sid = str(i)
        nodes.append({"id": sid, "title": sec.get("section_title", ""), "level": 1, "parent": None,
                      "description": sec.get("section_description", "")})
        for j, sub in enumerate(sec.get("subsections") or [], 1):
            nodes.append({"id": f"{sid}.{j}", "title": sub.get("subsection_title", ""), "level": 2, "parent": sid,
                          "description": sub.get("subsection_description", "")})
    return nodes


def _load_json(p: Path):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def extract(task_dir: Path) -> dict:
    task_dir = Path(task_dir)
    calls_path = task_dir / "_calls.jsonl"

    per_call: list[tuple[int, str, str]] = []
    tpl_seqs: dict[str, list[int]] = defaultdict(list)
    first_write_seq = None
    first_write_t = None

    outline_gen_raw = None
    outline_refine_raw = None
    restructure_calls: list[dict] = []
    writes: dict[str, list[dict]] = defaultdict(list)
    selects: list[dict] = []
    refines: dict[str, dict[str, dict]] = defaultdict(dict)

    with open(calls_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = r.get("seq")
            msgs = r.get("messages") or []
            system = "\n".join(_content(m) for m in msgs if m.get("role") == "system")
            user = "\n".join(_content(m) for m in msgs if m.get("role") == "user")
            if not msgs and r.get("prompt"):
                user = r["prompt"]
            comp = r.get("completion") or ""
            if not (system or user):
                tpl, hits = "api_error", []
            else:
                tpl, hits = classify(haystack(system, user))
            if r.get("status") not in (None, 200):
                pass
            role = ROLE_OF.get(tpl, "other")
            per_call.append((seq, tpl, role))
            tpl_seqs[tpl].append(seq)

            rx = TITLE_RX.get(tpl)
            title = ""
            if rx:
                m = rx.search(user)
                title = m.group(1).strip() if m else ""

            if tpl == "outline_gen":
                outline_gen_raw = _json_loads_loose(comp)
            elif tpl == "outline_refine":
                outline_refine_raw = _json_loads_loose(comp)
            elif tpl == "plan":
                d = _json_loads_loose(comp)
            elif tpl == "restructure":
                d = _json_loads_loose(comp)
                recs = (d or {}).get("recommendations") if isinstance(d, dict) else None
                restructure_calls.append({"seq": seq, "t": r.get("t"),
                                          "n_recommendations": len(recs) if recs is not None else None,
                                          "actions": dict(Counter(x.get("action") for x in recs)) if recs else {},
                                          "finish_reason": r.get("finish_reason")})
            elif tpl == "skeleton":
                d = _json_loads_loose(comp)
            elif tpl == "write":
                if first_write_seq is None:
                    first_write_seq, first_write_t = seq, r.get("t")
                ev = list(OrderedDict.fromkeys(PMID_IN_EVIDENCE.findall(user)))
                tr = list(OrderedDict.fromkeys(TRACED.findall(user)))
                writes[title].append({"seq": seq, "text": comp, "evidence": ev, "traced": tr,
                                      "n_trace_skipped": user.count("Skipped — no traceworthy markers"),
                                      "n_trace_resolved": len(TRACED.findall(user)),
                                      "finish_reason": r.get("finish_reason")})
            elif tpl == "select_best":
                blocks = [(int(k), v.strip()) for k, v in VARIANT_BLOCK.findall(user)]
                d = _json_loads_loose(comp)
                best = (d or {}).get("best_variant") if isinstance(d, dict) else None
                selects.append({"seq": seq, "variants": blocks, "best": best})
            elif tpl in ("refine1_structure", "refine2_citation", "refine3_readability"):
                ent = {"seq": seq, "text": comp}
                if tpl == "refine2_citation":
                    ent["evidence"] = list(OrderedDict.fromkeys(PMID_IN_EVIDENCE.findall(user)))
                refines[title][tpl] = ent
            elif tpl == "terminology_memory":
                pass
            elif tpl in ("consistency_review", "style_review", "abstract", "intro_overview", "fig_overview", "fig_sections"):
                d = _json_loads_loose(comp) if tpl in ("consistency_review", "style_review", "fig_sections") else None


    call_roles: list[tuple[list[int], str]] = []
    call_templates: list[tuple[list[int], str]] = []
    for seq, tpl, role in per_call:
        if call_roles and call_roles[-1][1] == role and call_roles[-1][0][1] == seq - 1:
            call_roles[-1][0][1] = seq
        else:
            call_roles.append([[seq, seq], role])
        if call_templates and call_templates[-1][1] == tpl and call_templates[-1][0][1] == seq - 1:
            call_templates[-1][0][1] = seq
        else:
            call_templates.append([[seq, seq], tpl])

    wdir = task_dir / "writing"
    f_initial = _load_json(wdir / "final_survey_outline.json")
    upd = sorted(wdir.glob("survey_outline_updated_*.json"), key=lambda p: int(re.search(r"_(\d+)\.json$", p.name).group(1)))
    plan_files = sorted(wdir.glob("writing_plan_updated_*.json"), key=lambda p: int(re.search(r"_(\d+)\.json$", p.name).group(1)))
    f_plan0 = _load_json(wdir / "writing_plan.json")
    ever_search: dict[str, bool] = {}
    for pl in [f_plan0] + [_load_json(q) for q in plan_files]:
        for it in pl or []:
            k = it.get("subsection_title", "")
            ever_search[k] = ever_search.get(k, False) or bool(it.get("trigger_additional_search"))

    gen_o = (outline_gen_raw or {}).get("outline") if isinstance(outline_gen_raw, dict) else None
    ref_o = (outline_refine_raw or {}).get("outline") if isinstance(outline_refine_raw, dict) else None
    if f_initial:
        outline_initial = outline_nodes(f_initial)
        src_initial = "writing/final_survey_outline.json"
        if ref_o and outline_nodes(ref_o) != outline_initial:
            pass
    elif ref_o:
        outline_initial = outline_nodes(ref_o); src_initial = "completion of outline_refine call"
    elif gen_o:
        outline_initial = outline_nodes(gen_o); src_initial = "completion of outline_gen call"
    else:
        outline_initial = []; src_initial = "none"

    revision_events = []
    for p in upd:
        o = _load_json(p)
        nd = outline_nodes(o) if o else []
        revision_events.append({"file": f"writing/{p.name}", "n_sections": sum(1 for n in nd if n["level"] == 1),
                                "n_subsections": sum(1 for n in nd if n["level"] == 2)})
    for i, rc in enumerate(restructure_calls):
        if i < len(revision_events):
            revision_events[i].update({"restructure_seq": rc["seq"], "t": rc["t"], "n_recommendations": rc["n_recommendations"], "actions": rc["actions"]})
    if len(restructure_calls) != len(upd):
        pass
    if upd:
        outline_final = outline_nodes(_load_json(upd[-1]) or {})
    else:
        outline_final = outline_initial; src_final = src_initial

    title2id = {n["title"]: n["id"] for n in outline_final if n["level"] == 2}

    def sid(title: str) -> str:
        return title2id.get(title, f"?:{title}")


    section_evidence: dict[str, list[str]] = {}
    for title, L in writes.items():
        L.sort(key=lambda x: x["seq"])
        ev = L[0]["evidence"]; tr = L[0]["traced"]
        section_evidence[sid(title)] = ev

    def match_selected(L: list[dict]) -> dict | None:
        for s in selects:
            texts = {k: v for k, v in s["variants"]}
            for x in L:
                if any(x["text"].strip() == v for v in texts.values()):
                    b = s["best"]
                    chosen = texts.get(b) if isinstance(b, int) else None
                    if chosen is None:
                        return {"seq": s["seq"], "best": b, "text": None}
                    hit = next((x2 for x2 in L if x2["text"].strip() == chosen), None)
                    return {"seq": s["seq"], "best": b, "text": chosen, "write_seq": hit["seq"] if hit else None}
        return None

    per_sub = {}
    for title, L in writes.items():
        sel = match_selected(L)
        distinct = len({x["text"].strip() for x in L})
        stages = {"draft_first": L[0]["text"]}
        if sel and sel.get("text"):
            stages["draft_selected"] = sel["text"]
        for st in ("refine1_structure", "refine2_citation", "refine3_readability"):
            if st in refines[title]:
                stages[st] = refines[title][st]["text"]
        per_sub[title] = {"write_seqs": [x["seq"] for x in L], "n_distinct_variants": distinct,
                          "select_seq": sel["seq"] if sel else None, "best_variant": sel["best"] if sel else None,
                          "selected_write_seq": sel.get("write_seq") if sel else None,
                          "stages": {k: {"words": word_count(v), "n_cites": len(cited(v)), "n_distinct_cites": len(set(cited(v)))}
                                     for k, v in stages.items()},
                          "_texts": stages}

    def assemble(which: str) -> str:
        title = (_load_json(upd[-1]) if upd else f_initial or {}).get("title", "") if (upd or f_initial) else ""
        out = [f"# {title}".rstrip(), ""]
        for n in outline_final:
            if n["level"] == 1:
                out += [f"## {n['title']}", ""]
            else:
                out += [f"### {n['title']}", ""]
                d = per_sub.get(n["title"])
                txt = None
                if d:
                    txt = d["_texts"].get(which) or (d["_texts"].get("draft_first") if which == "draft_selected" else None)
                out += [txt.strip() if txt else "[no draft]", ""]
        return "\n".join(out)

    draft_report = assemble("draft_first")

    def rng(t):
        s = tpl_seqs.get(t) or []
        return [s[0], s[-1]] if s else None

    refinement_passes = []
    for t in ("select_best", "refine1_structure", "refine2_citation", "refine3_readability", "consistency_review",
              "resolve_conflict", "style_review", "style_rewrite", "latex_format"):
        if t in tpl_seqs:
            refinement_passes.append({"role": ROLE_OF[t], "seq_range": rng(t)})

    sp = task_dir / "search_calls.jsonl"
    if sp.exists():
        with open(sp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                ts = r.get("ts") or ""
                if first_write_t:
                    if ts.replace("Z", "+00:00") < first_write_t:
                        pass
                    else:
                        pass
                for h in r.get("returned") or []:
                    p = h.get("pmid") if isinstance(h, dict) else None
                    if p:
                        if r.get("route") == "pdf":
                            pass
    bib = task_dir / "acm" / "references_master.bib"
    if bib.exists():
        pass

    def subs(t):
        parts = re.split(r"\n(?=### )", t)
        return {p.split("\n", 1)[0]: p for p in parts if p.startswith("### ")}


    sc_raw = _load_json(wdir / "subsection_contents.json")
    if isinstance(sc_raw, dict):
        for v in sc_raw.values():
            if isinstance(v, dict) and "subsection" in v:
                pass



    return {
        "first_write_seq": first_write_seq,
        "outline_initial": outline_initial,
        "outline_final": outline_final,
        "section_evidence": section_evidence,
        "draft_report": draft_report,
        "refinement_passes": [{"role": p.get("role"), "seq_range": p.get("seq_range")} for p in refinement_passes],
    }
