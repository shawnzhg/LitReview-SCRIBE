"""Window extractor for AutoSurvey task directories: rebuilds the outline, per-subsection evidence,
first complete draft and refinement passes from the call log and output files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


SIG_OUTLINE_GEN = "You need to draft a outline based on the given papers"
SIG_OUTLINE_MERGE = "You need to generate a final outline based on these provided outlines"
SIG_OUTLINE_SUBSECTION = "You need to enrich the section "
SIG_OUTLINE_EDIT = "You need to modify the outline to make it both comprehensive and logically coherent"
SIG_DRAFT = "Now you need to write the content for the subsection:"
SIG_REFINE_CITATION = 'Now you need to check whether the citations of "paper_title" in this subsection is correct'
SIG_REFINE_COHERENCE = "Now refine the subsection to enhance coherence"
SIG_JUDGE = "Please evaluate this survey about the topic"
SIG_NLI = "Is the Claim faithful to the Source?"
SIG_PROBE = "hello"

SIGNATURES: list[tuple[str, str]] = [
    ("outline_edit", SIG_OUTLINE_EDIT),
    ("outline_subsection", SIG_OUTLINE_SUBSECTION),
    ("outline_merge", SIG_OUTLINE_MERGE),
    ("outline_gen", SIG_OUTLINE_GEN),
    ("refine_coherence", SIG_REFINE_COHERENCE),
    ("refine_citation", SIG_REFINE_CITATION),
    ("draft", SIG_DRAFT),
    ("select_best", SIG_JUDGE),
    ("review", SIG_NLI),
]


A_EDIT_DRAFT_OUTLINE = "You have created a draft outline below:"
A_WRITE_OUTLINE = "You have created a overall outline below:"
A_WRITE_PAPERS = "Below are a list of papers for references:"
A_ENRICH_PAPERS = "These papers provided for references:"
A_ENRICH_FORMAT = "Return the outline in the format:"
A_CIT_SUBSECTION = "You have written a subsection below:"
A_INSTRUCTION = "\n<instruction>"
A_COH_PREV = "Previous Subsection:\n--- \n"
A_COH_FOLLOWING = "Following Subsection:\n---\n"
A_COH_TARGET = "Subsection to Refine: \n---\n"

RE_PAPER_TITLE = re.compile(r"^paper_title:\s*(.*)$", re.M)
RE_DRAFT_TARGET = re.compile(
    r'Now you need to write the content for the subsection:\n"(.+?)" under the section: "(.+?)"\n'
    r"The details of what to write in this subsection called ", re.S)
RE_CITE = re.compile(r"\[[^\]]*\]")
WS = re.compile(r"\s+")
SKIP_JSON = {"egress_proof.json", "listener_census.json", "pool_health.json"}
KEYLEN = 300


def ws(s: str | None) -> str:
    return WS.sub(" ", s or "").strip()


def tkey(s: str) -> str:
    return ws(s)[:KEYLEN]


def nocite(s: str) -> str:
    return ws(RE_CITE.sub("", s or ""))


def cited_titles(s: str) -> set[str]:
    out = set()
    for span in RE_CITE.findall(s or ""):
        for part in span[1:-1].split(";"):
            p = part.strip()
            if p:
                out.add(p)
    return out


def norm_title(t: str) -> str:
    t = re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()
    return re.sub(r"^the ", "", t)


def off_evidence(cited: set[str], papers: list[str]) -> int:
    pool = {norm_title(p) for p in papers}
    return sum(1 for c in cited if norm_title(c) not in pool)


def notation_spans(cited: set[str]) -> int:
    return sum(1 for c in cited if len(c) < 12 or "\\" in c or "=" in c or not re.search(r"[A-Za-z]{4}", c))


def words(s: str) -> int:
    return len((s or "").split())


def classify(system: str, user: str) -> str:
    if user.strip() == SIG_PROBE and len(user) < 32:
        return "probe"
    hay = system + "\n" + user
    for role, sig in SIGNATURES:
        if sig in hay:
            return role
    return "other"


BRIEF_SIGNATURES: list[tuple[str, str]] = [
    ("outline_gen", "You need to draft a outline based on the given papers"),
    ("outline_merge", "You need to generate a final outline based on these provided outlines"),
    ("outline_edit", "You need to modify the outline"),
    ("outline_subsection", "You need to enrich the section"),
    ("draft", "Return the content of subsection"),
    ("refine_citation", "Only return the subsection with correct citations"),
    ("refine_coherence", "refined content of the subsection"),
]


def classify_haystack(system: str, user: str) -> str:
    if user.strip() == SIG_PROBE and len(user) < 32:
        return "probe"
    hay = ws(system[:400]) + " || " + ws(user[:3000]) + " ~~ " + ws(user[-2000:])
    for role, sig in BRIEF_SIGNATURES:
        if ws(sig) in hay:
            return role
    return "other"


def strip_fmt(c: str) -> str:
    return (c or "").replace("<format>", "").replace("</format>", "")


def as_extract_title_sections_descriptions(outline: str) -> tuple[str, list[str], list[str]]:
    try:
        title = outline.split("Title: ")[1].split("\n")[0]
    except IndexError:
        return "", [], []
    sections, descriptions = [], []
    for i in range(100):
        if f"Section {i + 1}" in outline:
            try:
                sections.append(outline.split(f"Section {i + 1}: ")[1].split("\n")[0])
                descriptions.append(outline.split(f"Description {i + 1}: ")[1].split("\n")[0])
            except IndexError:
                break
    return title, sections, descriptions


def as_extract_subsections_subdescriptions(outline: str) -> tuple[list[str], list[str]]:
    subs, descs = [], []
    for i in range(100):
        if f"Subsection {i + 1}" in outline:
            try:
                subs.append(outline.split(f"Subsection {i + 1}: ")[1].split("\n")[0])
                descs.append(outline.split(f"Description {i + 1}: ")[1].split("\n")[0])
            except IndexError:
                break
    return subs, descs


def as_process_outlines(section_outline: str, sub_outlines: list[str]) -> str:
    title, sections, sec_desc = as_extract_title_sections_descriptions(section_outline)
    res = f"# {title}\n\n"
    for i, section in enumerate(sections):
        res += f"## {i + 1} {section}\nDescription: {sec_desc[i]}\n\n"
        subs, sub_desc = as_extract_subsections_subdescriptions(sub_outlines[i] if i < len(sub_outlines) else "")
        for j, sub in enumerate(subs):
            res += f"### {i + 1}.{j + 1} {sub}\nDescription: {sub_desc[j]}\n\n"
    return res


def as_parse_outline(outline: str) -> dict:
    res = {"title": "", "sections": [], "section_descriptions": [], "subsections": [], "subsection_descriptions": []}
    lines = (outline or "").split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            res["title"] = line[2:].strip()
        elif line.startswith("## "):
            res["sections"].append(line[3:].strip())
            if i + 1 < len(lines) and lines[i + 1].startswith("Description:"):
                res["section_descriptions"].append(lines[i + 1].split("Description:", 1)[1].strip())
                res["subsections"].append([])
                res["subsection_descriptions"].append([])
        elif line.startswith("### "):
            if res["subsections"]:
                res["subsections"][-1].append(line[4:].strip())
                if i + 1 < len(lines) and lines[i + 1].startswith("Description:"):
                    res["subsection_descriptions"][-1].append(lines[i + 1].split("Description:", 1)[1].strip())
    return res


def as_generate_document(parsed: dict, contents: list[list[str | None]]) -> str:
    doc = [f"# {parsed['title']}\n"]
    for i, section in enumerate(parsed["sections"]):
        doc.append(f"## {section}\n")
        for j, sub in enumerate(parsed["subsections"][i]):
            doc.append(f"### {sub}\n")
            if i < len(contents) and j < len(contents[i]) and contents[i][j] is not None:
                doc.append(contents[i][j] + "\n")
    return "\n".join(doc)


def _iter_calls(path: Path):
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _split_messages(r: dict) -> tuple[str, str]:
    msgs = r.get("messages") or []
    sysm = " ".join(m["content"] for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str))
    users = [m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)]
    return sysm, (users[-1] if users else "")


def _between(text: str, start_anchor: str, end_anchor: str | None = None) -> str:
    i = text.find(start_anchor)
    if i < 0:
        return ""
    seg = text[i + len(start_anchor):]
    if end_anchor:
        j = seg.find(end_anchor)
        if j >= 0:
            seg = seg[:j]
    k = seg.find("---\n")
    if k < 0:
        return ""
    seg = seg[k + 4:]
    m = seg.rfind("\n---")
    return seg[:m] if m >= 0 else seg


def _final_json(task_dir: Path) -> Path | None:
    cands = [p for p in task_dir.glob("*.json") if p.name not in SKIP_JSON and not p.name.startswith("_")]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


def extract(task_dir: Path, keep_intermediate: bool = False) -> dict:
    task_dir = Path(task_dir)
    calls = task_dir / "_calls.jsonl"
    if not calls.exists():
        return {
            "outline_initial": [],
            "outline_final": [],
            "section_evidence": {},
            "draft_report": "",
            "refinement_passes": [],
        }

    role_counts: dict[str, int] = {}
    haystack_roles: dict[str, int] = {}
    haystack_misses: dict[str, list[int]] = {}
    merge: tuple[int, str] | None = None
    enrich: list[tuple[int, str, list[str]]] = []
    enrich_probe: dict[int, str] = {}
    edit: tuple[int, str, str] | None = None
    drafts: dict[int, dict] = {}
    cits: dict[int, dict] = {}
    cohs: dict[int, dict] = {}
    prompt_hits: dict[str, list[tuple[int, bool]]] = {}

    for r in _iter_calls(calls):
        seq = r.get("seq")
        sysm, usr = _split_messages(r)
        comp = strip_fmt(r.get("completion") or "")
        role = classify(sysm, usr)
        hrole = classify_haystack(sysm, usr)
        role_counts[role] = role_counts.get(role, 0) + 1
        haystack_roles[hrole] = haystack_roles.get(hrole, 0) + 1
        if hrole != role:
            haystack_misses.setdefault(f"{role}->{hrole}", []).append(seq)

        usable = r.get("status") == 200 and not r.get("error") and bool(comp.strip())
        h = hashlib.sha1(usr.encode("utf-8", "replace")).hexdigest()
        prompt_hits.setdefault(h, []).append((seq, usable))
        if not usable:
            continue

        if role == "outline_gen":
            pass
        elif role == "outline_subsection":
            pl = usr[usr.find(A_ENRICH_PAPERS):usr.find(A_ENRICH_FORMAT)] if A_ENRICH_PAPERS in usr else ""
            enrich.append((seq, comp, RE_PAPER_TITLE.findall(pl)))
            i = usr.find(SIG_OUTLINE_SUBSECTION)
            enrich_probe[seq] = usr[i:i + 4000]
        elif role == "outline_merge":
            merge = (seq, comp)
        elif role == "outline_edit":
            edit = (seq, _between(usr, A_EDIT_DRAFT_OUTLINE), comp.replace("<format>\n", ""))
        elif role == "draft":
            m = RE_DRAFT_TARGET.search(usr)
            pl = usr[usr.find(A_WRITE_PAPERS):usr.find(A_INSTRUCTION)] if A_WRITE_PAPERS in usr else ""
            o = _between(usr, A_WRITE_OUTLINE, A_WRITE_PAPERS)
            drafts[seq] = {"sub": m.group(1) if m else None, "sec": m.group(2) if m else None,
                           "text": comp, "papers": RE_PAPER_TITLE.findall(pl), "outline_in_prompt": o}
        elif role == "refine_citation":
            pl = usr[usr.find(A_WRITE_PAPERS):usr.find(A_CIT_SUBSECTION)] if A_WRITE_PAPERS in usr else ""
            blk = _between(usr, A_CIT_SUBSECTION, A_INSTRUCTION)
            cits[seq] = {"in_key": tkey(blk), "text": comp, "papers": RE_PAPER_TITLE.findall(pl)}
        elif role == "refine_coherence":
            tgt = usr.split(A_COH_TARGET, 1)[1].rsplit("\n---", 1)[0] if A_COH_TARGET in usr else ""
            prv = usr.split(A_COH_PREV, 1)[1].split("\n---", 1)[0] if A_COH_PREV in usr else ""
            nxt = usr.split(A_COH_FOLLOWING, 1)[1].split("\n---", 1)[0] if A_COH_FOLLOWING in usr else ""
            cohs[seq] = {"in_key": tkey(tgt), "text": comp, "prev_key": tkey(prv), "next_key": tkey(nxt),
                         "t_req": r.get("t_req")}


    outline_initial_md = ""
    enrich_seq_by_section: dict[int, int] = {}
    if merge:
        _t, msecs, mdescs = as_extract_title_sections_descriptions(merge[1])
        for i, (nm, ds) in enumerate(zip(msecs, mdescs)):
            needle = f"{SIG_OUTLINE_SUBSECTION}{nm}.\nThe description of {nm}: {ds}\n"
            hits = [s for s, probe in enrich_probe.items() if needle in probe]
            if len(hits) == 1:
                enrich_seq_by_section[i] = hits[0]
        comp_by_seq = {s: c for s, c, _p in enrich}
        ordered = [comp_by_seq.get(enrich_seq_by_section.get(i, -1), "") for i in range(len(msecs))]
        outline_initial_md = as_process_outlines(merge[1], ordered)
    else:
        pass

    outline_final_md = edit[2] if edit else ""

    parsed_final = as_parse_outline(outline_final_md)
    outline_initial = _outline_nodes(as_parse_outline(outline_initial_md))
    outline_final = _outline_nodes(parsed_final)

    draft_by_name: dict[str, list[int]] = {}
    for s, d in drafts.items():
        draft_by_name.setdefault(d["sub"] or "", []).append(s)
    dkey_to_seq: dict[str, int] = {tkey(d["text"]): s for s, d in drafts.items()}
    ckey_to_seq: dict[str, int] = {tkey(c["text"]): s for s, c in cits.items()}
    cit_of_draft: dict[int, int] = {}
    for s, c in cits.items():
        ds = dkey_to_seq.get(c["in_key"])
        if ds is not None:
            cit_of_draft[ds] = s
    coh_of_cit: dict[int, int] = {}
    for s, h in cohs.items():
        cs = ckey_to_seq.get(h["in_key"])
        if cs is not None:
            coh_of_cit[cs] = s


    per_sub: list[dict] = []
    draft_c: list[list[str | None]] = []
    cit_c: list[list[str | None]] = []
    coh_c: list[list[str | None]] = []
    section_evidence: dict[str, list[str]] = {}
    for i, secname in enumerate(parsed_final["sections"]):
        draft_c.append([]); cit_c.append([]); coh_c.append([])
        for j, sub in enumerate(parsed_final["subsections"][i]):
            sid = f"{i + 1}.{j + 1}"
            ds = (draft_by_name.get(sub) or [None])[0]
            cs = cit_of_draft.get(ds) if ds is not None else None
            hs = coh_of_cit.get(cs) if cs is not None else None
            d = drafts.get(ds, {})
            c = cits.get(cs, {})
            h = cohs.get(hs, {})
            draft_c[i].append(d.get("text"))
            cit_c[i].append(c.get("text"))
            coh_c[i].append(h.get("text"))
            if d.get("papers"):
                section_evidence[sid] = d["papers"]
            per_sub.append({
                "id": sid, "section": secname, "title": sub,
                "description": (parsed_final["subsection_descriptions"][i][j]
                                if j < len(parsed_final["subsection_descriptions"][i]) else None),
                "draft_seq": ds, "refine_citation_seq": cs, "refine_coherence_seq": hs,
                "coherence_wave": 1 if j % 2 == 0 else 2,
                "n_papers_in_draft_prompt": len(d.get("papers") or []),
                "n_papers_in_citation_prompt": len(c.get("papers") or []),
                "draft_words": words(d.get("text")), "citation_words": words(c.get("text")),
                "coherence_words": words(h.get("text")),
                "draft_cite_spans": len(RE_CITE.findall(d.get("text") or "")),
                "citation_cite_spans": len(RE_CITE.findall(c.get("text") or "")),
                "coherence_cite_spans": len(RE_CITE.findall(h.get("text") or "")),
                "citation_pass_changed_text": nocite(d.get("text") or "") != nocite(c.get("text") or ""),
                "coherence_pass_changed_text": nocite(c.get("text") or "") != nocite(h.get("text") or ""),
                "cit_titles_removed": len(cited_titles(d.get("text") or "") - cited_titles(c.get("text") or "")),
                "cit_titles_added": len(cited_titles(c.get("text") or "") - cited_titles(d.get("text") or "")),
                "coh_titles_removed": len(cited_titles(c.get("text") or "") - cited_titles(h.get("text") or "")),
                "coh_titles_added": len(cited_titles(h.get("text") or "") - cited_titles(c.get("text") or "")),
                "cited_titles": len(cited_titles(h.get("text") or "")),
                "cited_titles_off_evidence": off_evidence(cited_titles(h.get("text") or ""), d.get("papers") or []),
                "cited_notation_spans": notation_spans(cited_titles(h.get("text") or "")),
                "coh_words_delta": words(h.get("text")) - words(c.get("text")),
            })
    papers_by_seq = {s: p for s, _c, p in enrich}
    for i, es in sorted(enrich_seq_by_section.items()):
        if papers_by_seq.get(es):
            section_evidence[f"outline:{i + 1}"] = papers_by_seq[es]

    wave_by_seq = _waves([(s, h["t_req"]) for s, h in cohs.items()])

    coh_seq_of: dict[str, int] = {p["id"]: p["refine_coherence_seq"] for p in per_sub}
    for i in range(len(parsed_final["sections"])):
        n = len(parsed_final["subsections"][i])
        even = [coh_c[i][k] if k % 2 == 0 else cit_c[i][k] for k in range(n)]
        for j in range(n):
            hs = coh_seq_of.get(f"{i + 1}.{j + 1}")
            if hs is None or hs not in cohs or n < 2:
                continue
            src = cit_c[i] if j % 2 == 0 else even
            if j == 0:
                exp_prev, exp_next = "", src[1]
            elif j == n - 1:
                exp_prev, exp_next = src[n - 2], ""
            else:
                exp_prev, exp_next = src[j - 1], src[j + 1]
            got = cohs[hs]
            if got["prev_key"] != tkey(exp_prev or "") or got["next_key"] != tkey(exp_next or ""):
                pass


    draft_report = as_generate_document(parsed_final, draft_c)
    raw_report = as_generate_document(parsed_final, cit_c)

    fj = _final_json(task_dir)
    if fj:
        try:
            d = json.load(open(fj))
        except Exception as e:
            pass

    tap = task_dir / "retrieval_tap.jsonl"
    tap_by_num: dict[int, int] = {}
    tap_total = 0
    tap_ids_by_query: dict[tuple[int, str], list[str]] = {}
    topic_pool: list[str] = []
    if tap.exists():
        for r in _iter_calls(tap):
            tap_total += 1
            num = r.get("num")
            tap_by_num[num] = tap_by_num.get(num, 0) + 1
            tap_ids_by_query.setdefault((num, ws(r.get("query"))), list(r.get("ids") or []))
            if num == 1500 or (not topic_pool and tap_total == 1):
                topic_pool = list(r.get("ids") or [])

    section_evidence_pmids: dict[str, list[str]] = {}
    for p in per_sub:
        ids = tap_ids_by_query.get((60, ws(p["description"])))
        if ids:
            section_evidence_pmids[p["id"]] = ids
    merge_descs = as_extract_title_sections_descriptions(merge[1])[2] if merge else []
    for i, ds in enumerate(merge_descs):
        ids = tap_ids_by_query.get((50, ws(ds)))
        if ids:
            section_evidence_pmids[f"outline:{i + 1}"] = ids

    refinement_passes = [
        {"role": "refine_citation", "seq_range": _rng(cits)},
        {"role": "refine_coherence", "seq_range": _rng({s: 1 for s in cohs if wave_by_seq.get(s) == 1})},
        {"role": "refine_coherence", "seq_range": _rng({s: 1 for s in cohs if wave_by_seq.get(s) == 2})},
    ]

    if role_counts.get("probe"):
        pass

    out = {
        "outline_initial": outline_initial,
        "outline_final": outline_final,
        "section_evidence": section_evidence,
        "section_evidence_pmids": section_evidence_pmids,
        "draft_report": draft_report,
        "refinement_passes": [{"role": p.get("role"), "seq_range": p.get("seq_range")} for p in refinement_passes],
        "per_subsection": [{"draft_seq": p.get("draft_seq")} for p in per_sub],
    }
    if keep_intermediate:
        out["_citation_pass_report"] = raw_report
    return out


def _rng(d) -> list[int] | None:
    ks = sorted(d)
    return [ks[0], ks[-1]] if ks else None


def _outline_nodes(parsed: dict) -> list[dict]:
    nodes = []
    for i, sec in enumerate(parsed["sections"]):
        sid = str(i + 1)
        nodes.append({"id": sid, "title": sec, "level": 1, "parent": None,
                      "description": parsed["section_descriptions"][i] if i < len(parsed["section_descriptions"]) else None})
        for j, sub in enumerate(parsed["subsections"][i] if i < len(parsed["subsections"]) else []):
            nodes.append({"id": f"{i + 1}.{j + 1}", "title": sub, "level": 2, "parent": sid,
                          "description": (parsed["subsection_descriptions"][i][j]
                                          if j < len(parsed["subsection_descriptions"][i]) else None)})
    return nodes


def _waves(items: list[tuple[int, float | None]]) -> dict[int, int]:
    ts = [(s, t) for s, t in items if isinstance(t, (int, float))]
    if len(ts) < 2:
        return {}
    vals = sorted(t for _s, t in ts)
    gaps = [(vals[i + 1] - vals[i], i) for i in range(len(vals) - 1)]
    gap, idx = max(gaps)
    if gap < 5.0:
        return {s: 1 for s, _t in ts}
    cut = (vals[idx] + vals[idx + 1]) / 2
    return {s: (1 if t <= cut else 2) for s, t in ts}
