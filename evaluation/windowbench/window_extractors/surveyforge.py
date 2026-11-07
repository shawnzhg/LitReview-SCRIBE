"""Window extractor for SurveyForge task directories: rebuilds the outline, per-subsection evidence,
first complete draft and refinement passes, and maps the final reference list to PMIDs."""

from __future__ import annotations

import json
import re
from pathlib import Path

SIG_PROBE = "hello"
SIG_OUTLINE_ROUGH_HEAD = "You are an AI assistant tasked with creating a concise, high-level, comprehensive and original academic survey outline for"
SIG_OUTLINE_ROUGH_TAIL = "</format>\n\nThe outline:"
SIG_OUTLINE_MERGE_TAIL = "Ensure all terms are fully written out without abbreviations or acronyms."
SIG_OUTLINE_MERGE_BODY = "2. AI-generated outlines from subsets of papers related to"
SIG_OUTLINE_SECOND_HEAD = "You are an expert in artificial intelligence writing a comprehensive outline of the survey about"
SIG_OUTLINE_SECOND_BODY = "You have created the following overall outline:"
SIG_OUTLINE_SECOND_SECTION = "You need to enrich the section **"
SIG_OUTLINE_REVISE_HEAD = "You are an expert in artificial intelligence tasked with refining a comprehensive survey outline about"
SIG_DRAFT_HEAD = "You are writing the subsection"
SIG_DRAFT_BODY = "The overall outline of your survey is as follows:"
SIG_PAPER_BLOCK = "Below are a list of papers for references:"
SIG_REFINE_CIT_TAIL = "Only return the subsection with correct citations:"
SIG_REFINE_CIT_BODY = "You have written a subsection below:"
SIG_REFINE_LCE_HEAD2 = "Now you need to help to refine one of the subsection to improve th ecoherence of your survey."
SIG_REFINE_LCE_TAIL = "The subsection content:"
SIG_REFINE_LCE_TARGET = "Subsection to Refine:"

R_REFINE_CIT = "refine_citation"
R_REFINE_COH = "refine_coherence"


_FINAL_JSON_SKIP = {"egress_proof.json", "listener_census.json", "pool_health.json"}

MIN_ANCHORS = 3
ANCHOR_MIN_RATIO = 0.8
ANCHOR_MARGIN = 3


def _ws(s: str | None) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def user_turn(rec: dict) -> str:
    ms = rec.get("messages") or []
    us = [m.get("content") or "" for m in ms if m.get("role") == "user"]
    return us[-1] if us else (rec.get("prompt") or "")


def classify_call(prompt: str) -> str:
    p = prompt or ""
    head = p.lstrip()
    tail = p.rstrip()
    if _ws(p) == SIG_PROBE:
        return "probe"
    if head.startswith(SIG_DRAFT_HEAD) and SIG_DRAFT_BODY in p:
        return "subsection_writing"
    if tail.endswith(SIG_REFINE_CIT_TAIL) and SIG_REFINE_CIT_BODY in p:
        return "check_citation"
    if tail.endswith(SIG_REFINE_LCE_TAIL) and SIG_REFINE_LCE_HEAD2 in p:
        return "lce"
    if head.startswith(SIG_OUTLINE_SECOND_HEAD) and SIG_OUTLINE_SECOND_BODY in p:
        return "subsection_outline"
    if head.startswith(SIG_OUTLINE_REVISE_HEAD):
        return "edit_final_outline"
    if head.startswith(SIG_OUTLINE_ROUGH_HEAD):
        if tail.endswith(SIG_OUTLINE_ROUGH_TAIL):
            return "rough_outline"
        if tail.endswith(SIG_OUTLINE_MERGE_TAIL) and SIG_OUTLINE_MERGE_BODY in p:
            return "merge_outline"
    return "unknown"


def iter_calls(path: Path):
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _strip_fmt(s: str) -> str:
    return (s or "").replace("<format>", "").replace("</format>", "")


def _key(s: str) -> str:
    return _ws(_strip_fmt(s))


def _blank_cites(s: str) -> str:
    return _ws(re.sub(r"\[[^\]\n]*\]", "[]", _strip_fmt(s)))


def _between(text: str, start: str, end: str) -> str | None:
    i = text.find(start)
    if i < 0:
        return None
    i += len(start)
    j = text.find(end, i)
    return text[i:j] if j >= 0 else text[i:]


def parse_outline_markdown(text: str) -> list[dict]:
    nodes: list[dict] = []
    lines = (text or "").split("\n")
    cur_section = None
    cur_id = None
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        if line.startswith("### "):
            title = line[4:].strip()
            nodes.append({"id": title.split(" ", 1)[0] if title[:1].isdigit() else title,
                          "title": title, "level": 2, "parent": cur_section, "description": "",
                          "bullets": []})
            cur_id = len(nodes) - 1
        elif line.startswith("## "):
            title = line[3:].strip()
            cur_section = title.split(" ", 1)[0] if title[:1].isdigit() else title
            nodes.append({"id": cur_section, "title": title, "level": 1, "parent": None,
                          "description": "", "bullets": []})
            cur_id = len(nodes) - 1
        elif line.startswith("# "):
            nodes.append({"id": "TITLE", "title": line[2:].strip(), "level": 0, "parent": None})
            cur_id = None
        elif line.startswith("Description:") and cur_id is not None:
            nodes[cur_id]["description"] = line.split("Description:", 1)[1].strip()
        elif cur_id is not None and re.match(r"^\d+\.\s", line):
            nodes[cur_id]["bullets"].append(line.split(". ", 1)[1].strip())
    return nodes


def outline_leaves(nodes: list[dict]) -> list[dict]:
    return [n for n in nodes if n.get("level") == 2]


def duplicate_first_last_sections(markdown_content: str) -> str:
    pattern = r"(## \d+\.?\s*.*?(?=\n##|\Z))"
    sections = re.findall(pattern, markdown_content, re.DOTALL)
    if len(sections) < 2:
        return markdown_content
    out = markdown_content
    for sec, last in ((sections[0], False), (sections[-1], True)):
        num = re.search(r"## (\d+)", sec).group(1)
        title = sec.split("\n")[0].strip()
        content = "\n".join(sec.split("\n")[1:]).strip()
        new = (f"{title}\n{content}\n\n### {num}.1 {title.split(maxsplit=2)[-1]}\nDescription: {content}\n"
               + ("" if last else "\n"))
        out = out.replace(sec, new)
    return out


def remove_first_last_subsection_titles(markdown_content: str) -> str:
    subsections = re.findall(r"\n(### \d+\.\d+[^\n]*)\n", markdown_content)
    if len(subsections) < 2:
        return markdown_content
    out = re.sub(r"\n" + re.escape(subsections[0]) + r"\n", "\n", markdown_content)
    out = re.sub(r"\n" + re.escape(subsections[-1]) + r"\n", "\n", out)
    return re.sub(r"\n\n\n+", "\n\n", out)


def outline_files(task_dir: Path) -> dict:
    wd = sorted(task_dir.glob("outlines_with_des_*.txt"))
    wo = sorted(task_dir.glob("outlines_without_des_*.txt"))
    return {
        "with_des": wd[-1] if wd else None,
        "without_des": wo[-1] if wo else None,
        "n_with_des": len(wd),
        "n_without_des": len(wo),
    }


def final_json_path(task_dir: Path) -> Path | None:
    cands = [p for p in task_dir.glob("*.json")
             if not p.name.startswith("_") and p.name not in _FINAL_JSON_SKIP]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


def load_retrieval_tap(task_dir: Path) -> dict:
    p = task_dir / "retrieval_tap.jsonl"
    rows = [json.loads(l) for l in open(p, errors="replace")] if p.exists() else []
    subsection = [r for r in rows if r.get("method") == "retrieve_id" and r.get("num") == 100]
    topic_pool = [r for r in rows if r.get("method") == "retrieve_id" and r.get("num") == 1500]
    outline_sub = [r for r in rows if r.get("method") == "retrieve_id"
                   and r.get("num") not in (100, 1500)]
    survey_db = [r for r in rows if r.get("method") == "get_ids_from_query"]
    cite = [r for r in rows if r.get("method") == "retrieve_id4citation"]
    union: list[str] = []
    seen = set()
    for r in subsection:
        for i in r.get("ids") or []:
            if i not in seen:
                seen.add(i)
                union.append(i)
    return {"rows": rows, "n_rows": len(rows), "subsection": subsection, "topic_pool": topic_pool,
            "outline_subqueries": outline_sub, "survey_db": survey_db, "citation": cite,
            "writer_pool_union": union}


def final_reference_maps(task_dir: Path) -> dict:
    fj = final_json_path(task_dir)
    refs_by_n: dict[str, str] = {}
    survey = ""
    if fj is not None:
        try:
            d = json.load(open(fj, errors="replace"))
            survey = d.get("survey") or ""
            refs_by_n = {str(k): str(v) for k, v in (d.get("reference") or {}).items() if v}
        except Exception:
            pass
    text = survey
    if not text:
        mds = [p for p in task_dir.glob("*.md") if not p.name.startswith("_")]
        if mds:
            text = max(mds, key=lambda p: p.stat().st_size).read_text(errors="replace")
    title_by_n = {}
    if "## References" in text:
        for n, t in re.findall(r"^\[(\d+)\]\s*(.+?)\s*$", text.split("## References", 1)[1], re.M):
            title_by_n[n] = t.strip()
    title_to_pmid = {t: refs_by_n[n] for n, t in title_by_n.items() if n in refs_by_n}
    return {"final_json": fj, "survey_text": text, "refs_by_n": refs_by_n,
            "title_by_n": title_by_n, "title_to_pmid": title_to_pmid}


def extract(task_dir: Path) -> dict:
    task_dir = Path(task_dir)
    calls_path = task_dir / "_calls.jsonl"

    merge: list[dict] = []
    second: list[dict] = []
    drafts: list[dict] = []
    checks: list[dict] = []
    lces: list[dict] = []

    for rec in iter_calls(calls_path):
        seq = rec.get("seq")
        u = user_turn(rec)
        tpl = classify_call(u)
        comp = rec.get("completion") or ""
        ok = rec.get("status") == 200 and comp != ""
        if tpl == "probe":
            pass
        elif tpl == "rough_outline":
            pass
        elif tpl == "merge_outline":
            merge.append({"seq": seq, "ok": ok, "outline": _strip_fmt(comp)})
        elif tpl == "subsection_outline":
            sec = _between(u, SIG_OUTLINE_SECOND_SECTION, "**, described as:") or ""
            pl = _between(u, "publication dates for this section:", "\n\n2. **Titles") or ""
            second.append({"seq": seq, "ok": ok, "section_hint": _ws(sec),
                           "n_papers_in_prompt": len(re.findall(r"paper_title:\s*", pl)),
                           "outline": _strip_fmt(comp)})
        elif tpl == "edit_final_outline":
            pass
        elif tpl == "subsection_writing":
            m = re.match(r'You are writing the subsection "(.*?)" under the section "(.*?)"', u.lstrip(),
                         re.S)
            papers_blk = _between(u, SIG_PAPER_BLOCK, "\n<instruction>") or ""
            drafts.append({
                "seq": seq, "ok": ok,
                "subsection": m.group(1) if m else None,
                "section": m.group(2) if m else None,
                "outline_in_prompt": (_between(u, "The overall outline of your survey is as follows:",
                                               "\n" + SIG_PAPER_BLOCK) or "").strip().strip("-").strip(),
                "paper_titles": [t.strip() for t in re.findall(r"paper_title:\s*(.+?)\s*\n", papers_blk)],
                "description": _ws(_between(u, "Subsection Focus:", "Core Requirements:") or ""),
                "draft": _strip_fmt(comp),
            })
        elif tpl == "check_citation":
            head = (u.split(SIG_REFINE_CIT_BODY, 1)[1].split("\n<instruction>", 1)[0]
                    if SIG_REFINE_CIT_BODY in u else "")
            body = head.split("---", 1)[1].rsplit("---", 1)[0] if head.count("---") >= 2 else head
            papers_blk = _between(u, SIG_PAPER_BLOCK, SIG_REFINE_CIT_BODY) or ""
            checks.append({"seq": seq, "ok": ok, "input": body, "output": _strip_fmt(comp),
                           "paper_titles": [t.strip()
                                            for t in re.findall(r"paper_title:\s*(.+?)\s*\n", papers_blk)]})
        elif tpl == "lce":
            tgt = _between(u, SIG_REFINE_LCE_TARGET, SIG_REFINE_LCE_TAIL) or ""
            tgt = tgt.split("---", 1)[1].rsplit("---", 1)[0] if tgt.count("---") >= 2 else tgt
            lces.append({
                "seq": seq, "ok": ok,
                "prev": _ws(_between(u, "Previous Subsection:", "Following Subsection:") or "").strip("- "),
                "next": _ws(_between(u, "Following Subsection:", SIG_REFINE_LCE_TARGET) or "").strip("- "),
                "input": tgt, "output": _strip_fmt(comp),
            })
        else:
            pass

    files = outline_files(task_dir)
    committed_md = files["with_des"].read_text(errors="replace") if files["with_des"] else ""
    if not committed_md and merge and second:
        committed_md = "\n\n".join([merge[-1]["outline"]] + [s["outline"] for s in second])
    outline_initial = parse_outline_markdown(committed_md)

    prompt_outlines = {d["outline_in_prompt"] for d in drafts if d["outline_in_prompt"]}
    outline_final_md = (sorted(prompt_outlines, key=len)[-1] if prompt_outlines
                        else duplicate_first_last_sections(committed_md))
    outline_final = parse_outline_markdown(outline_final_md)

    leaves_final = outline_leaves(outline_final)
    leaf_titles = [n["title"] for n in leaves_final]
    j_within_section: dict[str, int] = {}
    counter: dict[str | None, int] = {}
    for n in leaves_final:
        p = n["parent"]
        j_within_section[n["title"]] = counter.get(p, 0)
        counter[p] = counter.get(p, 0) + 1

    first_draft: dict[str, dict] = {}
    draft_retries: dict[str, int] = {}
    for d in sorted(drafts, key=lambda x: x["seq"]):
        t = d["subsection"]
        if t is None:
            continue
        if d["ok"] and t not in first_draft:
            first_draft[t] = d
        elif not d["ok"]:
            draft_retries[t] = draft_retries.get(t, 0) + 1

    tap = load_retrieval_tap(task_dir)
    sn = tap["subsection"]
    refs = final_reference_maps(task_dir)
    t2p = refs["title_to_pmid"]

    def _anchor_score(titles: list[str], ids: list[str]) -> tuple[int, int]:
        by_title: dict[str, list[int]] = {}
        for k, t in enumerate(titles):
            by_title.setdefault(t, []).append(k)
        chk = hit = 0
        for t, ks in by_title.items():
            if len(ks) != 1 or t not in t2p:
                continue
            chk += 1
            hit += (t2p[t] == ids[ks[0]])
        return chk, hit

    section_evidence: dict[str, list[str]] = {}
    evidence_intended: dict[str, list[str]] = {}
    for idx, title in enumerate(leaf_titles):
        d = first_draft.get(title)
        own_ids = list(sn[idx]["ids"]) if idx < len(sn) else []
        evidence_intended[title] = [i.split(".")[-1] for i in own_ids]
        if d is None:
            continue
        titles = d["paper_titles"]
        scored = []
        for k, row in enumerate(sn):
            ids_k = list(row.get("ids") or [])
            if len(ids_k) != len(titles):
                continue
            chk, hit = _anchor_score(titles, ids_k)
            if chk >= MIN_ANCHORS:
                scored.append((hit, chk, k))
        scored.sort(reverse=True)
        ok_resolve = bool(scored) and scored[0][0] >= ANCHOR_MIN_RATIO * scored[0][1]
        if ok_resolve and len(scored) > 1 and scored[0][0] - scored[1][0] < ANCHOR_MARGIN:
            ok_resolve = False
        if not ok_resolve:
            section_evidence[title] = titles
            continue
        hit, chk, k = scored[0]
        ids = list(sn[k]["ids"])
        section_evidence[title] = [i.split(".")[-1] for i in ids]

    draft_key_to_sub = {_key(d["draft"]): t for t, d in first_draft.items()}
    all_draft_keys = {_key(d["draft"]): d["subsection"] for d in drafts if d["ok"]}
    check_by_sub: dict[str, dict] = {}
    for c in sorted(checks, key=lambda x: x["seq"]):
        t = draft_key_to_sub.get(_key(c["input"])) or all_draft_keys.get(_key(c["input"]))
        if t is None:
            continue
        check_by_sub.setdefault(t, c)
    check_key_to_sub = {_key(c["output"]): t for t, c in check_by_sub.items()}
    lce_by_sub: dict[str, dict] = {}
    for l in sorted(lces, key=lambda x: x["seq"]):
        t = check_key_to_sub.get(_key(l["input"]))
        if t is None:
            continue
        lce_by_sub.setdefault(t, l)

    even_seqs = [l["seq"] for t, l in lce_by_sub.items() if j_within_section.get(t, 0) % 2 == 0]
    odd_seqs = [l["seq"] for t, l in lce_by_sub.items() if j_within_section.get(t, 0) % 2 == 1]
    refinement_passes = [
        {"role": R_REFINE_CIT, "seq_range": _range([c["seq"] for c in checks])},
        {"role": R_REFINE_COH, "seq_range": _range(even_seqs)},
        {"role": R_REFINE_COH, "seq_range": _range(odd_seqs)},
    ]

    draft_report = _assemble(outline_final, {t: d["draft"] for t, d in first_draft.items()})

    final_bodies = _final_bodies(refs["survey_text"])
    for i, title in enumerate(leaf_titles):
        key = title
        if i == 0 or i == len(leaf_titles) - 1:
            parent = leaves_final[i]["parent"]
            key = "SEC:" + parent if parent else title
        fin = final_bodies.get(key)
        if fin is None:
            continue
        rc = check_by_sub.get(title, {}).get("output", "")
        lc = lce_by_sub.get(title, {}).get("output", "")
        if _blank_cites(fin) == _blank_cites(rc):
            pass
        elif _blank_cites(fin) == _blank_cites(lc):
            pass
        else:
            pass


    for t, c in check_by_sub.items():
        d = first_draft.get(t)
        if d is None:
            continue
    for t, l in lce_by_sub.items():
        if j_within_section.get(t, 0) % 2 == 1 and t in check_by_sub:
            pass



    for m in re.findall(r"\[([0-9;,\s]+)\]", refs["survey_text"].split("## References")[0]):
        for part in re.split(r"[;,]", m):
            part = part.strip()
            if part.isdigit():
                pass



    return {
        "outline_initial": outline_initial,
        "outline_final": outline_final,
        "section_evidence": section_evidence,
        "section_evidence_intended": evidence_intended,
        "draft_report": draft_report,
        "refinement_passes": [{"role": p.get("role"), "seq_range": p.get("seq_range")} for p in refinement_passes],
    }


def _range(seqs: list[int]) -> str | None:
    if not seqs:
        return None
    return f"{min(seqs)}-{max(seqs)}"


def _assemble(outline_nodes: list[dict], body_by_sub: dict[str, str]) -> str:
    doc = []
    for n in outline_nodes:
        if n["level"] == 0:
            doc.append(f"# {n['title']}\n")
        elif n["level"] == 1:
            doc.append(f"## {n['title']}\n")
        else:
            doc.append(f"### {n['title']}\n")
            b = body_by_sub.get(n["title"])
            if b is not None:
                doc.append(b + "\n")
    return remove_first_last_subsection_titles("\n".join(doc))


def _final_bodies(survey_text: str) -> dict[str, str]:
    body = (survey_text or "").split("## References")[0]
    out: dict[str, str] = {}
    for part in re.split(r"\n(?=#{2,3} )", body):
        head, _, rest = part.partition("\n")
        head = head.strip()
        if head.startswith("### "):
            out[head[4:].strip()] = rest.strip()
        elif head.startswith("## "):
            out["SEC:" + head[3:].strip().split(" ", 1)[0]] = rest.strip()
    return out


