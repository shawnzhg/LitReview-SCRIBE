"""Window extractor for SurveyG task directories: rebuilds the outline, per-subsection evidence,
first complete draft and refinement passes from the call log, checkpoints and output files."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

SIG_OUTLINE_REVISE = "TASK: Regenerate the outline addressing all feedback"
SIG_OUTLINE_GEN = "You are creating a comprehensive literature review outline for"
SIG_OUTLINE_REVIEW = "carefully evaluating this literature review outline"
SIG_DIGEST = "You are a research analyst synthesizing a body of literature"
SIG_DRAFT = "Write a comprehensive literature review subsection titled"
SIG_REVIEW = "carefully evaluating this literature review subsection"
SIG_EVIDENCE_SELECT = "filtering retrieved papers for a literature review subsection"
SIG_REFINE = "Improve the following literature review subsection based on evaluation feedback"

SIGNATURES: list[tuple[str, str]] = [
    ("outline_revise", SIG_OUTLINE_REVISE),
    ("outline_gen", SIG_OUTLINE_GEN),
    ("outline_review", SIG_OUTLINE_REVIEW),
    ("digest", SIG_DIGEST),
    ("draft", SIG_DRAFT),
    ("review", SIG_REVIEW),
    ("evidence_select", SIG_EVIDENCE_SELECT),
    ("refine", SIG_REFINE),
]
BRIEF_ROLE = {
    "outline_gen": "outline_gen",
    "outline_revise": "outline_revise",
    "outline_review": "review",
    "digest": "digest",
    "draft": "draft",
    "review": "review",
    "evidence_select": "evidence_select",
    "refine": "refine_coherence",
    "other": "other",
}

M_PAPERS = "**Papers to reference (sorted chronologically):**"
M_PAPERS_END = "Each paper follows this format"
M_DRAFT_COMM = "**Community summaries:**"
M_DRAFT_DEV = "**Development directions:**"
M_RETRIEVED = "**Retrieved Papers**"
M_ADDED = "**Additional Papers Retrieved**:"
M_PREV_OUTLINE = "PREVIOUS OUTLINE:"


WS = re.compile(r"\s+")
WORD_RE = re.compile(r"\b\w+\b")
CITE_TEX = re.compile(r"\\cite[tp]?\*?\{([^}]*)\}")
CITE_MD = re.compile(r"\[((?:paper[0-9A-Za-z_]+\s*[,;]?\s*)+)\]")
PAPER_KEY = re.compile(r"\[(paper\w+)\]")
PAPER_HEAD = re.compile(r"^\[(paper\w+)\]\s*(.*?)\s*\((\d{4})\)\s*$", re.M)
PAPER_ENTRY = re.compile(r"^\[(paper\w+)\]\s*(.*?)\s*\((\d{4})\)\s*\nSummary:\s*(.*?)(?=\n\s*\n|\Z)", re.M | re.S)
DRAFT_TITLE = re.compile(r'subsection titled\s+"(.+?)"\s+in LaTeX format', re.S)
SUB_TITLE = re.compile(r"\*\*Subsection Title\*\*:\s*(.+)")
DIGEST_TOPIC = re.compile(r'survey topic\s+"(.+?)"')
GEN_TOPIC = re.compile(r'outline for:\s*"(.+?)"')
VERDICT = re.compile(r"PASS/FAIL:\s*\**\s*(PASS|FAIL)", re.I)
PMID_IN_NAME = re.compile(r"PMID[:_]?(\d+)")
FENCE_OPEN = re.compile(r"^\s*```(?:latex|tex)?\s*")
FENCE_CLOSE = re.compile(r"\s*```\s*$")
THINK = re.compile(r"<think>.*?</think>", re.S)


def ws(s: str | None) -> str:
    return WS.sub(" ", s or "").strip()


def haystack(system: str, user: str) -> str:
    return ws(system[:400]) + " || " + ws(user[:3000]) + " ~~ " + ws(user[-2000:])


def classify(system: str, user: str) -> str:
    hay = haystack(system, user)
    for role, sig in SIGNATURES:
        if sig in hay:
            return role
    return "other"


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


def _block(text: str, start: str, ends: tuple[str, ...]) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = len(text)
    for e in ends:
        k = text.find(e, i)
        if 0 <= k < j:
            j = k
    return text[i:j]


def _papers_block(user: str) -> str:
    return _block(user, M_PAPERS, (M_PAPERS_END,)).strip()


def _strip_fence(t: str) -> str:
    t = (t or "").strip()
    t = FENCE_OPEN.sub("", t)
    t = FENCE_CLOSE.sub("", t)
    return t.strip()


def _tex_body(completion: str) -> str:
    return _strip_fence(THINK.sub("", completion or ""))


def _tex_to_md(body: str, heading: str | None = None) -> str:
    body = _tex_body(body)
    body = re.sub(r"^\s*\\subsection\*?\{[^}]*\}\s*", "", body)
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = CITE_TEX.sub(lambda m: "[" + ",".join(x.strip() for x in m.group(1).split(",") if x.strip()) + "]", body)
    body = re.sub(r"\\(?:textbf|textit|emph)\{([^}]*)\}", r"\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return ((heading + "\n\n") if heading else "") + body.strip()


def _words(text: str) -> int:
    t = CITE_MD.sub(" ", text or "")
    t = re.sub(r"\\label\{[^}]*\}", " ", t)
    t = re.sub(r"\\(?:sub)*section\*?\{([^}]*)\}", r"\1", t)
    t = CITE_TEX.sub(" ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    return len(WORD_RE.findall(t))


def _cite_keys(text: str) -> list[str]:
    keys = [k for m in CITE_TEX.findall(text or "") for k in (x.strip() for x in m.split(",")) if k]
    keys += [k for m in CITE_MD.findall(text or "") for k in (x.strip() for x in re.split(r"[,;]", m)) if k]
    return list(dict.fromkeys(keys))


def _sentences(text: str) -> list[str]:
    t = ws(re.sub(r"\\[a-zA-Z]+\*?\{[^}]*\}", " ", text or ""))
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 20]


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, ws(a), ws(b)).ratio()


def _parse_outline_json(text: str):
    s = _strip_fence(text)
    try:
        return json.loads(s)
    except Exception:
        i, j = s.find("["), s.rfind("]")
        if 0 <= i < j:
            try:
                return json.loads(s[i : j + 1])
            except Exception:
                return None
    return None


def _outline_nodes(og) -> list[dict]:
    nodes: list[dict] = []
    if not isinstance(og, list):
        return nodes
    for i, sec in enumerate(og):
        if not isinstance(sec, dict):
            continue
        num = str(sec.get("section_number") or i + 1).strip()
        nodes.append({"id": num, "title": ws(sec.get("section_title") or sec.get("title") or ""), "level": 1,
                      "parent": None, "description": sec.get("section_focus"),
                      "proof_ids": [str(p) for p in (sec.get("proof_ids") or [])]})
        for j, sub in enumerate(sec.get("subsections") or []):
            if isinstance(sub, str):
                sub = {"title": sub}
            snum = str(sub.get("number") or f"{num}.{j + 1}").strip()
            nodes.append({"id": snum, "title": ws(sub.get("title") or sub.get("subsection_title") or ""), "level": 2,
                          "parent": num, "description": sub.get("subsection_focus"),
                          "proof_ids": [str(p) for p in (sub.get("proof_ids") or [])]})
    return nodes


def extract(task_dir: Path) -> dict:
    task_dir = Path(task_dir)
    calls_path = task_dir / "_calls.jsonl"
    if not calls_path.exists():
        return {
            "outline_initial": [],
            "outline_final": [],
            "section_evidence": {},
            "draft_report": "",
            "refinement_passes": [],
        }

    data: dict = {}
    dp = task_dir / "literature_review_data.json"
    if dp.exists():
        try:
            data = json.load(open(dp))
        except Exception as e:
            pass
    key2pmid: dict[str, str] = {}
    for fname, key in (data.get("citations_map") or {}).items():
        m = PMID_IN_NAME.search(str(fname))
        key2pmid[str(key)] = m.group(1) if m else str(fname)
    final_by_title: dict[str, str] = {ws(k): v for k, v in (data.get("subsections") or {}).items()}

    crawl_abs: dict[str, str] = {}
    cp = task_dir / "_app_info" / "crawl_papers.json"
    if cp.exists():
        try:
            crawl = json.load(open(cp))
            for p in crawl:
                m = PMID_IN_NAME.search(str(p.get("id") or ""))
                if m:
                    crawl_abs[m.group(1)] = ws(p.get("abstract") or "")
        except Exception as e:
            pass
    layer_counts: dict[str, int] = {}
    gp = task_dir / "_app_info" / "paper_citation_graph.json"
    if gp.exists():
        try:
            g = json.load(open(gp))
            for n in g.get("nodes") or []:
                layer_counts[str(n.get("layer"))] = layer_counts.get(str(n.get("layer")), 0) + 1
        except Exception as e:
            pass
    else:
        pass

    disk_outline = None
    op = task_dir / "survey_outline_gpt.json"
    if op.exists():
        try:
            disk_outline = json.load(open(op))
        except Exception as e:
            pass

    calls_total = 0
    roles_seq: list[tuple[int, str]] = []
    sig_nonmatch_by_role: dict[str, dict[str, int]] = {name: {} for name, _ in SIGNATURES}
    first_llm_t = None
    last_seq = -1
    topic = None
    outline_calls: list[dict] = []
    outline_reviews: list[dict] = []
    events: list[dict] = []

    for r in _iter_calls(calls_path):
        calls_total += 1
        seq = int(r.get("seq", calls_total - 1))
        last_seq = max(last_seq, seq)
        if first_llm_t is None:
            first_llm_t = r.get("t")
        if r.get("status") not in (None, 200) or r.get("error"):
            pass
        sysm, user = _split_messages(r)
        hay = haystack(sysm, user)
        role = classify(sysm, user)
        roles_seq.append((seq, role))
        for name, sig in SIGNATURES:
            if sig in hay:
                pass
            else:
                sig_nonmatch_by_role[name].setdefault(role, seq)
        comp = r.get("completion") or ""

        if role == "digest":
            blk = _papers_block(user)
            m = DIGEST_TOPIC.search(user)
            if m and topic is None:
                topic = m.group(1)
        elif role in ("outline_gen", "outline_revise"):
            if role == "outline_gen":
                m = GEN_TOPIC.search(user)
                if m:
                    topic = topic or m.group(1)
            og = _parse_outline_json(comp)
            outline_calls.append({"seq": seq, "role": role, "outline": og,
                                  "n_sections": len(og) if isinstance(og, list) else None,
                                  "reads_previous_outline": M_PREV_OUTLINE in user, "completion_chars": len(comp)})
        elif role == "outline_review":
            m = VERDICT.search(comp)
            outline_reviews.append({"seq": seq, "verdict": (m.group(1).upper() if m else None), "feedback": comp,
                                    "outline_in_prompt_tail": bool(re.search(r"proof_ids", user[-3000:]))})
        elif role == "draft":
            m = DRAFT_TITLE.search(user)
            blk = _papers_block(user)
            comm = _block(user, M_DRAFT_COMM, (M_DRAFT_DEV, M_PAPERS)).strip()
            dev = _block(user, M_DRAFT_DEV, (M_PAPERS,)).strip()
            events.append({"seq": seq, "role": role, "title": ws(m.group(1)) if m else "",
                           "papers": PAPER_ENTRY.findall(blk), "paper_keys": PAPER_KEY.findall(blk),
                           "n_papers_listed": len(PAPER_HEAD.findall(blk)),
                           "community_text": comm, "n_community_blocks": comm.count("Community summaries:"),
                           "dev_text": dev, "completion": comp})
        elif role in ("review", "evidence_select", "refine"):
            m = SUB_TITLE.search(user)
            ev = {"seq": seq, "role": role, "title": ws(m.group(1)) if m else ""}
            if role == "review":
                j = None
                try:
                    j = json.loads(_strip_fence(comp))
                except Exception:
                    j = None
                if isinstance(j, dict):
                    ev["score"] = j.get("overall_score")
                    ev["is_satisfactory"] = j.get("is_satisfactory")
                    ev["suggested_queries"] = [ws(q) for q in (j.get("suggested_queries") or [])]
                    ev["n_improvements"] = len(j.get("improvement_needed") or [])
                else:
                    ms = re.search(r'"overall_score"\s*:\s*([0-9.]+)', comp)
                    mi = re.search(r'"is_satisfactory"\s*:\s*(true|false)', comp)
                    ev["score"] = float(ms.group(1)) if ms else None
                    ev["is_satisfactory"] = (mi.group(1) == "true") if mi else None
                    ev["suggested_queries"] = re.findall(r'"([^"]{15,200})"', _block(comp, '"suggested_queries"', ("]",)))
                    ev["n_improvements"] = None
                    ev["json_parse_failed"] = True
            elif role == "evidence_select":
                cand = _block(user, M_RETRIEVED, ("**Task", "**IMPORTANT", "**Response", "**Output"))
                ev["candidate_keys"] = list(dict.fromkeys(PAPER_KEY.findall(cand)))
                ev["selected_keys"] = list(dict.fromkeys(PAPER_KEY.findall(comp)))
            else:
                added = _block(user, M_ADDED, ("**MANDATORY IMPROVEMENT ACTIONS", "**Improvement Instructions", "**Requirements"))
                ev["added_keys"] = list(dict.fromkeys(PAPER_KEY.findall(added)))
                ev["completion"] = comp
            events.append(ev)
        else:
            pass

    call_roles: list = []
    detailed: list = []
    for seq, role in roles_seq:
        for bucket, name in ((call_roles, BRIEF_ROLE.get(role, "other")), (detailed, role)):
            if bucket and bucket[-1][1] == name and bucket[-1][0][1] == seq - 1:
                bucket[-1][0][1] = seq
            else:
                bucket.append([[seq, seq], name])
    role_counts: dict[str, int] = {}
    for _, role in roles_seq:
        role_counts[role] = role_counts.get(role, 0) + 1

    gen = [c for c in outline_calls if c["role"] == "outline_gen"]
    outline_initial = _outline_nodes(gen[0]["outline"]) if gen else []
    if len(gen) != 1:
        pass
    last_outline_call = outline_calls[-1] if outline_calls else None
    if disk_outline is not None:
        outline_final = _outline_nodes(disk_outline)
    else:
        outline_final = _outline_nodes(last_outline_call["outline"]) if last_outline_call else []
    ote = task_dir / "outline_to_evaluate.txt"
    if ote.exists():
        txt = ote.read_text(errors="replace")

    subs_final = [n for n in outline_final if n["level"] == 2]
    groups: list[dict] = []
    cur = None
    for ev in events:
        if ev["role"] == "draft":
            cur = {"title": ev["title"], "draft_seq": ev["seq"], "draft_completion": ev["completion"],
                   "papers": ev["papers"], "paper_keys": ev["paper_keys"], "n_papers_listed": ev["n_papers_listed"],
                   "community_text": ev["community_text"], "n_community_blocks": ev["n_community_blocks"],
                   "dev_text": ev["dev_text"], "reviews": [], "selects": [], "refines": []}
            groups.append(cur)
        elif cur is None:
            pass
        else:
            if ev["title"] and ev["title"].lower() != cur["title"].lower():
                pass
            cur[{"review": "reviews", "evidence_select": "selects", "refine": "refines"}[ev["role"]]].append(ev)
    aligned_by_position = len(groups) == len(subs_final) and all(
        g["title"].lower() == n["title"].lower() for g, n in zip(groups, subs_final))
    for i, g in enumerate(groups):
        if aligned_by_position:
            g["id"] = subs_final[i]["id"]
            g["node"] = subs_final[i]
        else:
            best, bs = None, 0.0
            for n in subs_final:
                s = _sim(g["title"].lower(), n["title"].lower())
                if s > bs:
                    best, bs = n, s
            g["id"] = best["id"] if (best and bs >= 0.85) else f"unmatched:{g['title']}"
            g["node"] = best if (best and bs >= 0.85) else None
    if aligned_by_position:
        pass
    else:
        pass

    section_evidence: dict[str, list[str]] = {}
    section_evidence_keys: dict[str, list[str]] = {}
    section_proof_ids: dict[str, list[str]] = {}
    for g in groups:
        sid = g["id"]
        keys = list(dict.fromkeys(g["paper_keys"]))
        section_evidence_keys[sid] = keys
        section_evidence[sid] = [key2pmid.get(k, k) for k in keys]
        section_proof_ids[sid] = list((g["node"] or {}).get("proof_ids") or [])
        for k, t, y, s in g["papers"]:
            pm = key2pmid.get(k)
            if pm and crawl_abs.get(pm) and ws(s) == crawl_abs[pm]:
                pass
        added = list(dict.fromkeys(k for ev in g["refines"] for k in ev["added_keys"]))
    byproof: dict[tuple, set] = {}
    for g in groups:
        byproof.setdefault(tuple(section_proof_ids[g["id"]]), set()).add(tuple(sorted(section_evidence_keys[g["id"]])))
    shared = {k: v for k, v in byproof.items() if len(k) > 0 and sum(1 for g in groups if tuple(section_proof_ids[g["id"]]) == k) > 1}
    if shared:
        pass
    else:
        pass

    lines = [f"# {topic or data.get('title') or task_dir.name}", ""]
    by_id = {g["id"]: g for g in groups}
    for n in outline_final:
        if n["level"] == 1:
            lines += [f"## {n['id']} {n['title']}", ""]
            continue
        lines += [f"### {n['id']} {n['title']}", ""]
        g = by_id.get(n["id"])
        if g is None:
            lines += ["[no draft call found for this subsection]", ""]
            continue
        body = _tex_to_md(g["draft_completion"])
        lines += [body, ""]
    for g in groups:
        if str(g["id"]).startswith("unmatched:"):
            body = _tex_to_md(g["draft_completion"])
            lines += [f"### (unmatched) {g['title']}", "", body, ""]
    draft_report = "\n".join(lines)

    per_subsection = []
    for g in groups:
        sid = g["id"]
        first_body = _tex_to_md(g["draft_completion"])
        last_body = _tex_to_md(g["refines"][-1]["completion"]) if g["refines"] else first_body
        fin_txt = final_by_title.get(g["title"].lower()) or final_by_title.get(ws(g["title"]))
        if fin_txt is None:
            for k, v in final_by_title.items():
                if _sim(k, g["title"]) >= 0.95:
                    fin_txt = v
                    break
        which = "missing"
        if fin_txt is not None:
            sd, sr = _sim(_tex_body(fin_txt), _tex_body(g["draft_completion"])), (
                max((_sim(_tex_body(fin_txt), _tex_body(e["completion"])) for e in g["refines"]), default=0.0))
            which = "draft" if (sd >= 0.98 and sd >= sr) else ("refine" if sr >= 0.98 else "neither")
        ck = None
        if not str(sid).startswith("unmatched:"):
            hits = sorted(task_dir.glob("subsection_" + str(sid).replace(".", "_") + "_*_checkpoint.tex"))
            ck = hits[0] if hits else None
        ckw = "missing"
        if ck is not None:
            txt = ck.read_text(errors="replace")
            sd, sr = _sim(_tex_body(txt), _tex_body(g["draft_completion"])), (
                max((_sim(_tex_body(txt), _tex_body(e["completion"])) for e in g["refines"]), default=0.0))
            ckw = "draft" if (sd >= 0.98 and sd >= sr) else ("refine" if sr >= 0.98 else "neither")
        capped = bool(g["refines"]) and len(g["reviews"]) == len(g["refines"])
        dc, fc = _cite_keys(first_body), _cite_keys(last_body)
        sm = difflib.SequenceMatcher(None, first_body.split(), last_body.split(), autojunk=False)
        kept_words = sum(b.size for b in sm.get_matching_blocks())
        keep_w = kept_words / max(1, len(first_body.split()))
        keep_c = (len(set(dc) & set(fc)) / len(set(dc))) if dc else None
        sa, sb = _sentences(first_body), _sentences(last_body)
        keep_s = (len(set(sa) & set(sb)) / len(sa)) if sa else None
        per_subsection.append({
            "id": sid, "title": g["title"], "draft_seq": g["draft_seq"],
            "review_seqs": [e["seq"] for e in g["reviews"]], "review_scores": [e.get("score") for e in g["reviews"]],
            "final_is_satisfactory": (g["reviews"][-1].get("is_satisfactory") if g["reviews"] else None),
            "evidence_select_seqs": [e["seq"] for e in g["selects"]], "refine_seqs": [e["seq"] for e in g["refines"]],
            "n_refine_rounds": len(g["refines"]), "refine_cap_reached_no_closing_review": capped,
            "n_papers_in_draft_prompt": g["n_papers_listed"],
            "n_candidates_offered": [len(e["candidate_keys"]) for e in g["selects"]],
            "n_selected": [len(e["selected_keys"]) for e in g["selects"]],
            "added_paper_keys": list(dict.fromkeys(k for e in g["refines"] for k in e["added_keys"])),
            "suggested_queries": [q for e in g["reviews"] for q in (e.get("suggested_queries") or [])],
            "draft_words": _words(first_body), "final_words": _words(last_body),
            "draft_cites": len(dc), "final_cites": len(fc),
            "draft_words_kept_in_final": round(keep_w, 3), "draft_cites_kept_in_final": keep_c,
            "draft_sentences_kept_in_final": keep_s,
            "final_text_equals": which, "checkpoint_file": ck.name if ck else None, "checkpoint_equals": ckw,
        })

    def _pass(role_key: str) -> dict:
        seqs = [s for s, r in roles_seq if r == role_key]
        return {"role": BRIEF_ROLE[role_key], "seq_range": [min(seqs), max(seqs)] if seqs else None}

    refinement_passes = [_pass("review"), _pass("evidence_select"), _pass("refine")]
    if outline_reviews:
        seqs = [x["seq"] for x in outline_reviews]
        refinement_passes.insert(0, {"role": "review", "seq_range": [min(seqs), max(seqs)]})

    return {
        "outline_initial": outline_initial,
        "outline_final": outline_final,
        "section_evidence": section_evidence,
        "draft_report": draft_report,
        "refinement_passes": [{"role": p.get("role"), "seq_range": p.get("seq_range")} for p in refinement_passes],
        "per_subsection": [{"draft_seq": p.get("draft_seq")} for p in per_subsection],
    }
