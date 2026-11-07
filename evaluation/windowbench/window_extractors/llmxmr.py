"""Window extractor for LLMxMapReduce task directories: rebuilds the skeletons, per-node evidence,
first complete draft and refinement passes from the call log and output files."""

from __future__ import annotations

import json
import re
from pathlib import Path

SIG_SHARED_PREAMBLE = "conduct literature reviews based on provided materials"

SIG_SEARCH_QUERY    = "Please generate a set of search queries"
SIG_SCREEN          = "Please infer the degree of relevance between this web page and the topic"
SIG_CLEAN_PAGE      = "Output the main body text, removing image links, website URLs, advertisements"
SIG_PAGE_QUALITY    = "Evaluate the quality of the following content retrieved from the internet"
SIG_CLUSTER         = "your responsibility is to group these papers for writing digests"
SIG_SKELETON_LOCAL  = "construct the outline of the survey based on the provided **paper abstracts**"
SIG_SKELETON_AGG    = "based on the provided **initial outlines**"
SIG_DIGEST          = "the first step is to distill each paper into a concise **paper digest**"
SIG_SUGGEST_DIGEST  = "As an academic literature review architect, your task is to refine the theoretical framework"
SIG_SUGGEST_MERGE   = "has received independent reviews from multiple reference papers perspectives"
SIG_OUTLINE_REVISE  = "your task is to create a new version of the outline"
SIG_OUTLINE_ENTROPY = "from the perspective of outline information entropy"
SIG_WRITE_LEAF      = "create a single subsection for the final survey"
SIG_WRITE_PARENT    = "write a guidance of its child sub-sections"
SIG_REFINE_CITATION = "Convert multiple consecutive references to this form"
SIG_FINAL_ASSEMBLY  = "Create multiple Markdown tables or Mermaid charts"
SIGNATURES = [
    ("search_query",      SIG_SEARCH_QUERY),
    ("screen",            SIG_SCREEN),
    ("clean_page",        SIG_CLEAN_PAGE),
    ("page_quality",      SIG_PAGE_QUALITY),
    ("cluster",           SIG_CLUSTER),
    ("skeleton_local",    SIG_SKELETON_LOCAL),
    ("skeleton_agg",      SIG_SKELETON_AGG),
    ("digest",            SIG_DIGEST),
    ("suggest_from_digest", SIG_SUGGEST_DIGEST),
    ("suggest_merge",     SIG_SUGGEST_MERGE),
    ("outline_revise",    SIG_OUTLINE_REVISE),
    ("outline_entropy",   SIG_OUTLINE_ENTROPY),
    ("write_leaf",        SIG_WRITE_LEAF),
    ("write_parent",      SIG_WRITE_PARENT),
    ("refine_citation",   SIG_REFINE_CITATION),
    ("final_assembly",    SIG_FINAL_ASSEMBLY),
]

NATIVE_TO_BRIEF = {
    "search_query":       "other",
    "screen":             "evidence_select",
    "clean_page":         "other",
    "page_quality":       "evidence_select",
    "cluster":            "other",
    "skeleton_local":     "outline_gen",
    "skeleton_agg":       "outline_merge",
    "digest":             "digest",
    "suggest_from_digest": "review",
    "suggest_merge":      "outline_merge",
    "outline_revise":     "outline_revise",
    "outline_entropy":    "select_best",
    "write_leaf":         "draft",
    "write_parent":       "draft",
    "refine_citation":    "refine_citation",
    "final_assembly":     "final_assembly",
    None:                 "other",
}


CITE_BIBKEY = re.compile(r"'([A-Za-z0-9][A-Za-z0-9_]{3,})'")
HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(.*?)[ \t]*$")
LEAD_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+")
DASH_SEP = re.compile(r"(?m)^-{6,}\s*$")
FIELD_LABEL = re.compile(r"^\s*(Digest Construction|Digest Analysis)\s*:\s*(.*)$")
SCORE = re.compile(r"<SCORE>\s*([0-9.]+)\s*</SCORE>")


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _turns(rec: dict) -> tuple[str, str]:
    sys_t, usr_t = "", ""
    for m in rec.get("messages") or []:
        if m.get("role") == "system":
            sys_t = m.get("content") or ""
        elif m.get("role") == "user":
            usr_t = (m.get("content") or "") if not usr_t else usr_t + "\n" + (m.get("content") or "")
    return sys_t, usr_t


def haystack(sys_t: str, usr_t: str) -> str:
    return _ws(sys_t[:400] + " || " + usr_t[:3000] + " ~~ " + usr_t[-2000:])


def classify(sys_t: str, usr_t: str) -> tuple[str | None, list[str]]:
    hay = haystack(sys_t, usr_t)
    hits = [name for name, sig in SIGNATURES if sig in hay]
    return (hits[0] if len(hits) == 1 else None), hits


def _fence(text: str, start: int = 0) -> str | None:
    m = re.search(r"```(?:markdown)?[ \t]*\n(.*?)\n?```", text[start:], re.S)
    return m.group(1) if m else None


def _fence_after(text: str, marker: str) -> str | None:
    i = text.find(marker)
    return None if i < 0 else _fence(text, i)


def _completion_md(comp: str) -> str:
    f = _fence(comp or "")
    return (f if f is not None else (comp or "")).strip()


EMPH = re.compile(r"[*`]+")


def _norm_title(t: str) -> str:
    return _ws(EMPH.sub("", LEAD_NUM.sub("", t))).lower()


def htuple(md: str) -> tuple:
    return tuple((len(m.group(1)), _norm_title(m.group(2))) for m in HEADING.finditer(md or ""))


def strip_field_labels(md: str) -> str:
    out = []
    for ln in (md or "").split("\n"):
        m = FIELD_LABEL.match(ln)
        if m:
            rest = m.group(2).strip()
            if rest:
                out.append(rest)
        else:
            out.append(ln)
    return "\n".join(x for x in out if x.strip() != "")


def parse_skeleton(md: str) -> list[dict]:
    md = md or ""
    heads = list(HEADING.finditer(md))
    if not heads:
        return []
    nodes: list[dict] = []
    stack: list[tuple[int, str]] = []
    child_n: dict[str, int] = {}
    for k, m in enumerate(heads):
        lvl = len(m.group(1))
        raw = m.group(2).strip()
        num_m = LEAD_NUM.match(raw)
        num = num_m.group(1) if num_m else None
        title = _ws(LEAD_NUM.sub("", raw))
        body = md[m.end(): heads[k + 1].start() if k + 1 < len(heads) else len(md)]
        while stack and stack[-1][0] >= lvl:
            stack.pop()
        parent = stack[-1][1] if stack else None
        if num:
            nid = num
        elif parent is None:
            nid = "0"
        else:
            child_n[parent] = child_n.get(parent, 0) + 1
            nid = f"{parent}.{child_n[parent]}" if parent != "0" else str(child_n[parent])
        dc, da, plain, mode = [], [], [], None
        for ln in body.split("\n"):
            fm = FIELD_LABEL.match(ln)
            if fm:
                mode = fm.group(1)
                tgt = dc if mode == "Digest Construction" else da
                if fm.group(2).strip():
                    tgt.append(fm.group(2).strip())
            elif mode and ln.strip():
                (dc if mode == "Digest Construction" else da).append(ln.strip())
            elif ln.strip():
                plain.append(ln.strip())
        desc = "\n".join([x for x in (" ".join(dc), " ".join(da)) if x] + plain).strip()
        nodes.append({
            "id": nid, "title": title, "level": lvl, "parent": parent,
            "description": desc,
            "digest_construction": " ".join(dc) or None,
            "digest_analysis": " ".join(da) or None,
            "bibkeys": sorted(set(CITE_BIBKEY.findall(body))),
            "raw_heading": raw,
        })
        stack.append((lvl, nid))
    ids = {n["id"] for n in nodes}
    parents = {n["parent"] for n in nodes if n["parent"] in ids}
    for n in nodes:
        n["is_leaf"] = n["id"] not in parents
    return nodes


def _digest_bibkey(usr: str) -> str | None:
    m = re.search(r"##\s*Bibkey of the Reference Paper\s*\n\s*\['([^']+)'\]", usr)
    return m.group(1) if m else None


def _digest_skeleton(usr: str) -> str | None:
    return _fence_after(usr, "## Initial Skeleton")


def _digest_raw_paper(usr: str) -> str:
    i = usr.find("## Reference Paper")
    if i < 0:
        return ""
    j = usr.find("## Initial Skeleton", i)
    return usr[i + len("## Reference Paper"): j if j > 0 else len(usr)].strip()


def _entropy_skeleton(usr: str) -> str | None:
    return _fence_after(usr, "## **Skeleton**")


def _revise_parent_skeleton(usr: str) -> str | None:
    return _fence_after(usr, "## **Initial Skeleton**")


def _write_section(usr: str, leaf: bool) -> tuple[str | None, str | None]:
    key = "Sub-Section Description:" if leaf else "Section Description:"
    m = re.search(r"(?m)^" + re.escape(key) + r"\s*$", usr)
    if not m:
        return None, None
    blk = _fence(usr, m.end())
    if blk is None:
        return None, None
    hm = HEADING.search(blk)
    if not hm:
        return None, None
    return hm.group(0).strip(), blk[hm.end():].strip()


def _write_digest_chunks(usr: str) -> list[dict]:
    blob = _fence_after(usr, "Individual Paper Digests:")
    if blob is None:
        return []
    out = []
    for chunk in DASH_SEP.split(blob):
        c = chunk.strip()
        if not c:
            continue
        m = re.match(r"Paper bibkey:\s*\['([^']+)'\]\s*\nDigest:\s*\n(.*)$", c, re.S)
        if m:
            out.append({"label": m.group(1), "text": m.group(2).strip()})
        else:
            out.append({"label": None, "text": c})
    return out


def _refine_io(usr: str, comp: str) -> tuple[str, str]:
    a = usr.find("['Content']")
    b = usr.find("['Output Requirements']")
    inp = usr[a + len("['Content']"): b].strip() if (a >= 0 and b > a) else ""
    return inp, _completion_md(comp)


def extract(task_dir: Path) -> dict:
    task_dir = Path(task_dir)

    row: dict = {}
    sj = task_dir / "survey.jsonl"
    if sj.exists():
        try:
            for line in open(sj):
                if line.strip():
                    row = json.loads(line)
        except Exception as e:
            pass
    else:
        pass
    survey_outline = row.get("outline") or ""
    survey_content = row.get("content") or ""
    target_ht = htuple(strip_field_labels(survey_outline)) if survey_outline else None

    calls_path = task_dir / "_calls.jsonl"
    if not calls_path.exists():
        return {
            "outline_initial": [],
            "outline_final": [],
            "section_evidence": {},
            "draft_report": "",
            "refinement_passes": [],
        }

    calls: list[dict] = []
    native_counts: dict[str, int] = {}
    preamble_by_native: dict[str, int] = {}

    s0_md: str | None = None
    digests: dict[int, dict] = {}
    raw_paper_probes: list[str] = []
    entropy_final: tuple[int, str] | None = None
    entropy_last: tuple[int, str] | None = None
    parent_ht_counts: dict[tuple, list[int]] = {}
    write_rows: list[dict] = []
    refine_rows: list[dict] = []
    assembly: dict = {}
    first_outline_t = None
    first_write_t = None

    for line in open(calls_path):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        seq = int(rec.get("seq", len(calls)))
        sys_t, usr_t = _turns(rec)
        native, hits = classify(sys_t, usr_t)
        if len(hits) > 1:
            pass
        key = native or "unmatched"
        native_counts[key] = native_counts.get(key, 0) + 1
        if SIG_SHARED_PREAMBLE in _ws(usr_t):
            preamble_by_native[key] = preamble_by_native.get(key, 0) + 1
        comp = rec.get("completion") or ""
        calls.append({"seq": seq, "t": rec.get("t"), "native_role": native,
                      "role": NATIVE_TO_BRIEF.get(native, "other"),
                      "prompt_chars": len(usr_t), "completion_chars": len(comp),
                      "finish_reason": rec.get("finish_reason"), "status": rec.get("status")})

        if native == "screen":
            m = SCORE.search(comp)
        elif native == "page_quality":
            pass
        elif native == "skeleton_local":
            first_outline_t = first_outline_t or rec.get("t")
        elif native == "cluster":
            first_outline_t = first_outline_t or rec.get("t")
        elif native == "skeleton_agg":
            s0_md, s0_seq = _completion_md(comp), seq
        elif native == "digest":
            sk = _digest_skeleton(usr_t)
            raw = _digest_raw_paper(usr_t)
            if raw:
                raw_paper_probes.append(_ws(raw)[:300])
            digests[seq] = {"bibkey": _digest_bibkey(usr_t), "completion": comp,
                            "completion_ws": _ws(comp),
                            "skeleton_ht": htuple(sk or ""), "raw_paper_chars": len(raw),
                            "completion_chars": len(comp)}
        elif native == "suggest_from_digest":
            pass
        elif native == "suggest_merge":
            pass
        elif native == "outline_revise":
            pht = htuple(_revise_parent_skeleton(usr_t) or "")
            parent_ht_counts.setdefault(pht, []).append(seq)
        elif native == "outline_entropy":
            sk = _entropy_skeleton(usr_t) or ""
            ht = htuple(sk)
            m = SCORE.search(comp)
            entropy_last = (seq, sk)
            if target_ht and ht == target_ht:
                entropy_final = (seq, sk)
        elif native in ("write_leaf", "write_parent"):
            first_write_t = first_write_t or rec.get("t")
            usr_ws = _ws(usr_t)
            if any(p and p in usr_ws for p in raw_paper_probes):
                pass
            head, desc = _write_section(usr_t, native == "write_leaf")
            chunks = _write_digest_chunks(usr_t)
            ev = []
            for ch in chunks:
                full_ws = _ws(ch["text"])
                owners: list = []
                for off in (0, 200, max(0, len(full_ws) // 2)):
                    probe_ws = full_ws[off:off + 200]
                    if len(probe_ws) < 60:
                        continue
                    owners = [(s, d["bibkey"]) for s, d in digests.items() if probe_ws in d["completion_ws"]]
                    if len(owners) == 1:
                        break
                ev.append({"label": ch["label"], "chars": len(ch["text"]),
                           "from_digest_seq": owners[-1][0] if owners else None,
                           "from_digest_bibkey": owners[-1][1] if owners else None,
                           "n_owner_digests": len(owners)})
            write_rows.append({"seq": seq, "native_role": native, "heading": head,
                               "title_norm": _norm_title(re.sub(r"^#+\s*", "", head or "")),
                               "level": (len(head) - len(head.lstrip("#"))) if head else None,
                               "description": desc, "draft": _completion_md(comp),
                               "n_child_text_chars": len((usr_t.split("Subsections:", 1)[1].split("Individual Paper Digests:", 1)[0])
                                                         if (native == "write_parent" and "Subsections:" in usr_t) else ""),
                               "evidence": ev, "prompt_chars": len(usr_t)})
        elif native == "refine_citation":
            inp, out = _refine_io(usr_t, comp)
            refine_rows.append({"seq": seq, "in_chars": len(inp), "out_chars": len(out),
                                "identical": inp == out, "in": inp, "out": out})
        elif native == "final_assembly":
            assembly = {"seq": seq,
                        "n_figures": len(re.findall(r"(?m)^Section Title:", comp)),
                        "n_mermaid": comp.count("```mermaid"),
                        "n_tables": len(re.findall(r"```markdown", comp)),
                        "completion_chars": len(comp)}


    outline_initial = parse_skeleton(s0_md or "")

    if entropy_final:
        final_md = entropy_final[1]
    elif survey_outline:
        final_md = survey_outline
    elif entropy_last:
        final_md = entropy_last[1]
    else:
        final_md = ""
    outline_final = parse_skeleton(final_md)

    by_title_all: dict[str, list[dict]] = {}
    for n in outline_final:
        by_title_all.setdefault(_norm_title(n["title"]), []).append(n)
    taken: set[str] = set()

    def _node_for(title_norm: str, desc: str) -> dict | None:
        cands = by_title_all.get(title_norm) or []
        free = [c for c in cands if c["id"] not in taken] or cands
        if len(free) == 1:
            best = free[0]
        elif not free:
            return None
        else:
            import difflib
            d = _ws(desc or "")
            best = max(free, key=lambda c: difflib.SequenceMatcher(None, d[:600], _ws(c["description"])[:600]).ratio())
        taken.add(best["id"])
        return best

    s0_ht = htuple(s0_md or "")

    final_round_ht = htuple(final_md)
    digest_rounds = {}
    for s, d in digests.items():
        rnd = ("under_S0" if d["skeleton_ht"] == s0_ht
               else "under_final_skeleton" if d["skeleton_ht"] == final_round_ht else "other")
        digest_rounds.setdefault(rnd, []).append(s)
    for s in sorted(digest_rounds.get("under_final_skeleton", []) or digests.keys()):
        bk = digests[s]["bibkey"] or f"seq{s}"

    section_evidence: dict[str, list[str]] = {}
    for w in write_rows:
        node = _node_for(w["title_norm"], w["description"])
        sid = node["id"] if node else f"?{w['title_norm'][:40]}"
        w["section_id"] = sid
        bks, det = [], []
        for e in w["evidence"]:
            bk = e["label"] or e["from_digest_bibkey"]
            if bk and bk not in bks:
                bks.append(bk)
            det.append({"bibkey": bk, "chars": e["chars"], "labelled": e["label"] is not None,
                        "from_digest_seq": e["from_digest_seq"]})
        section_evidence[sid] = bks

    non_root = [n for n in outline_final if n["parent"] is not None]

    draft_by_sid = {w["section_id"]: w for w in write_rows}
    parts, missing = [], []
    root = next((n for n in outline_final if n["parent"] is None), None)
    if root:
        parts.append("#" * root["level"] + " " + root["raw_heading"])
    for n in non_root:
        w = draft_by_sid.get(n["id"])
        if not w:
            missing.append(n["id"])
            continue
        body = w["draft"]
        hm = HEADING.search(body)
        if hm and hm.start() < 8:
            body = body[hm.end():].strip()
        parts.append("#" * n["level"] + " " + n["raw_heading"] + "\n\n" + body.strip())
    draft_report = "\n\n".join(parts).strip()

    for r in refine_rows:
        probe = _ws(r["in"])[:200]
        owner = next((w["section_id"] for w in write_rows if probe and probe in _ws(w["draft"])), None)
        r["section_id"] = owner
    content_ws = _ws(survey_content)
    for r in refine_rows:
        flat = _ws(re.sub(r"\['[^']*'(?:\s*,\s*'[^']*')*\]", "", r["out"]))
        windows = [flat[o:o + 120] for o in (0, 400, max(0, len(flat) // 2))]
        found = [w in content_ws for w in windows if len(w) > 60]
        if found and found[0]:
            pass
        elif any(found):
            pass
        else:
            pass

    refinement_passes = []
    if refine_rows:
        refinement_passes.append({"role": "refine_citation", "seq_range": [refine_rows[0]["seq"], refine_rows[-1]["seq"]]})
    if assembly:
        refinement_passes.append({"role": "final_assembly", "seq_range": [assembly["seq"], assembly["seq"]]})

    searches, search_pmids, routes, cutoffs, last_search_ts = [], set(), set(), set(), None
    scp = task_dir / "search_calls.jsonl"
    if scp.exists():
        for line in open(scp):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            searches.append(s)
            routes.add(s.get("route"))
            cutoffs.add(s.get("cutoff"))
            last_search_ts = s.get("ts") or last_search_ts
            for h in s.get("returned") or []:
                if h.get("pmid"):
                    search_pmids.add(str(h["pmid"]))
    papers = []
    for p in row.get("papers") or []:
        m = re.search(r"/page/(\d+)", p.get("url") or "")
        papers.append({"bibkey": p.get("bibkey"), "title": p.get("title"), "pmid": m.group(1) if m else None})

    try:
        from ccbench.adapters import common as ccb_common
        from ccbench.adapters.llmxmr import CITE, REF_LINE
    except Exception as e:
        pass

    call_roles = []
    for c in calls:
        if call_roles and call_roles[-1][1] == c["role"]:
            call_roles[-1][0][1] = c["seq"]
        else:
            call_roles.append([[c["seq"], c["seq"]], c["role"]])
    call_roles = [[tuple(r), n] for r, n in call_roles]
    native_runs = []
    for c in calls:
        if native_runs and native_runs[-1][1] == c["native_role"]:
            native_runs[-1][0][1] = c["seq"]
        else:
            native_runs.append([[c["seq"], c["seq"]], c["native_role"]])


    return {
        "outline_initial": outline_initial,
        "outline_final": outline_final,
        "section_evidence": section_evidence,
        "draft_report": draft_report,
        "refinement_passes": [{"role": p.get("role"), "seq_range": p.get("seq_range")} for p in refinement_passes],
        "papers": papers,
    }
