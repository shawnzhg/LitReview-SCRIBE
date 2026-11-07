"""Window extractor for LiRA task directories: rebuilds the merged outline, per-section evidence,
first complete draft and refinement rounds from the call log and output files."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

SIG_PERSONA_EDIT = "You are an expert literature review editor"

SIG_OUTLINE_GEN = "Your task is to create a systematic literature review outline"
SIG_OUTLINE_MERGE = "You have created the below literature review outlines. Merge these outlines"
SIG_REVIEW_OUTLINE = "evaluating an outline with the topic"
SIG_REVIEW_DRAFT = "evaluating a literature review paper draft"
SIG_DRAFT_SECTION = "Your task is to write the contents for a literature review section"
SIG_DRAFT_TITLE_ABS = "Your task is to write the title and abstract for a literature review"
SIG_DRAFT_CONCLUSION = "Your task is to write the conclusion for a literature review"
SIG_REFINE_SECTION = "The literature review section you wrote has received feedback"
SIG_REFINE_TITLE_ABS = "The title and abstract you wrote has received feedback"
SIG_REFINE_CONCLUSION = "The conclusion you wrote has received feedback"
SIG_EDITOR_CONTINUE = "the edit maybe got cutoff"

SIGNATURES: list[tuple[str, str, str]] = [
    (SIG_OUTLINE_GEN, "outline_gen", "outline_gen"),
    (SIG_OUTLINE_MERGE, "outline_merge", "outline_merge"),
    (SIG_REVIEW_OUTLINE, "review_outline", "review"),
    (SIG_REVIEW_DRAFT, "review_draft", "review"),
    (SIG_DRAFT_SECTION, "draft_section", "draft"),
    (SIG_DRAFT_TITLE_ABS, "draft_title_abstract", "draft"),
    (SIG_DRAFT_CONCLUSION, "draft_conclusion", "draft"),
    (SIG_REFINE_SECTION, "refine_section", "refine_coherence"),
    (SIG_REFINE_TITLE_ABS, "refine_title_abstract", "refine_coherence"),
    (SIG_REFINE_CONCLUSION, "refine_conclusion", "refine_coherence"),
    (SIG_EDITOR_CONTINUE, "editor_continue", "final_assembly"),
    (SIG_PERSONA_EDIT, "editor", "final_assembly"),
]
HEADER_RE = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$", re.M)
BRACKET_CITE = re.compile(r"\[([^\[\]\n]{6,400})\]")
SEC_TITLE_RE = re.compile(r"^Section Title:[ \t]*(.*)$", re.M)
SEC_DESC_RE = re.compile(r"^Description:[ \t]*(.*?)(?=^Is a subsection of:|^References \(note)", re.M | re.S)
REF_TITLE_RE = re.compile(r"^TITLE:(.*)$", re.M)
REF_ENTRY_RE = re.compile(r"^TITLE:([^\n]*)\n+CONTENT:(.*?)(?=\n+TITLE:|\n-{3,}|\Z)", re.M | re.S)
REF_BLOCK_MARK = "References (note that you can leave some of these out"


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (t or "").lower()).strip()


def _norm_key(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip()).casefold()


def _user_text(rec: dict) -> str:
    msgs = rec.get("messages") or []
    us = [m.get("content") or "" for m in msgs if m.get("role") == "user"]
    if us:
        return us[-1]
    return rec.get("prompt") or ""


def _classify(user: str) -> tuple[str, str]:
    for sig, native, role in SIGNATURES:
        if sig in user:
            return native, role
    return "other", "other"


def _find_merged_outlines(task_dir: Path) -> list[Path]:
    def key(p: Path) -> tuple[int, str]:
        m = re.search(r"merged_outline_(\d+)\.json$", p.name)
        return (int(m.group(1)) if m else 0, p.name)

    return sorted(task_dir.glob("temp/srg/*/*/outline/merged_outline_*.json"), key=key)


def _lc_content(p: Path) -> str:
    try:
        d = json.loads(p.read_text())
    except Exception:
        return ""
    if isinstance(d, dict):
        return d.get("content") or ""
    return ""


def _parse_outline(markdown: str) -> list[dict]:
    nodes: list[dict] = []
    heads = list(HEADER_RE.finditer(markdown))
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(heads):
        level = len(m.group(1))
        title = m.group(2).strip()
        if not title:
            continue
        body_end = heads[i + 1].start() if i + 1 < len(heads) else len(markdown)
        desc = markdown[m.end():body_end].strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        nid = f"o{len(nodes)}"
        suggested = []
        for payload in BRACKET_CITE.findall(desc):
            for part in payload.split("|"):
                part = part.strip()
                if part:
                    suggested.append(part)
        nodes.append({
            "id": nid,
            "title": title,
            "level": level,
            "parent": stack[-1][1] if stack else None,
            "description": desc,
            "suggested_papers": list(dict.fromkeys(suggested)),
        })
        stack.append((level, nid))
    return nodes


def _pool(task_dir: Path) -> tuple[list[str], dict[str, str], dict[str, str], int]:
    papers: list[str] = []
    title2pmid: dict[str, str] = {}
    content2pmid: dict[str, str] = {}
    untitled = 0
    inp = task_dir / "data" / "scireviewgen" / "full_data_abs.json"
    if not inp.exists():
        return papers, title2pmid, content2pmid, untitled
    try:
        d = json.loads(inp.read_text())
    except Exception:
        return papers, title2pmid, content2pmid, untitled
    for rec in (d if isinstance(d, list) else [d]):
        for r in rec.get("references", []) or []:
            if r.get("id") is None:
                continue
            pm = str(r["id"])
            if (r.get("title") or "").strip():
                title2pmid.setdefault(_norm_title(r["title"]), pm)
            else:
                untitled += 1
            if (r.get("content") or "").strip():
                content2pmid.setdefault(_norm_title(r["content"])[:200], pm)
            papers.append(pm)
    return list(dict.fromkeys(papers)), title2pmid, content2pmid, untitled


def _strip_leading_h1(text: str) -> str:
    t = text.lstrip()
    m = re.match(r"^#{1,6}[ \t]+[^\n]*\n?", t)
    return t[m.end():].lstrip("\n") if m else t


def extract(task_dir: Path) -> dict:
    task_dir = Path(task_dir)
    calls_path = task_dir / "_calls.jsonl"

    calls_total = 0
    runs: list[list] = []

    first_draft: "OrderedDict[str, dict]" = OrderedDict()
    draft_ta: dict | None = None
    draft_concl: dict | None = None
    evidence: "OrderedDict[str, list[str]]" = OrderedDict()

    if calls_path.exists():
        with calls_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                calls_total += 1
                seq = rec.get("seq", calls_total - 1)
                user = _user_text(rec)
                native, role = _classify(user)
                if rec.get("status") not in (200, None) or not (rec.get("completion") or ""):
                    pass
                if runs and runs[-1][2] == role and runs[-1][3] == native:
                    runs[-1][1] = seq
                    runs[-1][4] += 1
                else:
                    runs.append([seq, seq, role, native, 1])

                comp = rec.get("completion") or ""

                if native in ("draft_section", "refine_section"):
                    mt = SEC_TITLE_RE.search(user)
                    title = mt.group(1).strip() if mt else ""
                    key = _norm_key(title)
                    md = SEC_DESC_RE.search(user)
                    if md is not None:
                        if md.group(1).strip():
                            pass
                    if REF_BLOCK_MARK in user:
                        entries = [(t.strip(), c.strip()) for t, c in REF_ENTRY_RE.findall(user)]
                        if not entries:
                            entries = [(t.strip(), "") for t in REF_TITLE_RE.findall(user)]
                        if key not in evidence:
                            evidence[key] = entries
                    if native == "draft_section" and key not in first_draft and comp:
                        first_draft[key] = {"seq": seq, "title": title, "completion": comp}
                elif native == "draft_title_abstract" and comp and draft_ta is None:
                    draft_ta = {"seq": seq, "completion": comp}
                elif native == "draft_conclusion" and comp and draft_concl is None:
                    draft_concl = {"seq": seq, "completion": comp}

    merged = _find_merged_outlines(task_dir)
    outline_initial = _parse_outline(_lc_content(merged[0])) if merged else []
    outline_final = _parse_outline(_lc_content(merged[-1])) if merged else []
    committed_md = _lc_content(merged[-1]) if merged else ""
    _lines = committed_md.split("\n")
    blank_after, tight_after = 0, 0
    for _i, _l in enumerate(_lines):
        if re.match(r"^#{1,6}\s+\S", _l):
            _nxt = _lines[_i + 1] if _i + 1 < len(_lines) else ""
            if _nxt.strip():
                tight_after += 1
            else:
                blank_after += 1

    papers, title2pmid, content2pmid, n_untitled_pool = _pool(task_dir)

    def _resolve(title: str, content: str) -> str:
        n = _norm_title(title)
        if n and n in title2pmid:
            return title2pmid[n]
        cn = _norm_title(content)[:200]
        if cn and cn in content2pmid:
            return content2pmid[cn]
        return title

    by_key = {_norm_key(n["title"]): n["id"] for n in outline_final}
    section_evidence: dict[str, list[str]] = {}
    for key, entries in evidence.items():
        sid = by_key.get(key, key)
        ids = []
        for t, c in entries:
            pm = _resolve(t, c)
            if pm == t:
                pass
            else:
                pass
            ids.append(pm)
        section_evidence[sid] = ids


    parts: list[str] = []
    if draft_ta:
        parts.append(draft_ta["completion"].strip())
    drafted, undrafted = [], []
    for node in outline_final:
        key = _norm_key(node["title"])
        fd = first_draft.get(key)
        if fd is None:
            undrafted.append(node["title"])
            continue
        drafted.append(node["title"])
        parts.append("#" * node["level"] + " " + node["title"] + "\n\n"
                     + _strip_leading_h1(fd["completion"]).strip())
    if draft_concl:
        parts.append(draft_concl["completion"].strip())
    draft_report = "\n\n".join(p for p in parts if p)

    refinement_passes = []
    for start, end, role, native, n in runs:
        if role in ("review", "refine_coherence", "final_assembly"):
            if native == "review_outline":
                continue
            refinement_passes.append({"role": role, "seq_range": f"{start}-{end}"})


    sc = task_dir / "search_calls.jsonl"
    if sc.exists():
        pass


    role_runs: list[list] = []
    for start, end, role, _native, _n in runs:
        if role_runs and role_runs[-1][2] == role:
            role_runs[-1][1] = end
        else:
            role_runs.append([start, end, role])

    return {
        "outline_initial": outline_initial,
        "outline_final": outline_final,
        "section_evidence": section_evidence,
        "draft_report": draft_report,
        "refinement_passes": [{"role": p.get("role"), "seq_range": p.get("seq_range")} for p in refinement_passes],
    }
