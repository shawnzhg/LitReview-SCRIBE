"""Adapter for SurveyG outputs: paper-token citations resolved through the citation map, and the
outline read from its outline file."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ccbench.adapters import common
from ccbench.model import OutlineNode, Rollout

CITE = re.compile(r"\[((?:paper[0-9A-Za-z_]+\s*[,;]?\s*)+)\]")
PMID_IN_NAME = re.compile(r"PMID[:_]?(\d+)")


def _outline(og) -> list[OutlineNode]:
    out: list[OutlineNode] = []
    if not isinstance(og, list):
        return out
    for i, sec in enumerate(og):
        sid = f"o{i}"
        out.append(OutlineNode(id=sid, title=str(sec.get("section_title") or sec.get("title") or ""), level=1))
        subs = sec.get("subsections") or []
        for j, sub in enumerate(subs):
            title = sub.get("subsection_title") or sub.get("title") or (sub if isinstance(sub, str) else "")
            out.append(OutlineNode(id=f"{sid}.{j}", title=str(title), level=2, parent=sid))
    return out


def adapt(task: str, task_dir: Path) -> Rollout:
    md = task_dir / "literature_review.md"
    if not md.exists():
        return common.bot("surveyg", task, task_dir, "literature_review.md missing")
    data = {}
    dp = task_dir / "literature_review_data.json"
    if dp.exists():
        data = json.load(open(dp))
    key2pmid: dict[str, list[str]] = {}
    for fname, key in (data.get("citations_map") or {}).items():
        m = PMID_IN_NAME.search(fname)
        if m:
            key2pmid[key] = [m.group(1)]

    def _resolver(payload: str):
        out = []
        ok = False
        for tok in re.split(r"[,;]", payload):
            tok = tok.strip()
            if tok in key2pmid:
                ok = True
                out.extend(key2pmid[tok])
        return out if ok else None

    text = md.read_text(errors="replace")
    text = re.sub(r"## Papers Included:.*?(?=\n## |\n# )", "", text, flags=re.S)
    report = common.report_from_markdown(text, CITE, _resolver, drop_after_header=("references", "bibliography", "papers included"))
    papers = []
    for fname in data.get("paper_list") or []:
        m = PMID_IN_NAME.search(fname)
        if m:
            papers.append(m.group(1))
    og = task_dir / "survey_outline_gpt.json"
    outline = _outline(json.load(open(og))) if og.exists() else common.outline_from_markdown(text)
    return common.assemble("surveyg", task, task_dir, papers=list(dict.fromkeys(papers)), outline=outline, report=report, graph=None, final_ok=True, bot_reason=None, extra_meta={"papers_processed": data.get("papers_processed")})
