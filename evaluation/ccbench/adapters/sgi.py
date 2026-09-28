"""Adapter for SurveyGen-I outputs: PMID-keyed citations and bibliography."""

from __future__ import annotations

import re
from pathlib import Path

from ccbench.adapters import common
from ccbench.ingest import history
from ccbench.model import Rollout

CITE = re.compile(r"\[((?:pmid\d+\s*[;,]?\s*)+)\]")
BIBKEY = re.compile(r"@\w+\{(pmid(\d+)),")


def _resolver(payload: str):
    ids = re.findall(r"pmid(\d+)", payload)
    return ids or None


def adapt(task: str, task_dir: Path) -> Rollout:
    md = task_dir / "acm" / "final_survey_refined.md"
    if not md.exists():
        return common.bot("sgi", task, task_dir, "final_survey_refined.md missing")
    text = md.read_text(errors="replace")
    report = common.report_from_markdown(text, CITE, _resolver)
    outline = common.outline_from_markdown(text)
    bib = task_dir / "acm" / "references_master.bib"
    kept = [m.group(2) for m in BIBKEY.finditer(bib.read_text(errors="replace"))] if bib.exists() else []
    events, _ = history.build_events(task_dir, 50)
    opened: dict[str, None] = {}
    for e in events:
        if e.kind == "open" and e.route == "pdf":
            for p in e.returned_ids:
                opened.setdefault(p, None)
    papers = list(opened) if opened else list(dict.fromkeys(kept))
    return common.assemble("sgi", task, task_dir, papers=papers, outline=outline, report=report, graph=None, final_ok=True, bot_reason=None, extra_meta={"bib_entries": len(kept), "n_opened_pdf": len(opened)})
