"""Adapter for AutoSurvey outputs: a markdown report with numbered citations resolved through its
reference map."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ccbench.adapters import common
from ccbench.model import Rollout

CITE = re.compile(r"\[([\d,\s;–-]+)\]")


def _find(task_dir: Path, suffix: str) -> Path | None:
    cands = [p for p in task_dir.glob(f"*{suffix}") if not p.name.startswith("_")]
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


def adapt(task: str, task_dir: Path, arm: str = "autosurvey", ref_value_to_pmid=lambda v: str(v)) -> Rollout:
    js = _find(task_dir, ".json")
    js = js if js and js.name not in ("egress_proof.json", "listener_census.json", "pool_health.json") else None
    if js is None:
        return common.bot(arm, task, task_dir, "survey json missing")
    d = json.load(open(js))
    text = d.get("survey") or ""
    if not text:
        md = _find(task_dir, ".md")
        text = md.read_text(errors="replace") if md else ""
    if not text:
        return common.bot(arm, task, task_dir, "survey text missing")
    refmap = {str(k): [ref_value_to_pmid(v)] for k, v in (d.get("reference") or {}).items() if v}
    report = common.report_from_markdown(text, CITE, common.numeric_resolver(refmap))
    outline = common.outline_from_markdown(text)
    papers = [refmap[k][0] for k in sorted(refmap, key=lambda x: int(x) if x.isdigit() else 10**9)]
    return common.assemble(arm, task, task_dir, papers=list(dict.fromkeys(papers)), outline=outline, report=report, graph=None, final_ok=True, bot_reason=None, extra_meta={"n_reference_entries": len(refmap), "survey_json": js.name})
