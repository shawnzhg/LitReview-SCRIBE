"""Adapter for LLMxMapReduce outputs: numbered citations resolved through the reference URLs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ccbench.adapters import common
from ccbench.model import Rollout

CITE = re.compile(r"\[([\d,\s;–-]+)\]")
REF_LINE = re.compile(r"^\[(\d+)\]\s+.*?/page/(\d+)\s*$", re.M)
URL_PMID = re.compile(r"/page/(\d+)")


def adapt(task: str, task_dir: Path) -> Rollout:
    sj = task_dir / "survey.jsonl"
    if not sj.exists() or sj.stat().st_size < 1024:
        return common.bot("llmxmr", task, task_dir, "survey.jsonl missing/empty")
    row = None
    with open(sj) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
    if not row or not row.get("content"):
        return common.bot("llmxmr", task, task_dir, "survey row without content")
    refmap: dict[str, list[str]] = {}
    for n, pmid in REF_LINE.findall(row.get("ref_str") or ""):
        refmap[n] = [pmid]
    papers = []
    for p in row.get("papers") or []:
        m = URL_PMID.search(p.get("url") or "")
        if m:
            papers.append(m.group(1))
    report = common.report_from_markdown(row["content"], CITE, common.numeric_resolver(refmap))
    outline = common.outline_from_markdown(row.get("outline") or row["content"])
    flags = task_dir / "_harness_flags.json"
    skel_fail = json.load(open(flags)).get("skeleton_parse_failed") if flags.exists() else None
    return common.assemble("llmxmr", task, task_dir, papers=list(dict.fromkeys(papers)), outline=outline, report=report, graph=None, final_ok=True, bot_reason=None, extra_meta={"cite_ratio": row.get("cite_ratio"), "outline_eval_score": row.get("outline_eval_score"), "skeleton_parse_failed": skel_fail, "n_ref_lines": len(refmap)})
