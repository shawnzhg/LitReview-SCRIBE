"""Adapter for DR-Tulu outputs: inline cite tags resolved to PMIDs through the logged search calls."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ccbench.adapters import common
from ccbench.ingest import logs
from ccbench.model import Rollout

CITE = re.compile(r'<cite id="([^"]+)">')
CITE_FULL = re.compile(r'<cite id="[^"]+">(.*?)</cite>', re.S)


def adapt(task: str, task_dir: Path) -> Rollout:
    ans = task_dir / "answer.md"
    trp = task_dir / "trace.json"
    if not ans.exists() or not trp.exists():
        return common.bot("drtulu", task, task_dir, "answer.md/trace.json missing")
    tr = json.load(open(trp)).get("full_traces") or {}
    calls = tr.get("tool_calls") or []
    searches = logs.load_search(task_dir)
    cite2pmid: dict[str, list[str]] = {}
    papers: list[str] = []
    misaligned = 0
    for i, tc in enumerate(calls):
        s = searches[i] if i < len(searches) else None
        if s is None or (tc.get("query") and s["params"].get("query") and tc["query"] != s["params"]["query"]):
            s = next((x for x in searches if x["params"].get("query") == tc.get("query")), None)
            if s is None:
                misaligned += 1
                continue
        ret = s["returned"]
        for k, doc in enumerate(tc.get("documents") or []):
            if k < len(ret):
                cite2pmid[f"{tc['call_id']}-{k}"] = [ret[k]]
        papers.extend(ret)

    def _resolver(payload: str):
        return cite2pmid.get(payload)

    text = ans.read_text(errors="replace")
    marked = CITE_FULL.sub(lambda m: "", text)
    marked = re.sub(r'<cite id="([^"]+)"\s*/?>', r"[\1]", text)
    marked = re.sub(r"</cite>", "", marked)
    cite_re = re.compile(r"\[([0-9a-f]{6,}-\d+)\]")
    report = common.report_from_markdown(marked, cite_re, _resolver)
    outline = common.outline_from_markdown(text)
    rep = json.load(open(task_dir / "report.json")) if (task_dir / "report.json").exists() else {}
    return common.assemble(
        "drtulu",
        task,
        task_dir,
        papers=list(dict.fromkeys(papers)),
        outline=outline,
        report=report,
        graph=None,
        final_ok=bool(rep) and rep.get("total_failed_tool_calls", 0) == 0 or bool(text),
        bot_reason=None,
        extra_meta={"tool_calls": len(calls), "misaligned_calls": misaligned, "stopped_reason": (rep.get("trace_stats") or {}).get("stopped_reason"), "answer_words": rep.get("answer_words")},
    )
