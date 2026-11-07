"""Adapter for LiRA outputs: title-style citations resolved to PMIDs against the bibliography it was
given."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

from ccbench.adapters import common
from ccbench.model import Rollout

CITE = re.compile(r"\[([^\[\]\n]{6,400})\]")
NUMERIC_ONLY = re.compile(r"^[\d,\s;–-]+$")


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def _find_final(task_dir: Path) -> Path | None:
    cands = sorted(task_dir.glob("temp/srg/*/*/final/final_*.json"))
    return cands[-1] if cands else None


def _find_outline(task_dir: Path) -> Path | None:
    cands = sorted(task_dir.glob("temp/srg/*/*/outline/merged_outline_*.json"))
    return cands[-1] if cands else None


def adapt(task: str, task_dir: Path) -> Rollout:
    fp = _find_final(task_dir)
    if fp is None:
        return common.bot("lira", task, task_dir, "final/final_*.json missing")
    content = json.load(open(fp)).get("content") or ""
    if not content.strip():
        return common.bot("lira", task, task_dir, "final content empty")
    title2pmid: dict[str, str] = {}
    papers: list[str] = []
    inp = task_dir / "data" / "scireviewgen" / "full_data_abs.json"
    if inp.exists():
        d = json.load(open(inp))
        recs = d if isinstance(d, list) else [d]
        for rec in recs:
            for r in rec.get("references", []):
                if r.get("id") is not None and r.get("title"):
                    pm = str(r["id"])
                    title2pmid[_norm(r["title"])] = pm
                    papers.append(pm)
    keys = list(title2pmid)
    cache: dict[str, list[str] | None] = {}

    def _resolve_one(t: str) -> str | None:
        n = _norm(t)
        if n in title2pmid:
            return title2pmid[n]
        if not keys:
            return None
        best = process.extractOne(n, keys, scorer=fuzz.token_set_ratio, score_cutoff=90)
        return title2pmid[best[0]] if best else None

    def _resolver(payload: str):
        if NUMERIC_ONLY.match(payload):
            return None
        if payload in cache:
            return cache[payload]
        out = []
        ok = False
        for part in payload.split("|"):
            pm = _resolve_one(part.strip())
            if pm:
                ok = True
                out.append(pm)
        cache[payload] = out if ok else None
        return cache[payload]

    report = common.report_from_markdown(content, CITE, _resolver)
    op = _find_outline(task_dir)
    outline = common.outline_from_markdown(json.load(open(op)).get("content") or "") if op else []
    if not outline:
        outline = common.outline_from_markdown(content)
    return common.assemble("lira", task, task_dir, papers=list(dict.fromkeys(papers)), outline=outline, report=report, graph=None, final_ok=True, bot_reason=None, extra_meta={"final_file": fp.name, "n_pool_titles": len(title2pmid)})
