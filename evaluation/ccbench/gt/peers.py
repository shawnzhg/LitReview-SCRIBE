"""Same-topic human peer reviews of a task, which form its reference band."""

from __future__ import annotations

import functools
import json

from ccbench import paths


@functools.lru_cache(maxsize=1)
def domain_index() -> dict:
    with open(paths.gt_manifest("domain_index.json")) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def per_paper() -> dict[str, dict]:
    with open(paths.gt_manifest("per_paper.json")) as f:
        d = json.load(f)
    if isinstance(d, list):
        d = {x["paper_id"]: x for x in d}
    return d


def topic_of(task: str) -> str | None:
    p = per_paper().get(task)
    return p.get("topic") if p else None


def peers(task: str, with_graph_only: bool = True, split: str | None = None) -> list[str]:
    topic = topic_of(task)
    if topic is None:
        return []
    papers = domain_index().get(topic, {}).get("papers", [])
    out = []
    for p in papers:
        if p == task:
            continue
        if with_graph_only and not (paths.dataset2000_root() / "graphs" / p / "graph_enhanced.json").exists():
            continue
        if split is not None:
            try:
                if paths.split_of(p) != split:
                    continue
            except FileNotFoundError:
                continue
        out.append(p)
    return out
