"""Task-level inputs shared by every system: the TaskSpec, the evaluation task list and its subfield
clusters."""

from __future__ import annotations

import functools
import json

from ccbench import paths
from ccbench.model import Context


@functools.lru_cache(maxsize=None)
def taskspec(task: str) -> dict:
    with open(paths.taskspec_path(task)) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def campaign50() -> dict:
    with open(paths.campaign50_json()) as f:
        return json.load(f)


def campaign50_tasks() -> list[str]:
    return list(campaign50()["tasks"])


@functools.lru_cache(maxsize=None)
def campaign50_clusters(level: str = "subfield") -> dict[str, str]:
    from ccbench.gt import peers

    out: dict[str, str] = {}
    for t in campaign50()["tasks"]:
        p = peers.per_paper().get(t) or {}
        out[t] = str(p.get(level) or t)
    return out


def context_for(task: str, pool_fingerprint: str | None = None, tools: list[str] | None = None) -> Context:
    ts = taskspec(task)
    return Context(
        task=task,
        topic=ts["question"],
        cutoff=int(ts["publication_cutoff"]),
        pool_fingerprint=pool_fingerprint,
        snapshot_id=ts.get("corpus_snapshot_id"),
        target_words=(ts.get("output_spec") or {}).get("target_words"),
        budget=ts.get("budget") or {},
        tools=tools or list(ts.get("allowed_tools") or []),
        review_type=ts.get("review_type"),
    )
