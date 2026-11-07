"""Branch-label map on the label space; the labels of published pipelines are derived from their
proxy and pool records."""

from __future__ import annotations

import json

from ccbench.config import prereg
from ccbench.model import Label

FINE_TO_COARSE = {"acquisition": "acquisition", "synthesis": "generation", "planning": "generation", "writing": "generation"}

SEARCH_ROUTES = {"s2_search", "s2_snippet_search", "bing_search"}


def resource_bin(n: int | None) -> str:
    n = int(n or 0)
    if n <= 0:
        return "0"
    if n <= 16:
        return "1-16"
    if n <= 64:
        return "17-64"
    if n <= 256:
        return "65-256"
    if n <= 1024:
        return "257-1024"
    return ">1024"


def results_effect(n: int | None, K: int) -> str:
    n = int(n or 0)
    if n == 0:
        return "results_none"
    if n <= 0.2 * K:
        return "results_few"
    if n <= 0.8 * K:
        return "results_some"
    return "results_many"


def sentences_effect(n: int | None) -> str:
    n = int(n or 0)
    if n == 0:
        return "sentences_none"
    return "sentences_few" if n <= 8 else "sentences_many"


def label_llm(status: int | None, finish_reason: str | None, out_tokens: int | None, stage: str, error: str | None = None) -> Label:
    if status is not None and status != 200:
        err = error if isinstance(error, str) else (json.dumps(error)[:80] if error else "")
        return Label(stage, "failure", "policy_llm", resource_bin(0), "aborted", retry_or_failure=(err or f"http_{status}")[:40])
    eff = "truncated" if finish_reason == "length" else "completion"
    return Label(stage, "generate", "policy_llm", resource_bin(out_tokens), eff, retry_or_failure=("length" if finish_reason == "length" else None))


def label_search(route: str, n_returned: int | None, K: int) -> Label:
    return Label("acquisition", "search", "pool_search", resource_bin(n_returned), results_effect(n_returned, K))


def label_fetch(route: str, n_returned: int | None, K: int) -> Label:
    if route in ("page", "pdf"):
        eff = sentences_effect(999 if (n_returned or 0) > 0 else 0)
    else:
        eff = results_effect(n_returned, K)
    return Label("acquisition", "open_document", "pool_fetch", resource_bin(n_returned), eff)


def label_from_trace(rec: dict) -> Label:
    bl = rec.get("branch_label") or {}
    return Label(
        stage=bl.get("stage") or rec.get("window") or "acquisition",
        action_class=bl.get("action_class") or rec.get("kind") or "generate",
        tool_class=bl.get("tool_class") or "none",
        resource_bin=bl.get("resource_bin") or "0",
        observable_effect=bl.get("observable_effect") or "completion",
        retry_or_failure=bl.get("retry_or_failure"),
        novelty=bl.get("novelty"),
    )


def coarsen(lab: Label) -> Label:
    return Label(
        stage=FINE_TO_COARSE.get(lab.stage, lab.stage),
        action_class=lab.action_class,
        tool_class=lab.tool_class,
        resource_bin=lab.resource_bin,
        observable_effect=lab.observable_effect,
        retry_or_failure=lab.retry_or_failure,
        novelty=lab.novelty,
    )


def d_L(a: Label, b: Label, weights: dict | None = None) -> float:
    w = weights or prereg()["controllability"]["label_weights"]
    d = 0.0
    for dim, wt in w.items():
        if getattr(a, dim) != getattr(b, dim):
            d += wt
    return min(1.0, d / sum(w.values()))
