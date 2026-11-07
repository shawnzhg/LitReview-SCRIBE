"""Builds the labelled event history of a published-pipeline run by merging its model, search and
retrieval logs on one clock."""

from __future__ import annotations

from pathlib import Path

from ccbench import labels
from ccbench.ingest import logs
from ccbench.model import Event


def build_events(task_dir: Path, K: int) -> tuple[list[Event], dict]:
    raw: list[dict] = []
    totals = {"in_tokens": 0, "out_tokens": 0, "usd": 0.0, "wall_ms": 0, "n_llm": 0, "n_llm_err": 0, "n_search": 0, "n_open": 0}

    for c in logs.iter_calls(task_dir):
        u = c["usage"] or {}
        t0 = c["t_req"]
        t1 = c["t_done"] or (t0 + (c["wall_ms"] or 0) / 1000.0 if t0 else None)
        raw.append({"kind": "llm", "t0": t0, "t1": t1, "rec": c, "out_tokens": u.get("completion_tokens") or 0})
        totals["in_tokens"] += u.get("prompt_tokens") or 0
        totals["out_tokens"] += u.get("completion_tokens") or 0
        totals["usd"] += c["cost_usd"] or 0.0
        totals["wall_ms"] += c["wall_ms"] or 0
        totals["n_llm"] += 1
        if c["status"] != 200:
            totals["n_llm_err"] += 1

    for s in logs.load_search(task_dir) + logs.load_tap(task_dir):
        route = s["route"]
        is_search = route in labels.SEARCH_ROUTES or route.startswith("tap:")
        kind = "search" if is_search else "open"
        t0 = s["t"]
        t1 = (t0 + (s.get("latency_ms") or 0) / 1000.0) if t0 else None
        raw.append({"kind": kind, "t0": t0, "t1": t1, "rec": s})
        totals["n_search" if is_search else "n_open"] += 1

    raw = [r for r in raw if r["t0"] is not None]
    raw.sort(key=lambda r: r["t0"])
    last_ret = max((r["t0"] for r in raw if r["kind"] != "llm"), default=None)

    events: list[Event] = []
    for i, r in enumerate(raw):
        rec = r["rec"]
        if r["kind"] == "llm":
            stage = "acquisition" if (last_ret is not None and r["t0"] <= last_ret) else "generation"
            lab = labels.label_llm(rec["status"], rec["finish_reason"], r["out_tokens"], stage, rec.get("error"))
            u = rec["usage"] or {}
            events.append(
                Event(
                    seq=i,
                    t_start=r["t0"],
                    t_end=r["t1"] or r["t0"],
                    kind="llm" if rec["status"] == 200 else "failure",
                    label=lab,
                    resources={"in_tokens": u.get("prompt_tokens") or 0, "out_tokens": u.get("completion_tokens") or 0, "wall_ms": rec["wall_ms"] or 0, "usd": rec["cost_usd"] or 0.0},
                    status=rec["status"],
                    finish_reason=rec["finish_reason"],
                    prompt_chars=rec["prompt_chars"],
                    completion_chars=rec["completion_chars"],
                )
            )
        else:
            route = rec["route"]
            n = rec.get("n_returned")
            lab = labels.label_search(route, n, K) if r["kind"] == "search" else labels.label_fetch(route, n, K)
            q = rec.get("params", {})
            query = q.get("query") or q.get("q") or (q.get("queries") or [None])[0]
            events.append(
                Event(
                    seq=i,
                    t_start=r["t0"],
                    t_end=r["t1"] or r["t0"],
                    kind=r["kind"],
                    label=lab,
                    resources={"in_tokens": 0, "out_tokens": 0, "wall_ms": int(rec.get("latency_ms") or 0), "usd": 0.0},
                    route=route,
                    n_returned=n,
                    returned_ids=rec.get("returned") or [],
                    query=query,
                )
            )
    return events, totals


def retrieved_union(events: list[Event]) -> list[str]:
    seen: dict[str, None] = {}
    for e in events:
        for p in e.returned_ids:
            seen.setdefault(p, None)
    return list(seen)
