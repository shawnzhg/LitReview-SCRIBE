"""Streaming loaders for the per-run model, search and retrieval logs and the task status records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iso_to_epoch(ts: str) -> float:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(timezone.utc).timestamp()


def iter_calls(task_dir: Path) -> Iterator[dict]:
    for r in iter_jsonl(task_dir / "_calls.jsonl"):
        msgs = r.get("messages")
        prompt = r.get("prompt")
        pchars = sum(len(m.get("content") or "") for m in msgs) if msgs else len(prompt or "")
        comp = r.get("completion") or ""
        yield {
            "seq": r.get("seq"),
            "t_req": r.get("t_req") or (iso_to_epoch(r["t"]) if r.get("t") else None),
            "t_done": r.get("t_done"),
            "wall_ms": r.get("wall_ms"),
            "status": r.get("status"),
            "path": r.get("path"),
            "model": r.get("model"),
            "requested_model": r.get("requested_model"),
            "finish_reason": r.get("finish_reason"),
            "usage": r.get("usage") or {},
            "cost_usd": r.get("cost_usd") or 0.0,
            "error": r.get("error"),
            "prompt_chars": pchars,
            "completion_chars": len(comp),
            "params": r.get("params") or {},
        }


def load_search(task_dir: Path) -> list[dict]:
    out = []
    for r in iter_jsonl(task_dir / "search_calls.jsonl"):
        out.append(
            {
                "t": iso_to_epoch(r["ts"]) if r.get("ts") else None,
                "route": r.get("route"),
                "params": r.get("params") or {},
                "cutoff": r.get("cutoff"),
                "n_returned": r.get("n_returned"),
                "returned": [str(x["pmid"]) for x in (r.get("returned") or []) if x and x.get("pmid") is not None],
                "latency_ms": r.get("latency_ms"),
                "total": r.get("total"),
            }
        )
    return out


def load_tap(task_dir: Path) -> list[dict]:
    out = []
    for r in iter_jsonl(task_dir / "retrieval_tap.jsonl"):
        qs = r.get("queries") or ([r["query"]] if r.get("query") else [])
        out.append(
            {
                "t": iso_to_epoch(r["ts"]) if r.get("ts") else None,
                "route": f"tap:{r.get('cls')}.{r.get('method')}",
                "params": {"queries": qs, "num": r.get("num")},
                "n_returned": r.get("n_ids"),
                "returned": [str(x) for x in (r.get("ids") or [])],
                "latency_ms": r.get("ms"),
            }
        )
    return out


def load_status(arm_dir: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    n_attempts: dict[str, int] = {}
    for r in iter_jsonl(arm_dir / "_arm" / "task_status.jsonl"):
        t = r.get("task")
        if not t:
            continue
        n_attempts[t] = n_attempts.get(t, 0) + 1
        latest[t] = r
    for t, r in latest.items():
        r["n_attempts"] = n_attempts[t]
    return latest
