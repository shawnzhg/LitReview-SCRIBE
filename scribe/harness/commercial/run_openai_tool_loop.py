#!/usr/bin/env python3
"""Runs gpt-5.6-luna once through the OpenAI Responses API tool loop on every task and entry
condition with the pool MCP server as its only tool, and records each request, response, report and
cost. Usage: python run_openai_tool_loop.py [workers]."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

import agent_prompt as P

OUT = Path(os.environ["COMMERCIAL_OUT"])
TASKSPECS = Path(os.environ["COMMERCIAL_TASKSPECS"])
TASKS = Path(os.environ["COMMERCIAL_TASKS"])
MODEL, EFFORT = "gpt-5.6-luna", "none"
PRICES = {"gpt-5.6-luna": (0.20e-6, 0.02e-6, 1.20e-6)}
PRICE_IN, PRICE_CACHED, PRICE_OUT = PRICES[MODEL]
CAP = 10.0
EST_PER_RUN = 0.2
MAX_TOOL_CALLS = 80
LEDGER = OUT / "openai_costs.jsonl"
CONDS = ("fixinput", "samepool")

client = OpenAI(max_retries=5)
lock = threading.Lock()
inflight = 0


def log(msg: str) -> None:
    with lock:
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def spent() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["cost_usd"] for l in LEDGER.read_text().splitlines() if l.strip())


def one(task: str, cond: str) -> None:
    global inflight
    out = OUT / "runs" / cond / task
    if out.exists():
        return
    spec = json.loads((TASKSPECS / f"{task}.json").read_text())
    text = P.prompt(spec)
    url = os.environ[f"MCP_URL_{cond.upper()}"].strip().rstrip("/")
    tok = Path(os.environ[f"MCP_TOKEN_FILE_{cond.upper()}"]).read_text().strip()
    tools = [{"type": "mcp", "server_label": "pool", "server_url": f"{url}/{tok[:16]}/mcp",
              "headers": {"Authorization": f"Bearer {tok}", "X-Task": task}, "require_approval": "never"}]
    with lock:
        if spent() + (inflight + 1) * EST_PER_RUN > CAP:
            raise SystemExit(f"cap guard: spent ${spent():.4f} inflight {inflight}")
        inflight += 1
    out.mkdir(parents=True)
    try:
        (out / "request.json").write_text(json.dumps(
            {"model": MODEL, "input": text, "tools": [{**tools[0], "headers": {"X-Task": task, "Authorization": "<redacted>"},
             "server_url": "<tunnel>/<token>/mcp"}], "max_tool_calls": MAX_TOOL_CALLS, "background": True,
             "reasoning_effort": EFFORT, "taskspec_hash": spec["content_hash"]}, indent=1))
        t0 = time.time()
        r = client.responses.create(model=MODEL, input=text, tools=tools, background=True,
                                    max_tool_calls=MAX_TOOL_CALLS, reasoning={"effort": EFFORT})
        (out / "response_id.txt").write_text(r.id)
        while r.status in ("queued", "in_progress"):
            time.sleep(15)
            r = client.responses.retrieve(r.id)
        d = r.model_dump()
        u = d.get("usage") or {}
        cached = ((u.get("input_tokens_details") or {}).get("cached_tokens")) or 0
        inp, outp = u.get("input_tokens", 0), u.get("output_tokens", 0)
        cost = (inp - cached) * PRICE_IN + cached * PRICE_CACHED + outp * PRICE_OUT
        n_mcp = sum(1 for it in d.get("output", []) if it.get("type", "").startswith("mcp_call"))
        words = len((r.output_text or "").split())
        n_err = sum(1 for it in d.get("output", []) if it.get("type") == "mcp_call" and it.get("error"))
        accepted = r.status == "completed" and words > 0 and n_err == 0
        rec = {"t": time.time(), "cond": cond, "effort": EFFORT, "task": task, "model": MODEL, "response_id": r.id,
               "status": r.status, "accepted": accepted, "n_mcp_errors": n_err, "input_tokens": inp,
               "cached_tokens": cached, "output_tokens": outp,
               "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens"),
               "cost_usd": round(cost, 6), "wall_s": round(time.time() - t0, 1)}
        with lock:
            with open(LEDGER, "a") as f:
                f.write(json.dumps(rec) + "\n")
        (out / "response.json").write_text(json.dumps(d, indent=1))
        (out / "report.md").write_text(r.output_text or "")
        log(f"[{cond}] {task} {r.status} accepted={accepted} words={words} mcp={n_mcp} mcp_err={n_err} ${cost:.4f} {time.time() - t0:.0f}s")
    except Exception as e:
        (out / "error.txt").write_text(traceback.format_exc())
        log(f"[{cond}] {task} ERROR {type(e).__name__}: {str(e)[:200]}")
    finally:
        with lock:
            inflight -= 1


def main(workers: int) -> None:
    tasks = json.loads(TASKS.read_text())
    jobs = [(t, c) for t in tasks for c in CONDS]
    log(f"{len(jobs)} jobs, workers={workers}, spent so far ${spent():.4f}")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: one(*j), jobs))
    done = sum((OUT / "runs" / c / t / "response.json").exists() for t, c in jobs)
    log(f"DONE {done}/{len(jobs)} with a response, total spent ${spent():.4f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 25)
