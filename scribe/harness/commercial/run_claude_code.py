#!/usr/bin/env python3
"""Runs Claude Code headless once on one task and entry condition with claude-sonnet-5 at low effort
and the pool MCP server as its only tool, and records the request, message stream, report and cost.
Usage: python run_claude_code.py <task> <cond>."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import agent_prompt as P

OUT = Path(os.environ["COMMERCIAL_OUT"])
TASKSPECS = Path(os.environ["COMMERCIAL_TASKSPECS"])
LEDGER = OUT / "claude_costs.jsonl"
PORTS = {"fixinput": 31971, "samepool": 31972}
MODEL, EFFORT = "claude-sonnet-5", "low"
MAX_TURNS = 80


def run(task: str, cond: str) -> str:
    out = OUT / "runs" / cond / task
    if out.exists():
        return "skip"
    out.mkdir(parents=True)
    cwd = OUT / "_cwd" / f"{cond}_{task}"
    cwd.mkdir(parents=True, exist_ok=True)
    assert not any(cwd.iterdir()), f"cwd not empty: {cwd}"
    spec = json.loads((TASKSPECS / f"{task}.json").read_text())
    text = P.prompt(spec)
    tok = Path(os.environ[f"MCP_TOKEN_FILE_{cond.upper()}"]).read_text().strip()
    cfg = {"mcpServers": {"pool": {"type": "http", "url": f"http://127.0.0.1:{PORTS[cond]}/{tok[:16]}/mcp",
                                   "headers": {"Authorization": f"Bearer {tok}", "X-Task": task}}}}
    cfg_path = OUT / "_mcp_cfg" / f"{cond}_{task}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg))
    os.chmod(cfg_path, 0o600)
    cmd = ["claude", "-p", text, "--model", MODEL, "--effort", EFFORT,
           "--tools", "", "--strict-mcp-config", "--mcp-config", str(cfg_path),
           "--allowedTools", "mcp__pool__search", "mcp__pool__fetch",
           "--setting-sources", "project", "--max-turns", str(MAX_TURNS),
           "--output-format", "stream-json", "--verbose", "--no-session-persistence"]
    (out / "request.json").write_text(json.dumps({
        "cmd": [c if c != text else "<prompt>" for c in cmd], "prompt": text, "task": task, "cond": cond,
        "mcp_url": f"http://127.0.0.1:{PORTS[cond]}/<token>/mcp", "x_task": task,
        "taskspec_hash": spec["content_hash"], "cwd": str(cwd)}, indent=1))
    env = {**os.environ, "MCP_TOOL_TIMEOUT": "600000", "MCP_TIMEOUT": "60000"}
    t0 = time.time()
    with open(out / "stream.jsonl", "w") as so, open(out / "stderr.txt", "w") as se:
        rc = subprocess.run(cmd, cwd=cwd, stdout=so, stderr=se, env=env).returncode
    wall = time.time() - t0
    msgs = [json.loads(l) for l in (out / "stream.jsonl").read_text().splitlines() if l.strip()]
    init = next((m for m in msgs if m.get("type") == "system" and m.get("subtype") == "init"), {})
    res = next((m for m in reversed(msgs) if m.get("type") == "result"), {})
    calls = [c for m in msgs if m.get("type") == "assistant"
             for c in (m.get("message") or {}).get("content", []) if c.get("type") == "tool_use"]
    errs = [c for m in msgs if m.get("type") == "user"
            for c in (m.get("message") or {}).get("content", []) if isinstance(c, dict) and c.get("is_error")]
    report = res.get("result") or ""
    (out / "report.md").write_text(report)
    model_ok = init.get("model") == MODEL
    rec = {"t": time.time(), "task": task, "cond": cond, "requested_model": MODEL, "model": init.get("model"),
           "model_ok": model_ok, "effort": EFFORT, "rc": rc, "subtype": res.get("subtype"),
           "is_error": res.get("is_error"), "num_turns": res.get("num_turns"), "n_tool_calls": len(calls),
           "n_search": sum(1 for c in calls if c.get("name") == "mcp__pool__search"),
           "tools_called": sorted({c["name"] for c in calls}), "n_tool_errors": len(errs),
           "tools_available": init.get("tools"), "mcp_servers": init.get("mcp_servers"),
           "usage": res.get("usage"), "total_cost_usd": res.get("total_cost_usd"),
           "words": len(report.split()), "wall_s": round(wall, 1)}
    (out / "result.json").write_text(json.dumps(rec, indent=1))
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in ("task", "cond", "model", "subtype", "num_turns", "n_tool_calls",
                                         "tools_called", "n_tool_errors", "tools_available", "words",
                                         "total_cost_usd", "wall_s")}), flush=True)
    if not model_ok:
        print(f"FAILED {cond} {task}: the CLI ran {init.get('model')!r}, not {MODEL}", file=sys.stderr, flush=True)
        return "model_mismatch"
    return res.get("subtype") or "noresult"


if __name__ == "__main__":
    run(*sys.argv[1:3])
