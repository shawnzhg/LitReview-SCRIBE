"""Converts the Claude Code runs in $COMMERCIAL_OUT into scorer runs under a fixed-input and a
same-pool key and builds their rollouts. Usage: python claude_code.py convert --out <dir> | build
--runs <dir>."""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared as S
from ccbench.adapters.autosurvey import CITE

KEYS = {"fixinput": "claude_sonnet_mcp.ref", "samepool": "claude_sonnet_mcp"}
LEDGER = "claude_costs.jsonl"
WIN_PAD_S = (5.0, 30.0)


def read_stream(path: Path) -> dict:
    uses: dict[str, dict] = {}
    calls: dict[str, dict] = {}
    order: list[str] = []
    init, refusal, n_asst, result, t_last = None, [], 0, None, None
    for l in open(path):
        m = json.loads(l)
        ts = m.get("timestamp")
        if ts:
            t_last = max(t_last or 0.0, S.epoch(ts))
        if m.get("type") == "result":
            result = {k: m.get(k) for k in ("duration_ms", "duration_api_ms", "num_turns", "subtype", "is_error", "stop_reason",
                                            "terminal_reason", "api_error_status")}
        if m.get("type") == "system" and m.get("subtype") == "init":
            init = {"model": m.get("model"), "tools": m.get("tools"), "claude_code_version": m.get("claude_code_version"),
                    "mcp_servers": m.get("mcp_servers")}
        if m.get("type") == "system" and "refusal" in str(m.get("subtype") or ""):
            refusal.append(m.get("subtype"))
        if m.get("type") == "assistant":
            n_asst += 1
            for c in (m.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    uses[c["id"]] = c
                    order.append(c["id"])
        if m.get("type") == "user" and isinstance((m.get("message") or {}).get("content"), list):
            for c in m["message"]["content"]:
                if not (isinstance(c, dict) and c.get("type") == "tool_result"):
                    continue
                u = uses.get(c.get("tool_use_id"))
                if u is None:
                    raise SystemExit(f"INGEST FAILED: {path}: tool_result for an unknown tool_use {c.get('tool_use_id')}")
                cc = c.get("content")
                txt = cc if isinstance(cc, str) else "".join(x.get("text", "") for x in cc or [] if isinstance(x, dict))
                name = u["name"].split("__")[-1]
                err = "tool_result is_error" if c.get("is_error") else None
                try:
                    body = json.loads(txt)
                except Exception:
                    body, err = {}, err or "tool_result is not JSON"
                if name == "search":
                    ids = [str(r["id"]) for r in body.get("results") or []]
                    arg = (u.get("input") or {}).get("query")
                elif name == "fetch":
                    ids = [str(body["id"])] if body.get("id") is not None else []
                    arg = str((u.get("input") or {}).get("id"))
                else:
                    ids, arg, err = [], None, "unknown tool"
                if not ts:
                    raise SystemExit(f"INGEST FAILED: {path}: tool_result without a timestamp")
                calls[u["id"]] = {"tool": name, "arg": arg, "ids": ids, "error": err, "call_id": u["id"], "t": S.epoch(ts)}
    missing = [i for i in order if i not in calls]
    return {"init": init, "calls": [calls[i] for i in order if i in calls], "tool_use_without_result": missing,
            "refusal": refusal, "n_assistant": n_asst, "result": result, "t_last": t_last}


def match_calls(calls: list[dict], recs: list[dict], t0: float, t1: float) -> tuple[list[dict], dict]:
    win = [r for r in recs if t0 - WIN_PAD_S[0] <= r["t"] <= t1 + WIN_PAD_S[1]]
    used: set[int] = set()
    ev, unmatched, mism = [], [], 0
    for c in calls:
        key = "query" if c["tool"] == "search" else "id"
        cand = [r for r in win if r["tool"] == c["tool"] and str(r.get(key)) == str(c["arg"]) and r["_line"] not in used]
        cand.sort(key=lambda r: abs(r["t"] - c["t"]))
        hit = next((r for r in cand if [str(x) for x in r.get("returned") or []] == c["ids"]), None)
        if hit is None:
            unmatched.append({"tool": c["tool"], "arg": c["arg"], "n_ids": len(c["ids"]), "n_cand": len(cand)})
            if cand:
                mism += 1
            ev.append({**c, "server_t": None, "log_line": None, "suppressed": [], "n_pool_hits": None})
            continue
        used.add(hit["_line"])
        ev.append({**c, "server_t": hit["t"], "log_line": hit["_line"], "suppressed": hit.get("suppressed") or [],
                   "n_pool_hits": hit.get("n_pool_hits")})
    lag = [e["t"] - e["server_t"] for e in ev if e["server_t"] is not None]
    stats = {"n_calls": len(calls), "n_matched": len(calls) - len(unmatched), "unmatched": unmatched, "returned_mismatch": mism,
             "log_lines_task": len(recs), "log_lines_in_window": len(win),
             "log_lines_in_window_unused": len([r for r in win if r["_line"] not in used]),
             "log_lines_outside_window": len(recs) - len(win),
             "stream_minus_server_s": [round(min(lag), 3), round(max(lag), 3)] if lag else None}
    return ev, stats


def is_refusal(st: dict, res: dict, text: str) -> str | None:
    why = []
    if st["refusal"]:
        why.append(f"stream system {st['refusal']}")
    if res.get("is_error"):
        why.append("result.json is_error=true (subtype " + str(res.get("subtype")) + ")")
    if text.lstrip().startswith("API Error:"):
        m = re.search(r"Details:\s*`?(\[[^\]]+\])", text)
        why.append(f"report 'API Error: ...' {m.group(1) if m else ''}".strip())
    return "vendor_refusal: " + "; ".join(why) if why else None


def convert(out: Path) -> int:
    ledger = S.jsonl(S.X / LEDGER)
    tasks = S.run_tasks()
    summary = {"schema": "claude_code_convert/1", "source": str(S.X), "conds": {},
               "source_sha256": S.source_hashes((LEDGER, "audit.json", "mcp_call_logs/mcp_fixinput.jsonl",
                                                 "mcp_call_logs/mcp_samepool.jsonl"))}
    for cond, key in KEYS.items():
        logs = S.load_mcp_log(f"mcp_{cond}.jsonl")
        arm = out / cond
        (arm / "_arm").mkdir(parents=True)
        st_lines, per = [], {}
        for task in tasks:
            src = S.X / "runs" / cond / task
            td = arm / task
            td.mkdir()
            res = json.load(open(src / "result.json"))
            req = json.load(open(src / "request.json"))
            st = read_stream(src / "stream.jsonl")
            text = (src / "report.md").read_text(errors="replace")
            t1 = float(res["t"])
            t0 = t1 - float(res.get("wall_s") or 0.0)
            ev, ms = match_calls(st["calls"], logs.get(task, []), t0, t1)
            if not st["result"] or st["result"].get("duration_ms") is None or st["t_last"] is None:
                raise SystemExit(f"INGEST FAILED: {cond}/{task}: no result message with duration_ms in the stream")
            run_s = float(st["result"]["duration_ms"]) / 1000.0
            r_done = st["t_last"]
            r_req = r_done - run_s
            with open(td / "search_calls.jsonl", "w") as f:
                for e in ev:
                    base = {"ts": S.iso(e["t"]), "task": task, "n_returned": len(e["ids"]), "returned": [{"pmid": p} for p in e["ids"]],
                            "latency_ms": None, "mcp_call_id": e["call_id"], "ts_source": "stream_tool_result",
                            "server_t": e["server_t"], "server_log_line": e["log_line"]}
                    if e["tool"] == "search":
                        rec = {**base, "route": "s2_search",
                               "params": {"query": e["arg"], "k": S.MCP_K, "native_route": "plain_search", "mcp_tool": "search"},
                               "suppressed": e["suppressed"], "n_pool_hits": e["n_pool_hits"]}
                    else:
                        rec = {**base, "route": "paper_by_id",
                               "params": {"id": e["arg"], "native_route": "graph/v1/paper/PMID:{id}", "mcp_tool": "fetch"}}
                    f.write(json.dumps(rec) + "\n")
            att = [r for r in ledger if r.get("cond") == cond and r.get("task") == task]
            acc = [r for r in att if r.get("t") == res["t"]]
            if len(acc) != 1:
                raise SystemExit(f"INGEST FAILED: {cond}/{task}: {len(acc)} ledger rows for the accepted run t={res['t']}")
            acc = acc[0]
            u = res.get("usage") or {}
            pin = sum(int(u.get(k) or 0) for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
            call = {"seq": 0, "t_req": r_req, "t_done": r_done, "wall_ms": int(st["result"]["duration_ms"]), "status": 200,
                    "path": "claude -p (Claude Code CLI headless agent loop; --tools '' + one MCP server)",
                    "model": (st["init"] or {}).get("model"), "requested_model": res.get("model"),
                    "finish_reason": res.get("subtype"),
                    "usage": {"prompt_tokens": pin, "completion_tokens": u.get("output_tokens"),
                              "reasoning_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens")},
                    "cost_usd": res.get("total_cost_usd"), "prompt": req.get("prompt") or "", "completion": text,
                    "params": {"effort": res.get("effort"), "model_alias": res.get("model_alias"), "num_turns": res.get("num_turns"),
                               "n_tool_calls": res.get("n_tool_calls"), "input_tokens": u.get("input_tokens"),
                               "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
                               "cache_read_input_tokens": u.get("cache_read_input_tokens"),
                               "server_tool_use": u.get("server_tool_use"),
                               "claude_code_version": (st["init"] or {}).get("claude_code_version"),
                               "duration_api_ms": st["result"].get("duration_api_ms"), "harness_wall_s": res.get("wall_s"),
                               "harness_t_end": res.get("t"),
                               "cost_note": "Claude Code's own API-price estimate"}}
            (td / "_calls.jsonl").write_text(json.dumps(call) + "\n")
            S.pool_health(td)
            shutil.copy2(src / "report.md", td / "report.md")
            refusal = is_refusal(st, res, text)
            prov = {"source_dir": str(src), "result_t": res["t"], "report_sha256": S.sha256(src / "report.md"),
                    "stream_sha256": S.sha256(src / "stream.jsonl"), "result_sha256": S.sha256(src / "result.json"),
                    "subtype": res.get("subtype"), "is_error": res.get("is_error"), "init": st["init"], "match": ms,
                    "tool_use_without_result": st["tool_use_without_result"], "vendor_refusal": refusal,
                    "stream_calls": len(st["calls"]), "result_n_tool_calls": res.get("n_tool_calls"),
                    "mcp_errors": sum(1 for c in st["calls"] if c["error"]), "attempts": len(att),
                    "P_stream": len({p for c in st["calls"] for p in c["ids"]}), "cli_result": st["result"],
                    "run_s_cli": run_s, "harness_wall_s": res.get("wall_s"), "harness_minus_cli_s": round(res.get("wall_s", 0) - run_s, 1)}
            (td / "provenance.json").write_text(json.dumps(prov, indent=1))
            for r in att:
                if r is acc:
                    continue
                st_lines.append({"task": task, "status": "not_accepted", "note": "a ledger attempt other than the accepted run",
                                 "t": r.get("t"), "wall_s": r.get("wall_s")})
            if refusal:
                st_lines.append({"task": task, "status": "failed", "note": f"{refusal}: recorded as a failure",
                                 "wall_s": round(run_s, 3), "harness_wall_s": res.get("wall_s"), "t": res["t"]})
            else:
                st_lines.append({"task": task, "status": "ok", "wall_s": round(run_s, 3), "harness_wall_s": res.get("wall_s"),
                                 "t": res["t"]})
            per[task] = {"matched": ms["n_matched"], "calls": ms["n_calls"], "unmatched": len(ms["unmatched"]),
                         "calls_equal_result": len(st["calls"]) == res.get("n_tool_calls"),
                         "outside_window": ms["log_lines_outside_window"], "in_window_unused": ms["log_lines_in_window_unused"],
                         "lag": ms["stream_minus_server_s"], "harness_minus_cli_s": prov["harness_minus_cli_s"],
                         "attempts": len(att), "vendor_refusal": refusal, "mcp_errors": prov["mcp_errors"],
                         "tool_use_without_result": len(st["tool_use_without_result"]),
                         "init_tools": (st["init"] or {}).get("tools"), "model": (st["init"] or {}).get("model")}
        S.write_status(arm, st_lines)
        lags = [x for v in per.values() if v["lag"] for x in v["lag"]]
        summary["conds"][cond] = {"key": key, "tasks": len(per),
                                  "calls": sum(v["calls"] for v in per.values()), "matched": sum(v["matched"] for v in per.values()),
                                  "unmatched": sum(v["unmatched"] for v in per.values()),
                                  "calls_differ_from_result_n_tool_calls": [t for t, v in per.items() if not v["calls_equal_result"]],
                                  "stream_minus_server_s_range": [min(lags), max(lags)] if lags else None,
                                  "runs_harness_wall_exceeds_cli_by_60s": sum(1 for v in per.values() if v["harness_minus_cli_s"] > 60),
                                  "log_lines_outside_accepted_windows": sum(v["outside_window"] for v in per.values()),
                                  "log_lines_in_window_unused": sum(v["in_window_unused"] for v in per.values()),
                                  "attempts_total": sum(v["attempts"] for v in per.values()),
                                  "vendor_refusals": {t: v["vendor_refusal"] for t, v in per.items() if v["vendor_refusal"]},
                                  "mcp_errors": sum(v["mcp_errors"] for v in per.values()),
                                  "tool_use_without_result": sum(v["tool_use_without_result"] for v in per.values()),
                                  "init_tools": sorted({str(v["init_tools"]) for v in per.values()}),
                                  "models": sorted({str(v["model"]) for v in per.values()}),
                                  "per_task": per}
    S.write_tasks(out, tasks)
    (out / "convert_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({c: {k: v for k, v in s.items() if k != "per_task"} for c, s in summary["conds"].items()}, indent=1))
    bad = [c for c, s in summary["conds"].items() if s["unmatched"] or s["mcp_errors"] or s["tool_use_without_result"]
           or s["calls_differ_from_result_n_tool_calls"] or s["models"] != ["claude-sonnet-5"]]
    return 1 if bad else 0


def adapt(task: str, task_dir: Path, key: str, cond: str, cstats: Counter | None = None):
    from ccbench.adapters import common
    from ccbench.ingest import logs
    status = logs.load_status(task_dir.parent).get(task, {})
    if status.get("status") == "failed":
        return common.bot(key, task, task_dir, "no_report: " + str(status.get("note") or "failed"))
    rp = task_dir / "report.md"
    if not rp.exists():
        return common.bot(key, task, task_dir, "no_report: report.md missing")
    raw = rp.read_text(errors="replace")
    if not raw.strip():
        return common.bot(key, task, task_dir, "no_report: report.md empty")
    text, pre = S.strip_preamble(raw)
    refmap, rst = S.references(text)
    cst = cstats if cstats is not None else Counter()
    report = common.report_from_markdown(text, CITE, S.cite_resolver(refmap, cst))
    outline = common.outline_from_markdown(text)
    papers: dict[str, None] = {}
    for s in logs.load_search(task_dir):
        for p in s["returned"]:
            papers.setdefault(p, None)
    ro = common.assemble(key, task, task_dir, papers=list(papers), outline=outline, report=report, graph=None, final_ok=True,
                         bot_reason=None, extra_meta={"n_reference_entries": rst["n_entries"], "reference_stats": rst,
                                                      "preamble": pre, "citation_tokens": dict(cst),
                                                      "adapter": "evaluation/board/ingest/claude_code.py:adapt", "draft_is_final": True,
                                                      "backbone": "claude-sonnet-5",
                                                      "agent": "Claude Code CLI headless, effort low"})
    return ro


def main(argv=None) -> int:
    return S.main(argv, __doc__, convert, adapt, KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
