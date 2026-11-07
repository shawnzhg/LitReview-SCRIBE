"""Converts the OpenAI tool-loop runs in $COMMERCIAL_OUT into scorer runs under a fixed-input and a
same-pool key and builds their rollouts. Usage: python openai_tool_loop.py convert --out <dir> |
build --runs <dir>."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared as S
from ccbench.adapters.autosurvey import CITE

KEYS = {"fixinput": "openai_luna_mcp.ref", "samepool": "openai_luna_mcp"}
LEDGER = "openai_costs.jsonl"
WIN_PAD_S = (5.0, 30.0)


def accepted_calls(resp: dict) -> list[dict]:
    out = []
    for o in resp.get("output") or []:
        if o.get("type") != "mcp_call":
            continue
        args = json.loads(o.get("arguments") or "{}")
        body = json.loads(o["output"]) if o.get("output") else {}
        if o.get("name") == "search":
            ids = [str(r["id"]) for r in body.get("results") or []]
            out.append({"tool": "search", "arg": args.get("query"), "ids": ids, "error": o.get("error"), "call_id": o.get("id")})
        elif o.get("name") == "fetch":
            ids = [str(body["id"])] if body.get("id") is not None else []
            out.append({"tool": "fetch", "arg": str(args.get("id")), "ids": ids, "error": o.get("error"), "call_id": o.get("id")})
        else:
            out.append({"tool": o.get("name"), "arg": None, "ids": [], "error": "unknown tool", "call_id": o.get("id")})
    return out


def match_calls(calls: list[dict], recs: list[dict], t0: float, t1: float) -> tuple[list[dict], dict]:
    win = [r for r in recs if t0 - WIN_PAD_S[0] <= r["t"] <= t1 + WIN_PAD_S[1]]
    if not win and calls:
        n = len(calls)
        ev = [{**c, "t": t0 + (i + 1) * (t1 - t0) / (n + 1), "log_line": -1, "suppressed": [], "n_pool_hits": None,
               "ts_source": "response_order_synthetic"} for i, c in enumerate(calls)]
        return ev, {"n_calls": n, "n_matched": 0, "unmatched": [], "returned_mismatch": 0, "log_lines_task": len(recs),
                    "log_lines_in_window": 0, "log_lines_in_window_unused": 0, "log_lines_outside_window": len(recs),
                    "fallback_response_only": True}
    used: set[int] = set()
    ev, unmatched, mism = [], [], 0
    for c in calls:
        key = "query" if c["tool"] == "search" else "id"
        cand = [r for r in win if r["tool"] == c["tool"] and str(r.get(key)) == str(c["arg"]) and r["_line"] not in used]
        cand.sort(key=lambda r: r["t"])
        hit = next((r for r in cand if [str(x) for x in r.get("returned") or []] == c["ids"]), None)
        if hit is None:
            unmatched.append({"tool": c["tool"], "arg": c["arg"], "n_ids": len(c["ids"]), "n_cand": len(cand)})
            if cand:
                mism += 1
            continue
        used.add(hit["_line"])
        ev.append({**c, "t": hit["t"], "log_line": hit["_line"], "suppressed": hit.get("suppressed") or [],
                   "n_pool_hits": hit.get("n_pool_hits")})
    ev.sort(key=lambda e: (e["t"], e["log_line"]))
    stats = {"fallback_response_only": False, "n_calls": len(calls), "n_matched": len(ev), "unmatched": unmatched, "returned_mismatch": mism,
             "log_lines_task": len(recs), "log_lines_in_window": len(win),
             "log_lines_in_window_unused": len([r for r in win if r["_line"] not in used]),
             "log_lines_outside_window": len(recs) - len(win)}
    return ev, stats


def convert(out: Path) -> int:
    ledger = S.jsonl(S.X / LEDGER)
    tasks = S.run_tasks()
    summary = {"schema": "openai_tool_loop_convert/1", "source": str(S.X), "conds": {},
               "source_sha256": S.source_hashes((LEDGER, "mcp_call_logs/mcp_fixinput.jsonl", "mcp_call_logs/mcp_samepool.jsonl"))}
    for cond, key in KEYS.items():
        logs = S.load_mcp_log(f"mcp_{cond}.jsonl")
        arm = out / cond
        (arm / "_arm").mkdir(parents=True)
        st_lines, per = [], {}
        for task in tasks:
            src = S.X / "runs" / cond / task
            td = arm / task
            td.mkdir()
            resp = json.load(open(src / "response.json"))
            req = json.load(open(src / "request.json"))
            calls = accepted_calls(resp)
            t0, t1 = float(resp["created_at"]), float(resp.get("completed_at") or resp["created_at"])
            ev, ms = match_calls(calls, logs.get(task, []), t0, t1)
            with open(td / "search_calls.jsonl", "w") as f:
                for e in ev:
                    if e["tool"] == "search":
                        rec = {"ts": S.iso(e["t"]), "task": task, "route": "s2_search",
                               "params": {"query": e["arg"], "k": S.MCP_K, "native_route": "plain_search", "mcp_tool": "search"},
                               "n_returned": len(e["ids"]), "returned": [{"pmid": p} for p in e["ids"]], "latency_ms": None,
                               "suppressed": e["suppressed"], "n_pool_hits": e["n_pool_hits"], "mcp_call_id": e["call_id"],
                               "ts_source": e.get("ts_source", "mcp_server_log")}
                    else:
                        rec = {"ts": S.iso(e["t"]), "task": task, "route": "paper_by_id",
                               "params": {"id": e["arg"], "native_route": "graph/v1/paper/PMID:{id}", "mcp_tool": "fetch"},
                               "n_returned": len(e["ids"]), "returned": [{"pmid": p} for p in e["ids"]], "latency_ms": None,
                               "mcp_call_id": e["call_id"], "ts_source": e.get("ts_source", "mcp_server_log")}
                    f.write(json.dumps(rec) + "\n")
            att = [r for r in ledger if r.get("cond") == cond and r.get("task") == task and r.get("effort", "none") == "none"]
            acc = [r for r in att if r.get("response_id") == resp["id"]]
            if len(acc) != 1:
                raise SystemExit(f"INGEST FAILED: {cond}/{task}: {len(acc)} ledger rows for the accepted response {resp['id']}")
            acc = acc[0]
            u = resp.get("usage") or {}
            text = "".join(c.get("text", "") for o in resp["output"] if o.get("type") == "message" for c in o.get("content") or [])
            call = {"seq": 0, "t_req": t0, "t_done": t1, "wall_ms": int(round((t1 - t0) * 1000)), "status": 200,
                    "path": "/v1/responses (background, server-side MCP tool loop)", "model": resp.get("model"),
                    "requested_model": req.get("model"), "finish_reason": resp.get("status"),
                    "usage": {"prompt_tokens": u.get("input_tokens"), "completion_tokens": u.get("output_tokens"),
                              "reasoning_tokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens")},
                    "cost_usd": acc.get("cost_usd"), "prompt": req.get("input") or "", "completion": text,
                    "params": {"reasoning_effort": (resp.get("reasoning") or {}).get("effort"), "temperature": resp.get("temperature"),
                               "max_tool_calls": resp.get("max_tool_calls"), "response_id": resp["id"]}}
            (td / "_calls.jsonl").write_text(json.dumps(call) + "\n")
            S.pool_health(td)
            shutil.copy2(src / "report.md", td / "report.md")
            prov = {"source_dir": str(src), "response_id": resp["id"], "report_sha256": S.sha256(src / "report.md"),
                    "response_sha256": S.sha256(src / "response.json"), "status": resp.get("status"),
                    "text_equals_report": text.strip() == (src / "report.md").read_text().strip(), "match": ms,
                    "mcp_errors": sum(1 for c in calls if c["error"]), "attempts_effort_none": len(att)}
            (td / "provenance.json").write_text(json.dumps(prov, indent=1))
            for r in att:
                if r is acc:
                    continue
                st_lines.append({"task": task, "status": "not_accepted", "note": "a ledger attempt other than the accepted response",
                                 "response_id": r.get("response_id"), "wall_s": r.get("wall_s")})
            st_lines.append({"task": task, "status": "ok", "wall_s": acc.get("wall_s") or round(t1 - t0, 1),
                             "response_id": resp["id"]})
            per[task] = {"fallback_response_only": ms["fallback_response_only"], "matched": ms["n_matched"], "calls": ms["n_calls"], "unmatched": len(ms["unmatched"]),
                         "outside_window": ms["log_lines_outside_window"], "in_window_unused": ms["log_lines_in_window_unused"],
                         "attempts": len(att), "text_equals_report": prov["text_equals_report"], "status": resp.get("status"),
                         "mcp_errors": prov["mcp_errors"]}
        S.write_status(arm, st_lines)
        summary["conds"][cond] = {"key": key, "tasks": len(per),
                                  "calls": sum(v["calls"] for v in per.values()), "matched": sum(v["matched"] for v in per.values()),
                                  "unmatched": sum(v["unmatched"] for v in per.values()),
                                  "fallback_response_only": [t for t, v in per.items() if v["fallback_response_only"]],
                                  "log_lines_outside_accepted_windows": sum(v["outside_window"] for v in per.values()),
                                  "log_lines_in_window_unused": sum(v["in_window_unused"] for v in per.values()),
                                  "attempts_total": sum(v["attempts"] for v in per.values()),
                                  "not_completed": [t for t, v in per.items() if v["status"] != "completed"],
                                  "mcp_errors": sum(v["mcp_errors"] for v in per.values()),
                                  "text_differs_from_report": [t for t, v in per.items() if not v["text_equals_report"]],
                                  "per_task": per}
    S.write_tasks(out, tasks)
    (out / "convert_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({c: {k: v for k, v in s.items() if k != "per_task"} for c, s in summary["conds"].items()}, indent=1))
    bad = [c for c, s in summary["conds"].items() if s["unmatched"] or s["not_completed"] or s["mcp_errors"]]
    return 1 if bad else 0


def adapt(task: str, task_dir: Path, key: str, cond: str):
    from ccbench.adapters import common
    from ccbench.ingest import logs
    rp = task_dir / "report.md"
    if not rp.exists():
        return common.bot(key, task, task_dir, "report.md missing")
    text = rp.read_text(errors="replace")
    if not text.strip():
        return common.bot(key, task, task_dir, "report.md empty")
    refmap, rst = S.references(text)
    report = common.report_from_markdown(text, CITE, common.numeric_resolver(refmap))
    outline = common.outline_from_markdown(text)
    papers: dict[str, None] = {}
    for s in logs.load_search(task_dir):
        for p in s["returned"]:
            papers.setdefault(p, None)
    ro = common.assemble(key, task, task_dir, papers=list(papers), outline=outline, report=report, graph=None, final_ok=True,
                         bot_reason=None, extra_meta={"n_reference_entries": rst["n_entries"], "reference_stats": rst,
                                                      "adapter": "evaluation/board/ingest/openai_tool_loop.py:adapt", "draft_is_final": True})
    return ro


def main(argv=None) -> int:
    return S.main(argv, __doc__, convert, adapt, KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
