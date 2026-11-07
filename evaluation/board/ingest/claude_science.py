"""Converts the Claude Science runs in $COMMERCIAL_OUT into scorer runs under a fixed-input and a
same-pool key and builds their rollouts. Usage: python claude_science.py convert --out <dir> |
build --runs <dir>."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared as S

KEYS = {"fixinput": "claude_science_mcp.ref", "samepool": "claude_science_mcp"}
WIN_PAD_S = (5.0, 5.0)
PRODUCT = "Claude Science"
PROMPTS = os.environ.get("CS_PROMPTS")

CITE_TAG = re.compile(r"[(<]cite index=\"([^\"]*)\">")
CITE_ALL = re.compile(r"(\[[\d,\s;–-]+\]|[(<]cite index=\"[^\"]*\">)")
CITE_CLOSE = re.compile(r"</cite>")


def load_mcp_log() -> tuple[dict[tuple[str, str], list[dict]], Counter]:
    by: dict[tuple[str, str], list[dict]] = {}
    other: Counter = Counter()
    for i, l in enumerate(open(S.X / "mcp_call_logs/mcp_cs.jsonl")):
        d = json.loads(l)
        if not d.get("task"):
            other[str(d.get("tool"))] += 1
            continue
        d["_line"] = i
        d["t"] = float(d["t"])
        by.setdefault((d["cond"], d["task"]), []).append(d)
    return by, other


def frame_window(fr: dict) -> tuple[float, float]:
    return fr["created_at"] / 1000.0, (fr["completed_at"] or fr["updated_at"]) / 1000.0


def messages_text(p: Path) -> str:
    return p.read_text(errors="replace")


def is_refusal(fr: dict, has_report: bool, page: str) -> str | None:
    why = []
    if fr.get("status") != "completed":
        why.append(f"frame status {fr.get('status')!r}")
    if not has_report:
        why.append("no report delivered")
    if re.search(r"safeguards flagged this message", page):
        why.append("page: the safeguards flagged this message")
    return "vendor_refusal: " + "; ".join(why) if why else None


def body_before_refs(text: str) -> str:
    lines = text.splitlines()
    cut = len(lines)
    for i, l in enumerate(lines):
        if S.is_ref_heading(l):
            cut = i
    return "\n".join(lines[:cut])


def strip_cite_close(text: str) -> tuple[str, int]:
    n = len(CITE_CLOSE.findall(text))
    return CITE_CLOSE.sub("", text), n


def tag_resolver(refmap: dict[str, list[str]], stats: Counter):

    def _r(payload: str):
        out: list[str] = []
        for tok in payload.split(","):
            tok = tok.strip()
            if not tok:
                continue
            m = re.match(r"^(\d+)-(\d+)$", tok) or re.match(r"^(\d+)$", tok)
            if not m:
                stats["tag_tok_other"] += 1
                continue
            stats["tag_tok"] += 1
            if len(m.group(1)) >= S.PMID_MIN_DIGITS:
                stats["tag_tok_pmid"] += 1
                p = str(int(m.group(1)))
                if p not in out:
                    out.append(p)
                continue
            r = refmap.get(str(int(m.group(1))))
            if r:
                out.extend(p for p in r if p not in out)
            else:
                stats["tag_tok_not_in_refs"] += 1
        return out or None

    return _r


def mark_resolver(refmap: dict[str, list[str]], stats: Counter):
    br, tg = S.cite_resolver(refmap, stats), tag_resolver(refmap, stats)

    def _r(mark: str):
        if mark.startswith("["):
            stats["marks_bracket"] += 1
            return br(mark[1:-1])
        m = CITE_TAG.match(mark)
        stats["marks_tag"] += 1
        return tg(m.group(1)) if m else None

    return _r


def prepare(raw: str) -> tuple[str, dict, int]:
    text, pre = S.strip_preamble(raw)
    text, n_close = strip_cite_close(text)
    return text, pre, n_close


def inline_cited(text: str, refmap: dict[str, list[str]]) -> tuple[set[str], int, int, Counter]:
    cst: Counter = Counter()
    rsv = mark_resolver(refmap, cst)
    cited: set[str] = set()
    n = un = 0
    for mk in CITE_ALL.findall(body_before_refs(text)):
        n += 1
        r = rsv(mk)
        if r is None:
            un += 1
        else:
            cited.update(r)
    return cited, n, un, cst


def convert(out: Path) -> int:
    if not PROMPTS:
        raise SystemExit("set CS_PROMPTS to the prompt directory the runs used")
    tasks = S.run_tasks()
    logs, other = load_mcp_log()
    summary = {"schema": "claude_science_convert/1", "source": str(S.X), "mcp_log_rows_without_task": dict(other), "conds": {},
               "source_sha256": S.source_hashes(("audit.json", "mcp_call_logs/mcp_cs.jsonl"))}
    for cond, key in KEYS.items():
        arm = out / cond
        (arm / "_arm").mkdir(parents=True)
        st_lines, per = [], {}
        for task in tasks:
            src = S.X / "runs" / cond / S.stem(tasks, task)
            td = arm / task
            td.mkdir()
            fr = json.load(open(src / "frame.json"))
            t0, t1 = frame_window(fr)
            recs = sorted(logs.get((cond, task), []), key=lambda r: (r["t"], r["_line"]))
            win = [r for r in recs if t0 - WIN_PAD_S[0] <= r["t"] <= t1 + WIN_PAD_S[1]]
            win_lines = {r["_line"] for r in win}
            outside = [r for r in recs if r["_line"] not in win_lines]
            msgs = messages_text(src / "messages.jsonl")
            qs = [r["query"] for r in win if r["tool"] == "search"]
            q_found = sum(1 for q in qs if q in msgs or json.dumps(q)[1:-1] in msgs)
            with open(td / "search_calls.jsonl", "w") as f:
                for r in win:
                    ids = [str(x) for x in r.get("returned") or []]
                    base = {"ts": S.iso(r["t"]), "task": task, "n_returned": len(ids), "returned": [{"pmid": p} for p in ids],
                            "latency_ms": None, "mcp_call_id": f"mcp_cs.jsonl:{r['_line']}", "ts_source": "server_log",
                            "server_t": r["t"], "server_log_line": r["_line"]}
                    if r["tool"] == "search":
                        rec = {**base, "route": "s2_search",
                               "params": {"query": r["query"], "k": S.MCP_K, "native_route": "plain_search", "mcp_tool": "search"},
                               "suppressed": r.get("suppressed") or [], "n_pool_hits": r.get("n_pool_hits")}
                    elif r["tool"] == "fetch":
                        rec = {**base, "route": "paper_by_id",
                               "params": {"id": str(r["id"]), "native_route": "graph/v1/paper/PMID:{id}", "mcp_tool": "fetch"},
                               "suppressed": r.get("suppressed") or [], "status": r.get("status")}
                    else:
                        raise SystemExit(f"INGEST FAILED: {cond}/{task}: unknown tool {r['tool']!r} in log line {r['_line']}")
                    f.write(json.dumps(rec) + "\n")
            kids = fr.get("_child_frames") or []
            k_in = sum(int(c.get("input_tokens") or 0) for c in kids)
            k_out = sum(int(c.get("output_tokens") or 0) for c in kids)
            k_cost = sum(float(c.get("total_cost") or 0.0) for c in kids)
            r_in, r_out = int(fr.get("input_tokens") or 0), int(fr.get("output_tokens") or 0)
            a_in, a_out = int(fr.get("aux_input_tokens") or 0), int(fr.get("aux_output_tokens") or 0)
            cost = float(fr.get("total_cost") or 0.0) + float(fr.get("aux_cost") or 0.0) + k_cost
            has_report = (src / "report.md").exists()
            text = (src / "report.md").read_text(errors="replace") if has_report else ""
            prompt = Path(PROMPTS) / cond / f"{S.stem(tasks, task)}.txt"
            call = {"seq": 0, "t_req": t0, "t_done": t1, "wall_ms": int(round((t1 - t0) * 1000)), "status": 200,
                    "path": f"{PRODUCT} (workbench agent loop + built-in reviewer; one pool MCP connector)",
                    "model": fr.get("model"), "requested_model": "claude-sonnet-5", "finish_reason": fr.get("status"),
                    "usage": {"prompt_tokens": r_in + a_in + k_in, "completion_tokens": r_out + a_out + k_out,
                              "reasoning_tokens": None},
                    "cost_usd": round(cost, 8), "prompt": prompt.read_text(errors="replace") if prompt.exists() else "",
                    "completion": text,
                    "params": {"effort": fr.get("effort"), "root_input_tokens": r_in, "root_output_tokens": r_out,
                               "root_cache_read_tokens": fr.get("cache_read_tokens"),
                               "root_cache_write_tokens": fr.get("cache_write_tokens"), "root_total_cost": fr.get("total_cost"),
                               "aux_input_tokens": a_in, "aux_output_tokens": a_out, "aux_cost": fr.get("aux_cost"),
                               "reviewer_frames": len(kids), "reviewer_input_tokens": k_in, "reviewer_output_tokens": k_out,
                               "reviewer_cost": round(k_cost, 8), "reviewer_models": sorted({str(c.get("model")) for c in kids}),
                               "frame_id": fr.get("id"), "agent_name": fr.get("agent_name"),
                               "cost_note": f"{PRODUCT}'s own API-price estimate, root + aux + reviewers"}}
            (td / "_calls.jsonl").write_text(json.dumps(call) + "\n")
            S.pool_health(td)
            if has_report:
                shutil.copy2(src / "report.md", td / "report.md")
            page = (src / "page.txt").read_text(errors="replace") if (src / "page.txt").exists() else ""
            refusal = is_refusal(fr, has_report, page)
            cards = [json.loads(l) for l in open(src / "cards.jsonl") if l.strip()]
            P = {str(p) for r in win for p in r.get("returned") or []}
            rp = S.review_pmid(task)
            leak = {}
            if has_report:
                ptext, _, _ = prepare(text)
                refmap, _ = S.references(ptext)
                ref_p = {p for v in refmap.values() for p in v}
                cited, _, _, _ = inline_cited(ptext, refmap)
                sup_rp = sum(1 for r in win if rp in [str(x) for x in r.get("suppressed") or []])
                leak = {"references_outside_P": sorted(ref_p - P), "inline_cited_outside_P": sorted(cited - P),
                        "target_review_pmid": rp, "target_review_in_references": rp in ref_p,
                        "target_review_cited_inline": rp in cited, "target_review_ever_returned": rp in P,
                        "target_review_suppressed_or_refused_in_log": sup_rp}
            prov = {"source_dir": str(src), "frame_id": fr.get("id"), "frame_status": fr.get("status"), "model": fr.get("model"),
                    "effort": fr.get("effort"), "window": [t0, t1], "window_pad_s": WIN_PAD_S,
                    "report_sha256": S.sha256(src / "report.md") if has_report else None,
                    "frame_sha256": S.sha256(src / "frame.json"), "messages_sha256": S.sha256(src / "messages.jsonl"),
                    "report_note": (src / "report_note.txt").read_text() if (src / "report_note.txt").exists() else None,
                    "log_rows_task_cond": len(recs), "log_rows_in_window": len(win), "log_rows_outside_window": len(outside),
                    "outside_rows_offsets_s": [round(r["t"] - (t0 if r["t"] < t0 else t1), 1) for r in outside][:5],
                    "n_search": sum(1 for r in win if r["tool"] == "search"), "n_fetch": sum(1 for r in win if r["tool"] == "fetch"),
                    "fetch_404": sum(1 for r in win if r.get("status") == 404),
                    "search_queries_verbatim_in_messages": [q_found, len(qs)],
                    "P": len(P), "reviewer_frames": len(kids), "reviewer_models": sorted({str(c.get("model")) for c in kids}),
                    "cards": dict(Counter(c.get("decision") or c.get("event") for c in cards)),
                    "vendor_refusal": refusal, "leakage": leak, "wall_s": round(t1 - t0, 3), "cost_usd_total": round(cost, 6)}
            (td / "provenance.json").write_text(json.dumps(prov, indent=1))
            for r in outside:
                st_lines.append({"task": task, "status": "outside_run", "note": "pool log row of this task and condition outside "
                                 "the run's frame window; not in P", "t": r["t"], "log_line": r["_line"], "tool": r["tool"]})
            if refusal:
                st_lines.append({"task": task, "status": "failed", "note": f"{refusal}: recorded as a failure",
                                 "wall_s": round(t1 - t0, 3), "t": t1})
            else:
                st_lines.append({"task": task, "status": "ok", "wall_s": round(t1 - t0, 3), "t": t1})
            per[task] = {"P": len(P), "in_window": len(win), "outside_window": len(outside), "q_verbatim": [q_found, len(qs)],
                         "vendor_refusal": refusal, "model": fr.get("model"), "effort": fr.get("effort"),
                         "reviewer_models": prov["reviewer_models"], "has_report": has_report,
                         "cites_outside_P": bool(leak.get("references_outside_P") or leak.get("inline_cited_outside_P")),
                         "denied_cards": sum(1 for c in cards if c.get("decision") == "deny")}
        S.write_status(arm, st_lines)
        qv = [v["q_verbatim"] for v in per.values()]
        summary["conds"][cond] = {"key": key, "tasks": len(per),
                                  "log_rows_in_windows": sum(v["in_window"] for v in per.values()),
                                  "log_rows_outside_windows": sum(v["outside_window"] for v in per.values()),
                                  "outside_by_task": {t: v["outside_window"] for t, v in per.items() if v["outside_window"]},
                                  "search_queries_verbatim_in_messages": [sum(a for a, _ in qv), sum(b for _, b in qv)],
                                  "vendor_refusals": {t: v["vendor_refusal"] for t, v in per.items() if v["vendor_refusal"]},
                                  "cites_outside_P": [t for t, v in per.items() if v["cites_outside_P"]],
                                  "models": sorted({str(v["model"]) for v in per.values()}),
                                  "efforts": sorted({str(v["effort"]) for v in per.values()}),
                                  "reviewer_models": sorted({m for v in per.values() for m in v["reviewer_models"]}),
                                  "denied_cards": sum(v["denied_cards"] for v in per.values()),
                                  "per_task": per}
    S.write_tasks(out, tasks)
    (out / "convert_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({c: {k: v for k, v in s.items() if k != "per_task"} for c, s in summary["conds"].items()}, indent=1))
    bad = [c for c, s in summary["conds"].items() if s["models"] != ["claude-sonnet-5"] or s["efforts"] != ["low"]
           or s["denied_cards"] or any(m not in ("claude-sonnet-5",) for m in s["reviewer_models"])]
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
    text, pre, n_close = prepare(raw)
    refmap, rst = S.references(text)
    cst = cstats if cstats is not None else Counter()
    report = common.report_from_markdown(text, CITE_ALL, mark_resolver(refmap, cst))
    outline = common.outline_from_markdown(text)
    papers: dict[str, None] = {}
    for s in logs.load_search(task_dir):
        for p in s["returned"]:
            papers.setdefault(p, None)
    ro = common.assemble(key, task, task_dir, papers=list(papers), outline=outline, report=report, graph=None, final_ok=True,
                         bot_reason=None, extra_meta={"n_reference_entries": rst["n_entries"], "reference_stats": rst,
                                                      "preamble": pre, "citation_tokens": dict(cst),
                                                      "cite_close_tags_stripped": n_close,
                                                      "adapter": "evaluation/board/ingest/claude_science.py:adapt", "draft_is_final": True,
                                                      "backbone": "claude-sonnet-5",
                                                      "agent": f"{PRODUCT}, effort low, built-in reviewer on"})
    return ro


def main(argv=None) -> int:
    return S.main(argv, __doc__, convert, adapt, KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
