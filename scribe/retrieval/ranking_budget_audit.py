#!/usr/bin/env python3
"""Audits the budgets and the ranking of a same-pool acquisition against the K_cap file and the traces.
Usage: python ranking_budget_audit.py --runs <dir> --level <n> --tasks <ids> --kcap <json> --kcap-sha256
<sha> [--seeds <s>] --out <json>."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import selection_audit as EA
import ranker_audit as CA
import ranker as CR

EFF_KEYS = ("max_search_calls", "max_ranked_output_K", "max_document_opens", "max_results_per_call",
            "max_docs_read")
ROWS_RULE = "min(CAP.rpc, 1000)"
KCAP_PINNED_KEYS = tuple(k for k in EFF_KEYS if k != "max_results_per_call")


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()] if p.exists() else []


def expected_rows(rb):
    if rb.get("rows_rule") != ROWS_RULE:
        return None, [f"retrieval_budget rows_rule {rb.get('rows_rule')!r} != {ROWS_RULE!r}"]
    meas = ((rb.get("caps") or {}).get("results_per_call") or {}).get("measured")
    if meas is None:
        return None, ["no measured CAP.rpc in the unit's retrieval_budget.json"]
    return min(int(meas), 1000), []


def unit_ranking(d, seed, counter):
    try:
        qa_path = Path(d) / "query_agent_record.json"
        qa_probs = []
        if qa_path.exists():
            qa = json.loads(qa_path.read_text())
            if not qa.get("ok") or qa.get("problems") or qa.get("tool_search_calls") != qa.get("n_searches"):
                qa_probs.append(f"query-agent record not ok / tool {qa.get('tool_search_calls')} != harness "
                                f"{qa.get('n_searches')}: {qa.get('problems')}"[:300])
        if (Path(d) / "ranked_bundle.json").exists():
            with tempfile.TemporaryDirectory(prefix="ranking_shadow_", dir=os.environ.get("TMPDIR")) as tmp:
                r = CA.unit_audit(EA.shadow(d, tmp), seed, counter)
            r["unit"], r["selection_shadow"] = str(d), True
        else:
            r = CA.unit_audit(d, seed, counter)
        if qa_probs:
            r["problems"] = list(r.get("problems") or []) + qa_probs
            r["agree"] = False
    except Exception as e:
        r = {"unit": str(d), "task": Path(d).parent.name, "K": None, "trace": {}, "decision_recounted": None,
             "problems": [f"audit crashed: {type(e).__name__}: {e}"[:400]], "agree_problems": [],
             "record_present": True, "record_ok": False, "record_problems": [f"audit crashed: {type(e).__name__}"],
             "final_called_record": None, "final_called_trace": None, "fallback_record": None, "fallback_trace": True,
             "cause_record": None, "cause_trace": None, "tool_score_fallback": None, "n_survivors": None,
             "final_n_candidates": None, "record_attempts_missing_from_trace": 0, "final_mode": None, "n_chunks": None,
             "n_chunks_failed": None, "n_final_from_fallback": None, "share_from_fallback": None,
             "n_context_overflow_attempts": 0, "n_papers": None, "agree": False}
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--kcap", required=True)
    ap.add_argument("--kcap-sha256", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--system", default="SCRIBE")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    counter = CR.PromptCounter()
    raw = Path(a.kcap).read_bytes()
    findings = []
    sha = hashlib.sha256(raw).hexdigest()
    if sha != a.kcap_sha256:
        findings.append(f"K_cap file sha256 {sha} != pin {a.kcap_sha256}")
    kc = json.loads(raw)["tasks"]
    tasks = [t for t in a.tasks.split(",") if t]
    seeds = [int(s) for s in a.seeds.split(",") if s != ""]
    units, no_bundle, rows = {}, [], []
    for seed in seeds:
        for t in tasks:
            d = Path(a.runs) / f"level{a.level}" / a.system / "native_chain" / t / f"seed{seed}"
            key = t if len(seeds) == 1 else f"{t}/seed{seed}"
            if not (d / "evidence_bundle.json").exists():
                no_bundle.append(key)
                continue
            f = []
            ent = kc.get(t)
            if not ent:
                findings.append(f"{key}: task not in the K_cap file")
                continue
            rb = json.loads((d / "retrieval_budget.json").read_text()) if (d / "retrieval_budget.json").exists() else {}
            eff = rb.get("effective") or {}
            if rb.get("cap_rule") != "K_cap":
                f.append(f"retrieval_budget rule {rb.get('cap_rule')} != K_cap")
            if rb.get("kcap_sha256") != a.kcap_sha256:
                f.append(f"retrieval_budget kcap_sha256 {rb.get('kcap_sha256')} != pin")
            bad = {k: (eff.get(k), ent["effective"].get(k)) for k in KCAP_PINNED_KEYS
                   if eff.get(k) is None or int(eff.get(k)) != int(ent["effective"][k])}
            if bad:
                f.append(f"effective budget != the K_cap file: {bad}")
            rpc, rprobs = expected_rows(rb)
            f += rprobs
            if rpc is not None and (eff.get("max_results_per_call") is None
                                    or int(eff["max_results_per_call"]) != rpc):
                f.append(f"effective rows per call {eff.get('max_results_per_call')} != {rpc} ({ROWS_RULE})")
            b = json.loads((d / "evidence_bundle.json").read_text())
            nP = len(b.get("papers") or [])
            if nP > int(ent["K_cap"]):
                f.append(f"|P| {nP} > K_cap {ent['K_cap']}")
            calls = jl(d / "pool_calls.jsonl")
            if rpc is None:
                rpc = min(int(ent["K_cap"]), 1000)
            big = [len(c.get("kept") or []) for c in calls if c.get("kind") == "search" and len(c.get("kept") or []) > rpc]
            if big:
                f.append(f"{len(big)} searches kept more than the rows-per-call cap {rpc}")
            fr = unit_ranking(d, seed, counter)
            if not fr["record_present"]:
                f.append("ranking records missing: the ranking was not recorded")
            elif not fr["record_ok"]:
                f.append(f"rank record problems: {fr['record_problems']}")
            if fr["record_present"] and not fr["agree"]:
                f.append(f"rank record disagrees with the trace: record fallback {fr['fallback_record']} final "
                         f"{fr['final_called_record']} / trace fallback {fr['fallback_trace']} final "
                         f"{fr['final_called_trace']}; {fr['record_attempts_missing_from_trace']} attempts not in trace")
            rows.append(fr)
            units[key] = {"n_findings": len(f), "findings": f[:10], "K_cap": ent["K_cap"], "n_papers": nP,
                          "fallback_fired": fr["fallback_record"], "fallback_cause": fr["cause_record"],
                          "final_called": fr["final_called_record"], "tool_score_fallback": fr["tool_score_fallback"],
                          "n_survivors": fr["n_survivors"], "final_n_candidates": fr["final_n_candidates"]}
            findings += [f"{key}: {x}" for x in f]
    n = len(rows)
    nf = sum(1 for r in rows if r["fallback_record"])
    causes = {}
    for r in rows:
        if r["fallback_record"]:
            causes[r["cause_record"]] = causes.get(r["cause_record"], 0) + 1
    s = CA.summarize(rows, no_bundle=no_bundle)
    extra = [f"{r['task']}: ranking {'; '.join((r['problems'] + r['agree_problems'])[:4])}" for r in rows if not r["agree"]]
    out = {"schema": "ranking_budget_audit/1.0", "level": int(a.level), "seeds": seeds, "kcap": a.kcap, "kcap_sha256": sha,
           "n_units_audited": len(units), "no_bundle": no_bundle, "n_findings": len(findings),
           "findings": findings[:200], "units": units,
           "fallback": {"n_units": n, "n_fallback": nf, "rate": nf / n if n else None, "causes": causes,
                        "n_tool_score_fallback": sum(1 for r in rows if r["tool_score_fallback"])},
           "ranking": dict(s, rows=[{k: v for k, v in r.items() if k != "unit"} for r in rows]),
           "ranking_detail_findings": extra[:200]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"### ranking and budget audit: {len(units)} units, {len(no_bundle)} without a bundle, {len(findings)} findings; "
          f"final mode tournament {s['n_tournament']} single {s['n_single']} none {s['n_no_final']}; task fallback "
          f"{s['n_task_fallback']}/{s['n_units']}; ranking overflow attempts {s['n_rank_overflow_attempts']}; "
          f"disagreements {s['n_disagree']} -> {a.out}", flush=True)
    for x in (findings + extra)[:10]:
        print("###   finding:", x[:400])
    return 0 if (not findings and s["n_disagree"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
