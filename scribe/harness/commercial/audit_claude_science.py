#!/usr/bin/env python3
"""Audits the Claude Science runs against the pool MCP log: retrieved set per run, PMIDs cited
outside it or the reference list, citation of the evaluated review and approval-card decisions.
Usage: python audit_claude_science.py."""

import json, os, re, statistics as st
from collections import Counter
from pathlib import Path
H = Path(os.environ["COMMERCIAL_OUT"])
O = H / "runs"
GOLD = Path(os.environ["COMMERCIAL_GOLD"])
ALLOW = Path(os.environ["COMMERCIAL_ALLOWLISTS"])
L = [json.loads(l) for l in open(H / "mcp_call_logs/mcp_cs.jsonl")]
tasks = json.loads(Path(os.environ["COMMERCIAL_TASKS"]).read_text())
rows = []
for cond in ("fixinput", "samepool"):
    for k, t in enumerate(tasks):
        stem = f"{k+1:02d}_{t}"; d = O / cond / stem
        r = {"cond": cond, "task": t, "stem": stem}
        if not (d / "frame.json").exists():
            r["missing"] = True; rows.append(r); continue
        fr = json.loads((d / "frame.json").read_text())
        t0, t1 = fr["created_at"] / 1000 - 5, (fr["completed_at"] or fr["updated_at"]) / 1000 + 5
        R = [x for x in L if x.get("task") == t and x.get("cond") == cond and t0 <= x["t"] <= t1]
        P = {p for x in R for p in x.get("returned", [])}
        rp = str(json.loads((GOLD / f"{t}.json").read_text())["review_pmid"])
        allow = set(json.loads((ALLOW / f"{t}.json").read_text()))
        rep = (d / "report.md").read_text() if (d / "report.md").exists() else ""
        parts = re.split(r"\n#+\s*References?\s*\n", rep)
        body, refs = (parts[0], parts[-1]) if len(parts) > 1 else (rep, "")
        cited = set(re.findall(r"PMID:?\s*(\d{5,9})", refs))
        cards = [json.loads(x) for x in open(d / "cards.jsonl")]
        r.update(model=fr["model"], effort=fr["effort"], status=fr["status"], cost=fr["total_cost"],
                 in_tok=fr["input_tokens"], out_tok=fr["output_tokens"],
                 reviewers=sum(1 for c in fr.get("_child_frames", []) if c.get("agent_name") == "REVIEWER"),
                 n_search=sum(x["tool"] == "search" for x in R), n_fetch=sum(x["tool"] == "fetch" for x in R),
                 P=len(P), suppressed=sum(len(x.get("suppressed", [])) for x in R),
                 report_words=len(body.split()), has_refs=bool(refs), cited=len(cited),
                 cited_outside_P=sorted(cited - P), cited_outside_allow=sorted(cited - allow) if cond == "fixinput" else None,
                 review_cited=rp in cited,
                 fmt_cite_index=len(re.findall(r"cite index=", body)), fmt_numeric=len(re.findall(r"\[\d{1,3}(?:[,–\-\s]+\d{1,3})*\]", body)),
                 fmt_pmid=len(re.findall(r"\[\d{7,9}", body)),
                 cards_allowed=sum(1 for c in cards if c.get("decision") == "allow"), cards_denied=sum(1 for c in cards if c.get("decision") == "deny"),
                 denied_cards=[c.get("card") for c in cards if c.get("decision") == "deny"],
                 refusal=bool(re.search(r"can't help with this|safety filter|usage policy", rep, re.I)))
        rows.append(r)
(H / "audit.json").write_text(json.dumps(rows, indent=1))
for cond in ("fixinput", "samepool"):
    R = [r for r in rows if r["cond"] == cond and not r.get("missing")]
    print(f"== {cond}: {len(R)}/{len(tasks)} (missing {sum(1 for r in rows if r['cond']==cond and r.get('missing'))})")
    if not R:
        continue
    print(f"  models {Counter((r['model'], r['effort']) for r in R)} status {Counter(r['status'] for r in R)}")
    print(f"  report words median {st.median(r['report_words'] for r in R):.0f} min {min(r['report_words'] for r in R)}; no References: {[r['stem'][:2] for r in R if not r['has_refs']]}")
    print(f"  pool calls median search {st.median(r['n_search'] for r in R)} fetch {st.median(r['n_fetch'] for r in R)}; |P| median {st.median(r['P'] for r in R)}; zero-P runs {[r['stem'][:2] for r in R if r['P']==0]}")
    print(f"  cited median {st.median(r['cited'] for r in R)}; runs citing outside P: {[(r['stem'][:2], len(r['cited_outside_P'])) for r in R if r['cited_outside_P']]}")
    if cond == "fixinput":
        print(f"  runs citing outside allowlist: {[(r['stem'][:2], len(r['cited_outside_allow'])) for r in R if r['cited_outside_allow']]}")
    print(f"  target review cited: {[r['stem'][:2] for r in R if r['review_cited']]}; suppressed hits total {sum(r['suppressed'] for r in R)}")
    print(f"  citation format: cite-index runs {sum(r['fmt_cite_index']>0 for r in R)}, numeric-only {sum(r['fmt_numeric']>0 and r['fmt_cite_index']==0 for r in R)}, [PMID] {sum(r['fmt_pmid']>0 for r in R)}, none {[r['stem'][:2] for r in R if r['fmt_cite_index']==0 and r['fmt_numeric']==0 and r['fmt_pmid']==0]}")
    print(f"  cards allowed {sum(r['cards_allowed'] for r in R)} denied {sum(r['cards_denied'] for r in R)} {Counter(x for r in R for x in r['denied_cards'])}; refusal-like {[r['stem'][:2] for r in R if r['refusal']]}")
    print(f"  cost total ${sum(r['cost'] for r in R):.2f} mean ${st.mean(r['cost'] for r in R):.2f}; reviewer passes median {st.median(r['reviewers'] for r in R)}")
