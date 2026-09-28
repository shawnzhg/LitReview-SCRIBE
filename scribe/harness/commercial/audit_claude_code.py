#!/usr/bin/env python3
"""Audits the Claude Code runs: acceptance, citation format, PMIDs cited outside the run's retrieved
set or the reference list, and citation of the evaluated review. Usage: python
audit_claude_code.py."""

import json, os, re, statistics as st
from pathlib import Path
BUNDLE = Path(os.environ["COMMERCIAL_OUT"])
OUT = BUNDLE / "runs"
GOLD = Path(os.environ["COMMERCIAL_GOLD"])
ALLOW = Path(os.environ["COMMERCIAL_ALLOWLISTS"])
rows = []
for cond in ("fixinput", "samepool"):
    for d in sorted((OUT / cond).iterdir()):
        task = d.name
        if not (d / "result.json").exists():
            rows.append(dict(cond=cond, task=task, ok=False, missing=True))
            continue
        r = json.loads((d / "result.json").read_text())
        rp = str(json.loads((GOLD / f"{task}.json").read_text())["review_pmid"])
        allow = set(json.loads((ALLOW / f"{task}.json").read_text()))
        P, n_search = set(), 0
        for l in (d / "stream.jsonl").read_text().splitlines():
            m = json.loads(l)
            content = (m.get("message") or {}).get("content")
            for c in content if isinstance(content, list) else []:
                if not isinstance(c, dict):
                    continue
                if m.get("type") == "assistant" and c.get("type") == "tool_use" and c.get("name") == "mcp__pool__search":
                    n_search += 1
                if m.get("type") == "user" and c.get("type") == "tool_result":
                    cc = c.get("content")
                    txt = cc if isinstance(cc, str) else json.dumps(cc)
                    P |= set(re.findall(r'\\?"id\\?"\s*:\s*\\?"(\d+)', txt))
        rep = (d / "report.md").read_text()
        parts = re.split(r"\n#+\s*References\s*\n", rep)
        body, refs = (parts[0], parts[-1]) if len(parts) > 1 else (rep, "")
        cited = set(re.findall(r"PMID:?\s*(\d{5,9})", refs))
        pmid_brackets = len(re.findall(r"\[\d{7,9}(?:[,;\s]+\d{7,9})*\]", body))
        num_brackets = len(re.findall(r"\[\d{1,3}(?:[,–\-\s]+\d{1,3})*\]", body))
        rows.append(dict(cond=cond, task=task,
                         ok=r["subtype"] == "success" and r.get("model_ok") is True and r["n_tool_errors"] == 0 and r["words"] > 0,
                         model=r.get("model"), words=len(body.split()), n_search=n_search, calls=r["n_tool_calls"],
                         fetch="mcp__pool__fetch" in r["tools_called"], P=len(P), cited=len(cited),
                         outside_P=len(cited - P), outside_allow=len(cited - allow) if cond == "fixinput" else None,
                         review_cited=rp in cited, fmt="pmid" if pmid_brackets > num_brackets else "numeric",
                         no_refs=not refs, cost=r["total_cost_usd"], wall=r["wall_s"]))
json.dump(rows, open(BUNDLE / "audit.json", "w"), indent=1)
for cond in ("fixinput", "samepool"):
    R = [x for x in rows if x["cond"] == cond and not x.get("missing")]
    print(f"== {cond}: n={len(R)} accepted={sum(x['ok'] for x in R)} missing={sum(1 for x in rows if x['cond'] == cond and x.get('missing'))}")
    if not R:
        continue
    print(f"  models {sorted({str(x['model']) for x in R})}")
    print(f"  words median {st.median(x['words'] for x in R):.0f}; searches median {st.median(x['n_search'] for x in R)}; tool calls median {st.median(x['calls'] for x in R)}; used fetch {sum(x['fetch'] for x in R)}/{len(R)}")
    print(f"  |P| median {st.median(x['P'] for x in R)}; cited median {st.median(x['cited'] for x in R)}; no References section {sum(x['no_refs'] for x in R)}")
    print(f"  runs citing PMIDs outside own P: {sum(x['outside_P']>0 for x in R)} (total {sum(x['outside_P'] for x in R)} PMIDs)")
    if cond == "fixinput": print(f"  runs citing outside allowlist: {sum((x['outside_allow'] or 0)>0 for x in R)}")
    print(f"  target review cited: {sum(x['review_cited'] for x in R)}; [PMID]-style citations: {sum(x['fmt']=='pmid' for x in R)}")
    print(f"  cost-equivalent ${sum(x['cost'] or 0 for x in R):.2f}; wall median {st.median(x['wall'] for x in R):.0f}s")
