#!/usr/bin/env python3
"""Audits the Elicit runs: the gathered papers against the task's reference list, the report's
papers against the gathered set, and the report and exports. Usage: python audit_elicit.py."""

import csv, json, os, re, statistics as st
from pathlib import Path
H = Path(os.environ["COMMERCIAL_OUT"]); D = H / "runs"
ALLOW = Path(os.environ["COMMERCIAL_ALLOWLISTS"])
tasks = json.loads(Path(os.environ["COMMERCIAL_TASKS"]).read_text())
MAX_CITED = 80


def idmap(d):
    m = {}
    if (d / "preflight_search.json").exists():
        j = json.loads((d / "preflight_search.json").read_text())
        for b in (j if isinstance(j, list) else [j]):
            if not isinstance(b, dict):
                continue
            for p in b.get("papers", []):
                pm = p.get("pmid") or (re.match(r"PUBMED-(\d+)$", p.get("elicitId") or "") or [None, None])[1]
                if p.get("elicitId") and pm:
                    m[p["elicitId"]] = str(pm)
    return m


rows = []
for t in tasks:
    d = D / t; r = {"task": t}
    if not (d / "final.json").exists():
        r["missing"] = True; rows.append(r); continue
    fin = json.loads((d / "final.json").read_text())
    allow = set(json.loads((ALLOW / f"{t}.json").read_text()))
    S = list(csv.DictReader(open(d / "exports/search.csv.final", encoding="utf-8-sig")))
    xf = d / "exports/extract.csv.postreport"
    X = list(csv.DictReader(open(xf if xf.exists() else d / "exports/extract.csv.final", encoding="utf-8-sig")))
    M = idmap(d); pmid_of = lambda pid: M.get(pid) or ((re.match(r"PUBMED-(\d+)$", pid or "") or [None, None])[1])
    got = {pmid_of(x["Paper ID"]) for x in S} - {None}
    inc = [x for x in X if (x.get("Included in report", "").strip().lower() in ("yes", "true", "1")) or
           ("Included in report" not in x and (x.get("Report citation") or "").strip())]
    inc_p = {pmid_of(x["Paper ID"]) for x in inc} - {None}
    body = (d / "report_body.md").read_text() if (d / "report_body.md").exists() else ""
    ex = sorted({p.name.split(".")[0] + "." + p.name.split(".")[1] for p in (d / "exports").glob("*.final")})
    req = json.loads((d / "request.json").read_text())
    r.update(status=fin.get("status"), completed_at=fin.get("_t"),
             allowlist_n=len(allow), gathered=len(S), unmapped_ids=sum(1 for x in S if not pmid_of(x["Paper ID"])), gathered_pmid=len(got), gathered_in_allow=len(got & allow),
             allow_not_gathered=len(allow - got), gathered_not_in_allow=len(got - allow),
             extract_rows=len(X), included_in_report=len(inc), included_not_gathered=len(inc_p - got),
             report_words=len(body.split()), cite_markers=len(re.findall(r"\{[0-9a-f]+_\d+\}", body)),
             exports=ex, extraction_generate=req.get("extraction", {}).get("generate"))
    rows.append(r)
(H / "audit.json").write_text(json.dumps(rows, indent=1))
R = [r for r in rows if not r.get("missing")]
print(f"runs {len(R)}/{len(tasks)} missing {[r['task'] for r in rows if r.get('missing')]}; status {sorted({r['status'] for r in R})}")
if R:
    print(f"entry: gathered == allowlist in {sum(r['allow_not_gathered']==0 and r['gathered_not_in_allow']==0 for r in R)} runs; "
          f"max allowlist-not-gathered {max(r['allow_not_gathered'] for r in R)}, any gathered outside allowlist: {[r['task'][-8:] for r in R if r['gathered_not_in_allow']]}")
    print(f"   not gathered (allowlist PMIDs Elicit could not return): median {st.median(r['allow_not_gathered'] for r in R)}, total {sum(r['allow_not_gathered'] for r in R)} / {sum(r['allowlist_n'] for r in R)}")
    print(f"report: words median {st.median(r['report_words'] for r in R):.0f} min {min(r['report_words'] for r in R)}; cite markers median {st.median(r['cite_markers'] for r in R)} zero {[r['task'][-8:] for r in R if r['cite_markers']==0]}")
    print(f"included-in-report median {st.median(r['included_in_report'] for r in R)} max {max(r['included_in_report'] for r in R)}; over {MAX_CITED}: {[r['task'][-8:] for r in R if r['included_in_report'] > MAX_CITED]}; included not in gathered set: {[r['task'][-8:] for r in R if r['included_not_gathered']]}")
    need = {"search.csv", "extract.csv", "report.txt", "report.bib", "report.ris"}
    print(f"missing core exports: {[(r['task'][-8:], sorted(need - set(r['exports']))) for r in R if need - set(r['exports'])]}; no pdf/docx: {[r['task'][-8:] for r in R if 'report.pdf' not in r['exports']]}")
