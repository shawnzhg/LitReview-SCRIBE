#!/usr/bin/env python3
"""Downloads the extraction table of every completed Elicit run, which links its citation markers to
papers. Usage: python elicit_fetch_postreport.py."""

import csv, io, json, requests
import run_elicit_sr as E
D = E.OUT / "runs"
for d in sorted(D.glob("pmcid_*")):
    if not (d / "final.json").exists():
        continue
    sid = (d / "session_id.txt").read_text().strip() if (d / "session_id.txt").exists() else json.loads((d / "final.json").read_text())["sessionId"]
    fin = E.get(f"{E.API}/sessions/systematic-reviews/{sid}").json()
    ok = []
    for fmt, url in ((fin.get("data") or {}).get("extract") or {}).items():
        if isinstance(url, str) and url.startswith("http"):
            r = requests.get(url, timeout=300)
            if r.status_code == 200:
                (d / "exports" / f"extract.{fmt}.postreport").write_bytes(r.content); ok.append(fmt)
    txt = (d / "exports" / "extract.csv.postreport").read_text(encoding="utf-8-sig") if "csv" in ok else ""
    has = bool(txt) and "Report citation" in next(csv.reader(io.StringIO(txt)))
    print(d.name, ok, "report-cols" if has else "NO report cols", fin.get("dataFreshness"), flush=True)
