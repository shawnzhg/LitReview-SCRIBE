#!/usr/bin/env python3
"""Runs Elicit Systematic Review once on one task under fixed input, with PubMed searches for the
task's reference-list PMIDs as its gather step, and logs every request, poll, export and the report.
Usage: python run_elicit_sr.py <task>."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

OUT = Path(os.environ["COMMERCIAL_OUT"])
TASKSPECS = Path(os.environ["COMMERCIAL_TASKSPECS"])
ALLOWLISTS = Path(os.environ["COMMERCIAL_ALLOWLISTS"])
API = "https://elicit.com/api/v2"
COND = "fixinput"
USAGE_LEDGER = OUT / "elicit_usage.jsonl"


def hdr() -> dict:
    key = os.environ["ELICIT_API_KEY"].strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def get(url: str, **kw):
    for i in range(6):
        try:
            r = requests.get(url, headers=hdr(), timeout=120, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(20 * (i + 1)); continue
            return r
        except requests.RequestException:
            time.sleep(20 * (i + 1))
    raise RuntimeError(f"GET failed repeatedly: {url}")


def usage() -> dict:
    return get(f"{API}/usage").json()


def main(task: str) -> None:
    out = OUT / "runs" / task
    if (out / "final.json").exists():
        sys.exit(f"{task} already done")
    if (out / "create_response.json").exists() and not (out / "session_id.txt").exists():
        sys.exit(f"{task} already run")
    (out / "exports").mkdir(parents=True, exist_ok=True)
    spec = json.loads((TASKSPECS / f"{task}.json").read_text())
    allow = json.loads((ALLOWLISTS / f"{task}.json").read_text())
    chunks = [allow[i:i + 100] for i in range(0, len(allow), 100)]
    queries = [" OR ".join(f"{p}[pmid]" for p in c) for c in chunks]
    assert all(len(q) <= 2000 for q in queries)
    body = {
        "researchQuestion": spec["question"],
        "title": spec["question"],
        "searches": [{"query": q, "corpus": "pubmed", "searchMode": "keyword", "maxResults": 300} for q in queries],
        "extraction": {"generate": True, "useFigures": False},
        "generateReport": True,
        "isPublic": False,
    }
    (out / "request.json").write_text(json.dumps({**body, "_task": task, "_cond": COND,
                                                  "_taskspec_hash": spec["content_hash"], "_allowlist_n": len(allow)}, indent=1))

    if (out / "session_id.txt").exists():
        sid = (out / "session_id.txt").read_text().strip()
        self_url = f"{API}/sessions/systematic-reviews/{sid}"
        u0 = json.loads((out / "usage_before.json").read_text()) if (out / "usage_before.json").exists() else usage()
        print(f"[resume] {task} {sid}", flush=True)
        return poll_and_finish(task, out, sid, self_url, u0, time.time())

    pfs = []
    for q in queries:
        pf = requests.post(f"{API}/search/papers", headers=hdr(), timeout=120,
                           json={"query": q, "corpus": "pubmed", "searchMode": "keyword", "maxResults": 300})
        pfs.append(pf.json())
    (out / "preflight_search.json").write_text(json.dumps(pfs))

    u0 = usage(); (out / "usage_before.json").write_text(json.dumps({**u0, "_t": now()}, indent=1))
    if u0.get("extraUsage") is not None:
        sys.exit("ABORT: extra usage is enabled on the account; refusing to run")
    if not u0.get("hasUsageRemaining"):
        sys.exit("no usage remaining")
    t0 = time.time()
    r = requests.post(f"{API}/sessions/systematic-reviews", headers=hdr(), json=body, timeout=120)
    (out / "create_response.json").write_text(json.dumps({"_t": now(), "http": r.status_code, "body": r.json() if r.text else None}, indent=1))
    if r.status_code not in (200, 202):
        sys.exit(f"create failed {r.status_code}: {r.text[:800]}")
    sid = r.json()["sessionId"]
    self_url = r.json()["links"]["self"]
    (out / "session_id.txt").write_text(sid)
    print(f"[create] {sid} {r.json().get('url')}", flush=True)
    return poll_and_finish(task, out, sid, self_url, u0, t0)


def poll_and_finish(task, out, sid, self_url, u0, t0):
    timeline, seen = {}, set()
    while True:
        g = get(self_url)
        d = g.json()
        with open(out / "polls.jsonl", "a") as f:
            f.write(json.dumps({"_t": now(), "_elapsed_s": round(time.time() - t0, 1), "http": g.status_code, "body": d}) + "\n")
        st, stage = d.get("status"), d.get("executionStage")
        for k in (f"status:{st}", f"stage:{stage}", *[f"data:{s}" for s in (d.get("data") or {})]):
            timeline.setdefault(k, round(time.time() - t0, 1))
        (out / "stage_timeline.json").write_text(json.dumps(timeline, indent=1))
        fresh = d.get("dataFreshness") or "none"
        for sname, sval in (d.get("data") or {}).items():
            for fmt, url in sval.items():
                if not isinstance(url, str) or not url.startswith("http"):
                    continue
                tag = (sname, fmt, fresh)
                if tag in seen:
                    continue
                dl = requests.get(url, timeout=300)
                if dl.status_code == 200:
                    (out / "exports" / f"{sname}.{fmt}.{fresh.replace(':', '')}").write_bytes(dl.content)
                    seen.add(tag)
        print(f"[poll] {time.time() - t0:.0f}s status={st} stage={stage} data={sorted((d.get('data') or {}).keys())} exports={d.get('exportsStatus')}", flush=True)
        if st in ("completed", "failed"):
            if st == "completed" and d.get("exportsStatus") == "generating":
                time.sleep(30); continue
            break
        if st == "pausedForInsufficientQuota":
            (out / "paused.json").write_text(json.dumps({"_t": now(), "sessionId": sid, "links": d.get("links"),
                                                         "executionStage": stage}, indent=1))
            u1 = usage(); (out / "usage_at_pause.json").write_text(json.dumps({**u1, "_t": now()}, indent=1))
            print("[paused] insufficient quota; the exports so far are saved and a later call resumes polling", flush=True)
            return "paused"
        time.sleep(30)

    fin = get(f"{self_url}?include=reportBody").json()
    (out / "final.json").write_text(json.dumps({"_t": now(), **fin}, indent=1))
    rep = ((fin.get("data") or {}).get("report") or {}).get("result") or {}
    if rep.get("reportBody"):
        (out / "report_body.md").write_text(rep["reportBody"])
    if rep.get("abstract"):
        (out / "report_abstract.md").write_text(rep["abstract"])
    for sname, sval in (fin.get("data") or {}).items():
        for fmt, url in sval.items():
            if isinstance(url, str) and url.startswith("http"):
                dl = requests.get(url, timeout=300)
                if dl.status_code == 200:
                    (out / "exports" / f"{sname}.{fmt}.final").write_bytes(dl.content)
    lst = get(f"{API}/sessions").json()
    (out / "session_list_entry.json").write_text(json.dumps(
        [s for s in lst.get("sessions", []) if s.get("sessionId") == sid], indent=1))
    time.sleep(60)
    u1 = usage(); (out / "usage_after.json").write_text(json.dumps({**u1, "_t": now()}, indent=1))
    rec = {"t": now(), "task": task, "cond": COND, "sessionId": sid, "status": fin.get("status"),
           "percent_before": u0.get("percentUsed"), "percent_after": u1.get("percentUsed"),
           "percent_this_run": (u1.get("percentUsed") or 0) - (u0.get("percentUsed") or 0),
           "wall_s": round(time.time() - t0, 1), "settings": fin.get("settings")}
    with open(USAGE_LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[done] {task} status={fin.get('status')} usage {u0.get('percentUsed')}% -> {u1.get('percentUsed')}%  wall {time.time() - t0:.0f}s", flush=True)
    return fin.get("status")


if __name__ == "__main__":
    main(sys.argv[1])
