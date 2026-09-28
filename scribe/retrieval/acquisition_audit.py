#!/usr/bin/env python3
"""Audits a same-pool acquisition before generation (provenance, cutoff, review exclusion, logged
searches, budget, snapshot). Usage: python acquisition_audit.py --runs <dir> --level <n> --tasks
<ids> --cutoffs <json> --service-log <file> [--seeds <s>] --out <json>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

POOL_SNAPSHOT_ID = "pool-bm25-263f4ee078ad2e00"


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--level", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--cutoffs", required=True)
    ap.add_argument("--service-log", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--system", default="SCRIBE")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = Path(a.runs)
    cuts = json.loads(Path(a.cutoffs).read_text())
    svc = jl(a.service_log)
    findings, units, no_bundle = [], {}, []
    seeds = [x for x in a.seeds.split(",") if x != ""]
    claimed: dict = {}
    for t, seed in [(t, sd) for sd in seeds for t in a.tasks.split(",") if t]:
        d = runs / f"level{a.level}" / a.system / "native_chain" / t / f"seed{seed}"
        ukey = t if len(seeds) == 1 else f"{t}/seed{seed}"
        if not (d / "evidence_bundle.json").exists():
            no_bundle.append(ukey)
            continue
        f = []
        cut = int(cuts[t])
        b = json.loads((d / "evidence_bundle.json").read_text())
        prov = json.loads((d / "pool_provenance.json").read_text()) if (d / "pool_provenance.json").exists() else {}
        if not prov.get("ok"):
            f.append("pool_provenance missing or not ok")
        if prov.get("cutoff") != cut:
            f.append(f"provenance cutoff {prov.get('cutoff')} != {cut}")
        if prov.get("bm25_meta_sha256_16") != "263f4ee078ad2e00":
            f.append("index sha16 not the pin")
        pr = prov.get("cutoff_probe") or {}
        if (pr.get("before_cutoff") or {}).get("status") != 200:
            f.append(f"cutoff probe before-cutoff not 200: {pr.get('before_cutoff')}")
        af = pr.get("at_cutoff") or {}
        if af.get("pmid") is not None and af.get("status") != 404:
            f.append(f"cutoff probe at-cutoff not 404: {af}")
        calls = jl(d / "pool_calls.jsonl")
        sc = [c for c in calls if c.get("kind") == "search"]
        returned = set()
        for c in sc:
            for x in c.get("returned") or []:
                returned.add(str(x["pmid"]))
                if x.get("year") is None or int(x["year"]) >= cut:
                    f.append(f"search returned pmid {x['pmid']} ({x.get('year')}) not dated before the cutoff")
            if c.get("cutoff") != cut:
                f.append(f"pool_calls cutoff {c.get('cutoff')} != {cut}")
        kept = set()
        for c in sc:
            kept.update(str(x["pmid"]) for x in c.get("kept") or [])
        pids = [str(p["paper_id"]) for p in b.get("papers") or []]
        for p in b.get("papers") or []:
            if p.get("year") is None or int(p["year"]) >= cut:
                f.append(f"bundle paper {p['paper_id']} year {p.get('year')} not before the cutoff")
        not_logged = [p for p in pids if p not in kept]
        if not_logged:
            f.append(f"{len(not_logged)} bundle papers were never returned by a logged search: {not_logged[:5]}")
        gold = runs / "gold" / "dev" / f"{t}.json"
        rp = str(json.loads(gold.read_text()).get("review_pmid")) if gold.exists() else None
        if rp and (rp in pids or rp in set(map(str, b.get("discovered_union") or []))):
            f.append(f"the review under evaluation {rp} reached the bundle / discovered_union")
        tr = [r for r in jl(d / "trace.jsonl") if r.get("kind") == "retrieval_search"]
        if len(tr) != len(sc):
            f.append(f"trace search events {len(tr)} != pool_calls searches {len(sc)}")
        for e, c in zip(tr, sc):
            rr = e.get("retrieval") or {}
            if rr.get("query") != c.get("query") or rr.get("snapshot_id") != POOL_SNAPSHOT_ID or rr.get("cutoff_year") != cut:
                f.append("trace search event disagrees with pool_calls (query / snapshot / cutoff)")
                break
            logged = [x["paper_id"] for x in rr.get("returned") or []]
            if any(x not in {k["pmid"] for k in c.get("kept") or []} for x in logged):
                f.append("trace returned ids are not the backend's kept ids")
                break
        st_all = [r for r in svc if r.get("task") == t and r.get("route") == "plain_search"]
        if len(seeds) == 1:
            if len(st_all) != len(sc):
                f.append(f"service log has {len(st_all)} searches for {t}, pool_calls {len(sc)}")
            else:
                for c, s in zip(sc, st_all):
                    if (s.get("params") or {}).get("q") != c.get("query") or s.get("cutoff") != cut or \
                            [x["pmid"] for x in s.get("returned") or []] != [x["pmid"] for x in c.get("returned") or []]:
                        f.append("service log row disagrees with pool_calls (query / cutoff / returned pmids)")
                        break
                    if [x.get("score") for x in s.get("returned") or []] != [x.get("score") for x in c.get("returned") or []]:
                        f.append("service log row disagrees with pool_calls (returned scores)")
                        break
        else:
            used = claimed.setdefault(t, set())
            for c in sc:
                sig = (c.get("query"), cut, [(x["pmid"], x.get("score")) for x in c.get("returned") or []])
                hit = next((i for i, s in enumerate(st_all) if i not in used and
                            ((s.get("params") or {}).get("q"), s.get("cutoff"),
                             [(x["pmid"], x.get("score")) for x in s.get("returned") or []]) == sig), None)
                if hit is None:
                    f.append("a pool_calls search has no identical service-log row (query / cutoff / pmids / scores)")
                    break
                used.add(hit)
        rb = json.loads((d / "retrieval_budget.json").read_text()) if (d / "retrieval_budget.json").exists() else None
        eff = (rb or {}).get("effective") or {}
        if rb is None:
            f.append("retrieval_budget.json missing (the pool budget was not applied)")
        else:
            excl = set(map(str, rb.get("exclude_ids") or []))
            dl = set()
            for c in sc:
                dl.update(str(x["pmid"]) for x in c.get("kept") or [])
            for c in calls:
                if c.get("kind") == "open" and c.get("found"):
                    dl.add(str(c.get("pmid")))
            dl -= excl
            cap_calls, cap_rpc = eff.get("max_search_calls"), eff.get("max_results_per_call")
            cap_docs, cap_K, cap_open = eff.get("max_docs_read"), eff.get("max_ranked_output_K"), eff.get("max_document_opens")
            if cap_calls is not None and len(sc) > int(cap_calls):
                f.append(f"{len(sc)} searches > budget {cap_calls}")
            big = [len(c.get("kept") or []) for c in sc if cap_rpc is not None and len(c.get("kept") or []) > int(cap_rpc)]
            if big:
                f.append(f"{len(big)} searches kept more rows than the per-call cap {cap_rpc} (max {max(big)})")
            if cap_docs is not None and len(dl) > int(cap_docs):
                f.append(f"{len(dl)} distinct docs delivered > docs_read cap {cap_docs}")
            if cap_K is not None and len(pids) > int(cap_K):
                f.append(f"|P| {len(pids)} > K {cap_K}")
            n_open_found = sum(1 for c in calls if c.get("kind") == "open")
            if cap_open is not None and n_open_found > int(cap_open):
                f.append(f"{n_open_found} opens > budget {cap_open}")
            if any(c.get("docs_cap") != eff.get("max_docs_read") or c.get("rpc_cap") != eff.get("max_results_per_call")
                   for c in sc):
                f.append("a search ran with caps other than the unit's effective budget")
        mans = [m for m in jl(d / "manifests.jsonl") if m.get("window") == "acquisition"]
        if not mans or any(m.get("corpus_snapshot_id") != POOL_SNAPSHOT_ID for m in mans):
            f.append("acquisition manifest missing or not stamped with the pool snapshot")
        opens = [c for c in calls if c.get("kind") == "open"]
        units[ukey] = {"n_findings": len(f), "findings": f[:10], "n_papers": len(pids), "n_search": len(sc),
                    "effective_budget": eff,
                    "docs_read": len(dl) if rb is not None else None,
                    "rows_per_call_max": max((len(c.get("kept") or []) for c in sc), default=0),
                    "fetch_depth": sorted({c.get("service_limit") for c in sc}), "k": sorted({c.get("k_requested") for c in sc}),
                    "n_open": len(opens), "n_returned_distinct": len(returned), "cutoff": cut,
                    "review_returned_by_service": bool(rp and rp in returned)}
        findings += [f"{ukey}: {x}" for x in f]
    out = {"schema": "acquisition_audit/1.1", "level": int(a.level), "seeds": seeds,
           "n_units_audited": len(units), "no_bundle": no_bundle,
           "n_findings": len(findings), "findings": findings[:200], "units": units,
           "budget_used": {"search_calls_median": sorted(u["n_search"] for u in units.values())[len(units) // 2] if units else None,
                           "papers_median": sorted(u["n_papers"] for u in units.values())[len(units) // 2] if units else None,
                           "opens_median": sorted(u["n_open"] for u in units.values())[len(units) // 2] if units else None},
           "service_returned_target_review": sorted(t for t, u in units.items() if u["review_returned_by_service"])}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"### acquisition audit: {len(units)} units, {len(no_bundle)} without a bundle, {len(findings)} findings; "
          f"budget used {out['budget_used']}; target review returned by the service (and dropped by our tool) on "
          f"{len(out['service_returned_target_review'])} units -> {a.out}")
    for x in findings[:10]:
        print("###   finding:", x)
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
