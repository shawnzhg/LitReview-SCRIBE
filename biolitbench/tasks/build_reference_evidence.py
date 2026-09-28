#!/usr/bin/env python3
"""Writes each task's reference-derived evidence bundle (the review's cited papers with an abstract,
ranked by the review's own reference weight) and its abstract sentences from GLKB, the files the
harness, the published-pipeline inputs and the canonical bundles read; needs network access and the
glkb_client module. Usage: python build_reference_evidence.py [--tasks <json>] [--out <dir>]."""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scribe" / "harness" / "runners"))
from common import RUNS, seal, sha256_str

DATASET = Path(os.environ.get("SCRIBE_DATASET_ROOT") or "/nonexistent/SCRIBE_DATASET_ROOT")
EVAL_TASKS = Path(os.environ.get("SCRIBE_EVAL_TASKS") or "/nonexistent/SCRIBE_EVAL_TASKS")


def fetch_sentences(pmids, batch=400):
    import glkb_client as G
    out = {}
    todo = sorted(pmids)
    for i in range(0, len(todo), batch):
        rows = G.query("UNWIND $p AS pid MATCH (s:Sentence) WHERE s.id STARTS WITH 'pmid'+pid+'_' "
                       "RETURN s.id AS id, s.text AS t", {"p": todo[i:i + batch]})
        for r in rows:
            out.setdefault(r["id"].split("_")[0][4:], []).append((r["id"], r["t"]))
    for v in out.values():
        v.sort(key=lambda x: int(x[0].rsplit("_", 1)[-1]))
    return out


def ref_num(rid):
    return int(rid) if rid.isdigit() else 1 << 30


def bundle(task, refs, weights, sents):
    by_pmid = {}
    for rid, v in refs.items():
        if not (v.get("pmid") and v.get("abstract")):
            continue
        e = by_pmid.setdefault(v["pmid"], {"v": v, "w": 0, "rids": []})
        e["w"] += int(weights.get(rid, 0) or 0)
        e["rids"].append(rid)
    usable = sorted(((min(e["rids"], key=ref_num), e["v"], e["w"]) for e in by_pmid.values()),
                    key=lambda t: (-t[2], ref_num(t[0])))
    papers, evidence, task_sents = [], [], {}
    for rank, (rid, v, w) in enumerate(usable, 1):
        pmid = v["pmid"]
        papers.append({"paper_id": pmid, "doi": v.get("doi"), "title": v.get("title"), "year": None, "rank": rank,
                       "score": float(weights.get(rid, 0) or 0), "first_seen_step": None,
                       "retrieval_provenance": {"query": "", "tool": "reference_derived", "rank_from_tool": None,
                                                "route": "cited_by_human"},
                       "decision": "include",
                       "decision_reason": (f"cited by the review; summed authored weight {w} over "
                                           f"reference id(s) {sorted(by_pmid[pmid]['rids'])}"),
                       "post_cutoff": False})
        evidence.append({"evidence_id": f"e_{pmid}", "paper_id": pmid, "granularity": "abstract",
                         "locator": f"pmid{pmid}", "text_hash": sha256_str(v["abstract"]), "condition": None})
        for loc, text in sents.get(pmid, []):
            evidence.append({"evidence_id": f"e_{pmid}_{loc.rsplit('_', 1)[-1]}", "paper_id": pmid,
                             "granularity": "abstract_sentence", "locator": loc, "text_hash": sha256_str(text),
                             "condition": None})
            task_sents[loc] = text
    b = {"schema_version": "evidence_bundle/1.0", "task_id": task, "task_spec_hash": None,
         "provenance_tier": "reference_derived", "retrieval_status": "done", "papers": papers, "evidence": evidence,
         "discovered_union": [], "failures": [], "budget_remaining": {}, "validation": {
             "ranking": "reference_weight_desc_then_refid — authored importance, NOT a retrieval ranking",
             "n_refs_total": len(refs), "n_distinct_papers": len(usable),
             "n_duplicate_reference_ids": sum(len(e["rids"]) - 1 for e in by_pmid.values()),
             "coverage": round(len(usable) / max(len(refs), 1), 4),
             "sentences_source": "GLKB Sentence nodes; text in sentences/<task>.json for offline use"}}
    return seal(b), task_sents


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(EVAL_TASKS), help="task list JSON (default $SCRIBE_EVAL_TASKS)")
    ap.add_argument("--out", default=str(RUNS / "oracle" / "dev"), help="output root (default $SCRIBE_RUNS_ROOT/oracle/dev)")
    a = ap.parse_args(argv)
    tasks = json.loads(Path(a.tasks).read_text())["tasks"]
    out = Path(a.out)
    for sub in ("evidence_bundle", "sentences"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    refs = {t: json.loads((DATASET / "refs_enriched" / f"{t}.json").read_text()) for t in tasks}
    sents = fetch_sentences({v["pmid"] for r in refs.values() for v in r.values() if v.get("pmid") and v.get("abstract")})
    for t in tasks:
        weights = json.loads((DATASET / "graphs" / t / "graph_enhanced.json").read_text()).get("reference_weights", {})
        b, task_sents = bundle(t, refs[t], weights, sents)
        (out / "evidence_bundle" / f"{t}.json").write_text(json.dumps(b))
        (out / "sentences" / f"{t}.json").write_text(json.dumps(task_sents))
    print(f"wrote {len(tasks)} evidence bundles and sentence files to {out}")


if __name__ == "__main__":
    main()
