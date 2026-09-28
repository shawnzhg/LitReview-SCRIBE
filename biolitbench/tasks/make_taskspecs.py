#!/usr/bin/env python3
"""Writes the TaskSpec and gold file of every held-out task: the review title as the question, the
review's publication year as the cutoff, before which all evidence must be published, and the review
itself excluded. Usage: python make_taskspecs.py --pool-index <bm25 dir> [--force]."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scribe" / "harness" / "runners"))
from common import RUNS, seal, validate

INP = RUNS / "_taskinputs"
BENCH = Path(os.environ.get("SCRIBE_BENCH_RESULTS") or "/nonexistent/SCRIBE_BENCH_RESULTS")
OUT = RUNS / "taskspecs" / "dev"
GOLD = RUNS / "gold" / "dev"
OPENS_FLOOR = 60
TOOLS = ["pool_search", "pool_fetch"]
AUDIENCE = "biomedical researchers familiar with the general field but not with this specific literature"


def snapshot_id(index_dir: Path) -> str:
    return "pool-bm25-" + hashlib.sha256((index_dir / "meta.json").read_bytes()).hexdigest()[:16]


def build(rec: dict, snapshot: str) -> dict:
    review_pmid = rec.get("review_pmid")
    spec = {
        "schema_version": "taskspec/1.0",
        "task_id": rec["task_id"],
        "split": "dev",
        "question": rec["title"],
        "review_type": "narrative",
        "scope": {
            "population": "",
            "intervention_or_topic": rec.get("topic") or rec.get("topic_from_refs") or "",
            "comparators": [],
            "outcomes": [],
            "inclusion": [],
            "exclusion": ([f"the review under evaluation itself (PMID {review_pmid})"]
                          if review_pmid else []),
        },
        "audience": AUDIENCE,
        "corpus_snapshot_id": snapshot,
        "publication_cutoff": rec["pub_year_min"],
        "allowed_tools": list(TOOLS),
        "budget": {"max_document_opens": OPENS_FLOOR},
        "output_spec": {"target_words": int(rec["body_words"]), "citation_style": "numeric"},
    }
    return seal(spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-index", required=True, help="the BM25 index directory of the frozen pool")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    snapshot = snapshot_id(Path(a.pool_index))
    rows = [json.loads(l) for l in (INP / "dev_task_inputs.jsonl").open()]

    OUT.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    existing = list(OUT.glob("*.json"))
    if existing and not a.force:
        sys.exit(f"REFUSING to overwrite {len(existing)} frozen specs in {OUT}. "
                 f"A TaskSpec change invalidates every artifact generated under it. Use --force "
                 f"only if nothing has been generated yet.")

    written, bad, index = 0, [], []
    for rec in rows:
        if not rec.get("review_pmid"):
            bad.append({"task_id": rec["task_id"], "errors": ["no review_pmid"]})
            continue
        spec = build(rec, snapshot)
        errs = validate("task_spec", spec)
        if errs:
            bad.append({"task_id": rec["task_id"], "errors": errs})
            continue
        (OUT / f"{rec['task_id']}.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False))

        R = json.loads((BENCH / "benchmark_claims" / rec["task_id"] / "references.json").read_text())["references"]
        gold = {
            "task_id": rec["task_id"],
            "task_spec_hash": spec["content_hash"],
            "review_pmid": str(rec["review_pmid"]),
            "gold_pmids": sorted({str(v["pmid"]) for v in R.values() if v.get("pmid")}),
            "n_references_total": len(R),
            "cutoff_year": rec["pub_year_min"],
            "target_words": int(rec["body_words"]),
            "human_section_titles": rec["section_titles"],
            "n_claims": rec.get("n_claims"),
        }
        (GOLD / f"{rec['task_id']}.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False))
        index.append({"task_id": rec["task_id"], "task_spec_hash": spec["content_hash"],
                      "cutoff": spec["publication_cutoff"],
                      "target_words": spec["output_spec"]["target_words"],
                      "n_gold_pmids": len(gold["gold_pmids"]), "review_pmid": gold["review_pmid"]})
        written += 1

    (OUT.parent / "dev_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in index))
    print(json.dumps({"written": written, "invalid": len(bad), "errors": bad[:10], "snapshot": snapshot,
                      "distinct_spec_hashes": len({r["task_spec_hash"] for r in index})}, indent=1))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
