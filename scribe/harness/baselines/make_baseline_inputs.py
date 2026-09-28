#!/usr/bin/env python3
"""Writes the refs.json of every evaluation task (topic, cutoff, target length and the reference
papers with a known year before the cutoff, without the evaluated review) for the published-pipeline
drivers. Usage: python make_baseline_inputs.py --tasks <json> --taskspecs <dir> --gold <dir> --out <dir>."""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

RUNNERS = Path(__file__).resolve().parents[1] / "runners"


def abstracts_for(R, task_id):
    texts = R.texts_from_oracle(task_id)
    return {p: d.get("abstract", "") for p, d in texts.items()}


def load_years(path):
    return {str(k): int(v) for k, v in json.loads(Path(path).read_text()).items() if v is not None}


def review_pmid(gold_dir, task_id):
    rp = json.loads((Path(gold_dir) / f"{task_id}.json").read_text()).get("review_pmid")
    if not rp:
        raise SystemExit(f"no review_pmid for {task_id} in {gold_dir}")
    return str(rp)


def refs_record(task_id, spec, bundle, abstracts, years, review):
    cutoff = int(spec["publication_cutoff"])
    refs = [{"pmid": str(pp["paper_id"]), "doi": pp.get("doi"), "title": pp.get("title") or "",
             "year": years[str(pp["paper_id"])], "abstract": abstracts.get(pp["paper_id"]) or ""}
            for pp in bundle["papers"]
            if str(pp["paper_id"]) != review and years.get(str(pp["paper_id"]), cutoff) < cutoff]
    return {"topic": spec["question"], "n_refs": len(refs), "source": "gold",
            "task_id": task_id, "cutoff_year": cutoff,
            "target_words": spec["output_spec"]["target_words"], "refs": refs}


def build(R, task_id, specs_dir, years, gold_dir):
    spec = json.loads((specs_dir / f"{task_id}.json").read_text())
    b = R.load_oracle(task_id, "evidence_bundle")
    if not b:
        return None
    return refs_record(task_id, spec, b, abstracts_for(R, task_id), years, review_pmid(gold_dir, task_id))


def task_list(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(l)["task_id"] for l in p.open() if l.strip()]
    d = json.loads(p.read_text())
    return list(d["tasks"] if isinstance(d, dict) else d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True,
                    help="the evaluation task list (.json) or a TaskSpec index (.jsonl)")
    ap.add_argument("--taskspecs", required=True, help="directory of the tasks' TaskSpec json files")
    ap.add_argument("--gold", default=os.environ.get("POOL_GOLD_DIR"),
                    help="directory of the tasks' gold records with review_pmid (default: $POOL_GOLD_DIR)")
    ap.add_argument("--out", required=True, help="output directory: <out>/<task>/refs.json")
    a = ap.parse_args(argv)
    if not a.gold:
        raise SystemExit("--gold (or POOL_GOLD_DIR) is required")
    sys.path.insert(0, str(RUNNERS))
    import runner as R
    years = load_years(R.REF_YEARS)
    out, specs = Path(a.out), Path(a.taskspecs)
    tasks = task_list(a.tasks)
    n, skipped = 0, []
    for t in tasks:
        d = build(R, t, specs, years, a.gold)
        if d is None:
            skipped.append(t)
            continue
        od = out / t
        od.mkdir(parents=True, exist_ok=True)
        (od / "refs.json").write_text(json.dumps(d))
        n += 1
    print(json.dumps({"written": n, "n_tasks": len(tasks), "skipped": skipped, "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
