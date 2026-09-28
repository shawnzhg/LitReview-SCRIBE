#!/usr/bin/env python3
"""Renders each task's reference list in LiRA's bibliography format, keeping the references with an
abstract, an integer PMID and a year before the task's cutoff. Usage: python make_lira_input.py
--inputs <dir> --out <dir>."""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def lira_surveys(inputs):
    surveys, dropped = [], 0
    for i, d in enumerate(sorted(p for p in Path(inputs).iterdir() if p.is_dir())):
        data = json.loads((d / "refs.json").read_text())
        cutoff = int(data["cutoff_year"])
        refs = []
        for n, r in enumerate(data["refs"], 1):
            if not (r.get("abstract") or "").strip() or r.get("year") is None or int(r["year"]) >= cutoff:
                dropped += 1
                continue
            try:
                rid = int(r["pmid"])
            except (TypeError, ValueError):
                dropped += 1
                continue
            refs.append({"num": n, "id": rid, "title": r["title"] or "",
                         "content": r["abstract"]})
        if not refs:
            continue
        surveys.append({"id": i, "topic": data["topic"], "references": refs,
                        "task_id": data["task_id"]})
    return surveys, dropped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="the make_baseline_inputs.py output directory")
    ap.add_argument("--out", required=True, help="LiRA data dir: writes <out>/scireviewgen/full_data_abs.json")
    a = ap.parse_args(argv)
    surveys, dropped = lira_surveys(a.inputs)
    od = Path(a.out) / "scireviewgen"
    od.mkdir(parents=True, exist_ok=True)
    (od / "full_data_abs.json").write_text(json.dumps(surveys))
    print(json.dumps({"surveys": len(surveys), "refs_dropped": dropped,
                      "out": str(od / "full_data_abs.json")}))


if __name__ == "__main__":
    main()
