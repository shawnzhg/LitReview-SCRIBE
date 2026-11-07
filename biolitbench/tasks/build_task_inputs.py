#!/usr/bin/env python3
"""Writes the per-task inputs that make_taskspecs.py reads: the review's publication year and PMID
from its JATS XML, its title, body word count and section titles from the benchmark claims, and its
topic. Usage: python build_task_inputs.py --manifest <biolit_dev.jsonl> --xml <dir of <task>.xml> [--out <jsonl>]."""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scribe" / "harness" / "runners"))
from common import RUNS

BENCH = Path(os.environ.get("SCRIBE_BENCH_RESULTS") or "/nonexistent/SCRIBE_BENCH_RESULTS")
DATASET = Path(os.environ.get("SCRIBE_DATASET_ROOT") or "/nonexistent/SCRIBE_DATASET_ROOT")
PUBDATE = re.compile(r'<pub-date([^>]*)>(.*?)</pub-date>', re.S)
DATETYPE = re.compile(r'(?:pub-type|date-type)="([^"]+)"')
YEAR = re.compile(r"<year>\s*(\d{4})\s*</year>")
PMID_ID = re.compile(r'<article-id pub-id-type="pmid">\s*(\d+)')
HEAD = 200_000


def pub_years(xml_text: str) -> dict:
    out = {}
    for attrs, body in PUBDATE.findall(xml_text[:HEAD]):
        m = YEAR.search(body)
        if m:
            t = DATETYPE.search(attrs)
            out.setdefault(t.group(1) if t else "unlabelled", int(m.group(1)))
    return out


def one(task_id: str, xml_dir: Path, topic: dict) -> dict:
    rec = {"task_id": task_id, "pub_year_min": None, "review_pmid": None}
    xml = xml_dir / f"{task_id}.xml"
    if xml.exists():
        text = xml.read_text(errors="replace")
        years = pub_years(text)
        rec["pub_year_min"] = min(years.values()) if years else None
        m = PMID_ID.search(text[:HEAD])
        rec["review_pmid"] = m.group(1) if m else None
    bc = BENCH / "benchmark_claims" / task_id
    rs = json.loads((bc / "review_structure_llm.json").read_text())
    rec["title"] = (rs.get("paper") or {}).get("title")
    rec["body_words"] = sum(len((p.get("text") or "").split()) for p in rs.get("paragraphs") or [])
    rec["section_titles"] = [s.get("title") for s in rs.get("sections") or []]
    rec["topic_from_refs"] = json.loads((bc / "references.json").read_text()).get("topic")
    rec["topic"] = topic.get(task_id)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="biolit_dev.jsonl written by make_splits.py")
    ap.add_argument("--xml", required=True, help="directory of the reviews' JATS XML, <task>.xml")
    ap.add_argument("--out", default=str(RUNS / "_taskinputs" / "dev_task_inputs.jsonl"))
    a = ap.parse_args(argv)
    topic = {p["paper_id"]: p.get("topic") for p in json.loads((DATASET / "manifests" / "per_paper.json").read_text())}
    tasks = [json.loads(l)["task_id"] for l in open(a.manifest) if l.strip()]
    rows = [one(t, Path(a.xml), topic) for t in tasks]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"written": len(rows), "no_pub_year": [r["task_id"] for r in rows if not r["pub_year_min"]],
                      "no_review_pmid": [r["task_id"] for r in rows if not r["review_pmid"]]}))


if __name__ == "__main__":
    main()
