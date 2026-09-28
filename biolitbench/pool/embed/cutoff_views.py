#!/usr/bin/env python3
"""Writes, for each task, the FAISS row ids of the pool papers published before its cutoff, without
the evaluated review, as <cutoff>.<task>.npy. Usage: python cutoff_views.py --task-cutoffs <json>
--gold <dir> --db <faiss dir> --out <dir>."""

from __future__ import annotations

import argparse
import json
import os


def load_cutoffs(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for task, v in raw.items():
        year = int(v)
        if not (1900 <= year <= 2100):
            raise SystemExit("task %r: implausible cutoff year %r" % (task, year))
        out[task] = year
    return out


def review_pmids(gold_dir: str, tasks):
    out = {}
    for t in tasks:
        p = os.path.join(gold_dir, "%s.json" % t)
        if not os.path.isfile(p):
            raise SystemExit("no gold record for %s at %s" % (t, p))
        with open(p, "r", encoding="utf-8") as f:
            rp = json.load(f).get("review_pmid")
        if not rp:
            raise SystemExit("%s: gold record has no review_pmid" % t)
        out[t] = str(rp)
    return out


def view_name(year: int, task: str) -> str:
    return "%d.%s.npy" % (year, task)


def build(task_cutoffs: str, gold: str, db: str, out: str) -> int:
    import numpy as np

    cutoffs = load_cutoffs(task_cutoffs)
    reviews = review_pmids(gold, sorted(cutoffs))
    years = np.load(os.path.join(db, "id_year.npy"))
    pmids = np.load(os.path.join(db, "id_pmid.npy"))
    if years.ndim != 1 or years.shape != pmids.shape:
        raise SystemExit("id_year.npy %s and id_pmid.npy %s must be 1-D and aligned" % (years.shape, pmids.shape))
    os.makedirs(out, exist_ok=True)
    for task, year in sorted(cutoffs.items()):
        mask = (years > 0) & (years < year) & (pmids != int(reviews[task]))
        ids = (np.nonzero(mask)[0] + 1).astype("int64")
        if ids.size == 0:
            raise SystemExit("%s: cutoff %d selects no pool paper" % (task, year))
        np.save(os.path.join(out, view_name(year, task)), ids)
    with open(os.path.join(out, "task_to_cutoff.json"), "w", encoding="utf-8") as f:
        json.dump({"tasks": {t: {"cutoff": y, "review_pmid": reviews[t]} for t, y in sorted(cutoffs.items())},
                   "id_space": "external (IndexIDMap ids, 1..N)", "n_pool": int(years.size)}, f, indent=2)
    print("[cutoff] %d task views -> %s" % (len(cutoffs), out), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-cutoffs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return build(a.task_cutoffs, a.gold, a.db, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
