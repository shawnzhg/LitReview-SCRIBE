"""Writes the per-task fixed-input reference lists: the review's resolved references dated before its
publication cutoff, without the evaluated review itself. Usage: python -m ccbench.fair.make_allowlists
--out <dir>."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ccbench import paths
from ccbench.gt import graphs
from ccbench.ingest import gold


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in gold.campaign50_tasks():
        g = graphs.load(t)
        cutoff = int(gold.taskspec(t)["publication_cutoff"])
        review = str(json.load(open(paths.gold_path(t)))["review_pmid"])
        allow = sorted((p for p in g.reachable_pmids(cutoff) if p != review), key=int)
        json.dump(allow, open(out / f"{t}.json", "w"))
        np.save(out / f"{t}.npy", np.array([int(p) for p in allow], dtype=np.int64))
        rows.append({"task": t, "cutoff": cutoff, "n_gold": len(g.pmids), "n_allowed": len(allow)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "_summary.csv", index=False)
    print(f"wrote {len(df)} allowlists to {out}")


if __name__ == "__main__":
    main()
