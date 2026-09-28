"""Converts each task's reference list into the row ids of the AutoSurvey FAISS database for fixed
input, keeping only rows in the task's cutoff view (a known year before the cutoff, the evaluated
review removed). Usage: python make_allowlists_idx.py --arm autosurvey --allowlists <dir>
--pool-index <dir> --cutoff-views <dir>."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DB = {"autosurvey": "autosurvey_db"}


def load_map(arm: str, pool_index: Path) -> dict[str, int]:
    with open(pool_index / DB[arm] / "arxivid_to_index_abs.json") as f:
        return json.load(f)


def cutoff_view(views: Path, task: str) -> set[int]:
    hits = sorted(views.glob(f"[0-9][0-9][0-9][0-9].{task}.npy"))
    if len(hits) != 1:
        raise SystemExit(f"expected one cutoff view <cutoff>.{task}.npy in {views}, found {len(hits)}")
    return {int(x) for x in np.load(hits[0])}


def build(arm: str, allow_dir: Path, out_dir: Path, pool_index: Path, views: Path) -> None:
    m = load_map(arm, pool_index)
    n = 0
    for src in sorted(allow_dir.glob("*.json")):
        visible = cutoff_view(views, src.stem)
        pmids = [str(x) for x in json.load(open(src))]
        idx = [m[p] for p in pmids if p in m and int(m[p]) in visible]
        np.save(out_dir / f"{src.stem}.{arm}.idx.npy", np.array(sorted(idx), dtype=np.int64))
        n += 1
    print(f"{arm}: wrote {n} index-space allowlists -> {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(DB))
    ap.add_argument("--allowlists", required=True,
                    help="directory of <task>.json reference lists (ccbench.fair.make_allowlists)")
    ap.add_argument("--pool-index", required=True,
                    help="directory holding autosurvey_db/ (biolitbench/pool/embed/assemble_db.py)")
    ap.add_argument("--cutoff-views", required=True,
                    help="the <cutoff>.<task>.npy views of biolitbench/pool/embed/cutoff_views.py (CUTOFF_DIR)")
    a = ap.parse_args(argv)
    d = Path(a.allowlists)
    build(a.arm, d, d, Path(a.pool_index), Path(a.cutoff_views))


if __name__ == "__main__":
    main()
