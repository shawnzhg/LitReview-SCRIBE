"""Converts each task's reference list into the row ids of the AutoSurvey FAISS database for fixed
input, without the evaluated review named in the cutoff views' task_to_cutoff.json. Usage: python
make_allowlists_idx.py --arm autosurvey --allowlists <dir> --pool-index <dir> --cutoff-views <dir>."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DB = {"autosurvey": "autosurvey_db"}


def load_map(arm: str, pool_index: Path) -> dict[str, int]:
    with open(pool_index / DB[arm] / "arxivid_to_index_abs.json") as f:
        return json.load(f)


def review_pmid(views: Path, task: str) -> str:
    p = views / "task_to_cutoff.json"
    ent = (json.loads(p.read_text())["tasks"] if p.is_file() else {}).get(task) or {}
    if not ent.get("review_pmid"):
        raise SystemExit(f"no review_pmid for {task} in {p}")
    return str(ent["review_pmid"])


def build(arm: str, allow_dir: Path, out_dir: Path, pool_index: Path, views: Path) -> None:
    m = load_map(arm, pool_index)
    n = 0
    for src in sorted(allow_dir.glob("*.json")):
        review = review_pmid(views, src.stem)
        pmids = [str(x) for x in json.load(open(src))]
        idx = [m[p] for p in pmids if p in m and p != review]
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
                    help="biolitbench/pool/embed/cutoff_views.py output (CUTOFF_DIR); names each task's review")
    a = ap.parse_args(argv)
    d = Path(a.allowlists)
    build(a.arm, d, d, Path(a.pool_index), Path(a.cutoff_views))


if __name__ == "__main__":
    main()
