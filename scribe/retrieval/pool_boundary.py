#!/usr/bin/env python3
"""Picks one PMID of each publication year from the pool index for the per-task cutoff probe. Usage:
python pool_boundary.py --index <dir> --years <from-to> --out <json>."""

import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--years", default="2010-2027")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.years.split("-"))
    idx = Path(a.index)
    raw = (idx / "meta.json").read_bytes()
    meta = json.loads(raw)
    want = set(range(lo, hi + 1))
    got = {}
    t0 = time.time()
    for s in meta["shards"]:
        d = idx / "shards" / s["name"]
        yr = np.load(d / "year.npy", mmap_mode="r")
        pm = np.load(d / "pmid.npy", mmap_mode="r")
        y = np.asarray(yr)
        for Y in sorted(want - set(got)):
            hit = np.flatnonzero(y == Y)
            if hit.size:
                got[Y] = str(int(np.asarray(pm[hit]).min()))
        if set(got) >= want:
            break
    out = {"schema": "pool_boundary/1.0", "index": str(idx),
           "bm25_meta_sha256_16": hashlib.sha256(raw).hexdigest()[:16],
           "stats_fingerprint": meta.get("stats_fingerprint"), "years": [lo, hi],
           "by_year": {str(k): v for k, v in sorted(got.items())},
           "missing_years": sorted(want - set(got)), "seconds": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("bm25_meta_sha256_16", "by_year", "missing_years", "seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
