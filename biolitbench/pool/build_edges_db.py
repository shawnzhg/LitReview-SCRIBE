#!/usr/bin/env python3
"""Builds the memory-mapped citation-edge store (forward and backward adjacency and citation counts)
from the exported edge and metadata shards. Usage: python build_edges_db.py --edges <glob> --meta
<glob> --out <dir>."""

from __future__ import annotations

import argparse
import array
import glob as globmod
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_common import sha256_file

try:
    import orjson as _fastjson

    def loads(b):
        return _fastjson.loads(b)
except ImportError:
    def loads(b):
        return json.loads(b)

FLUSH_VALUES = 1 << 23
CHUNK = 1 << 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def resolve(patterns, kind):
    out = []
    for p in patterns:
        pth = Path(p)
        if pth.is_dir():
            out.extend(sorted(str(x) for x in pth.glob(f"{kind}_*.jsonl.gz")))
        elif any(c in p for c in "*?["):
            out.extend(sorted(globmod.glob(p)))
        else:
            out.append(p)
    sealed = [s for s in out if s.endswith(".jsonl.gz") and Path(s).is_file()]
    dropped = [s for s in out if s not in sealed]
    if dropped:
        log(f"  skipping {len(dropped)} unsealed/missing: {[Path(d).name for d in dropped]}")
    seen, uniq = set(), []
    for s in sealed:
        r = str(Path(s).resolve())
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def stage_pairs(shards, tmp: Path, resume: bool):
    marker = tmp / "pairs.json"
    path = tmp / "pairs.i64"
    if resume and marker.exists() and path.exists():
        st = json.loads(marker.read_text())
        if st.get("n_bytes") == path.stat().st_size:
            log(f"  resume: {st['n_edges']} packed edges already staged")
            return st
    t0 = time.time()
    buf = array.array("q")
    n_edges = n_rows = 0
    max_pmid = 0
    with open(path, "wb") as out:
        for si, sh in enumerate(shards):
            with gzip.open(sh, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = loads(line)
                    src = int(r["pmid"])
                    refs = r.get("refs") or []
                    if src > max_pmid:
                        max_pmid = src
                    n_rows += 1
                    hi = src << 32
                    for d in refs:
                        d = int(d)
                        if d > max_pmid:
                            max_pmid = d
                        buf.append(hi | d)
                    n_edges += len(refs)
                    if len(buf) >= FLUSH_VALUES:
                        buf.tofile(out)
                        del buf[:]
            log(f"  packed {si + 1}/{len(shards)} {Path(sh).name}: "
                f"{n_edges} edges, max_pmid={max_pmid}, RSS {peak_rss_gb():.2f} GB")
        if buf:
            buf.tofile(out)
            del buf[:]
    if max_pmid >= (1 << 32):
        sys.exit(f"pmid {max_pmid} does not fit in uint32; the packing scheme assumes it does")
    st = {"n_edges": n_edges, "n_rows": n_rows, "max_pmid": max_pmid,
          "n_bytes": path.stat().st_size, "seconds": round(time.time() - t0, 1)}
    marker.write_text(json.dumps(st))
    log(f"  packed {n_edges} edges from {n_rows} rows in {st['seconds'] / 60:.1f} min")
    return st


def build_direction(src_path: Path, out: Path, prefix: str, max_pmid: int,
                    swap_to: Path = None):
    t0 = time.time()
    a = np.fromfile(src_path, dtype=np.int64)
    log(f"  {prefix}: loaded {a.size} pairs ({a.nbytes / 1e9:.2f} GB), sorting")
    a.sort()

    keep = np.empty(a.size, dtype=bool)
    keep[0] = True
    np.not_equal(a[1:], a[:-1], out=keep[1:])
    n_dup = int(a.size - np.count_nonzero(keep))
    if n_dup:
        a = a[keep]
    del keep

    self_loops = 0
    sm = np.zeros(a.size, dtype=bool)
    for i in range(0, a.size, CHUNK):
        c = a[i:i + CHUNK]
        sm[i:i + CHUNK] = (c >> 32) == (c & 0xFFFFFFFF)
    self_loops = int(np.count_nonzero(sm))
    if self_loops:
        a = a[~sm]
    del sm
    n_final = int(a.size)
    log(f"  {prefix}: {n_final} unique edges (dropped {n_dup} duplicate, {self_loops} self)")

    if swap_to is not None:
        with open(swap_to, "wb") as f:
            for i in range(0, n_final, CHUNK):
                c = a[i:i + CHUNK]
                (((c & 0xFFFFFFFF) << 32) | (c >> 32)).tofile(f)
        log(f"  {prefix}: wrote swapped pairs for the reverse direction")

    val = np.lib.format.open_memmap(out / f"{prefix}_val.npy", mode="w+",
                                    dtype=np.uint32, shape=(n_final,))
    for i in range(0, n_final, CHUNK):
        val[i:i + CHUNK] = (a[i:i + CHUNK] & 0xFFFFFFFF).astype(np.uint32)
    val.flush()
    del val

    keys = np.arange(max_pmid + 2, dtype=np.int64) << 32
    indptr = np.searchsorted(a, keys, side="left").astype(np.int64)
    indptr[-1] = n_final
    del keys, a
    n_keys = int(np.count_nonzero(np.diff(indptr)))
    np.save(out / f"{prefix}_indptr.npy", indptr)
    del indptr
    log(f"  {prefix}: done in {time.time() - t0:.0f}s, {n_keys} non-empty keys, "
        f"peak RSS {peak_rss_gb():.2f} GB")
    return {"n_edges": n_final, "n_keys": n_keys, "n_duplicate_dropped": n_dup,
            "n_self_loops_dropped": self_loops, "seconds": round(time.time() - t0, 1)}


def stage_ncit(shards, out: Path):
    t0 = time.time()
    pm, val = array.array("l"), array.array("l")
    for si, sh in enumerate(shards):
        with gzip.open(sh, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                r = loads(line)
                pm.append(int(r["pmid"]))
                c = r.get("n_citation")
                val.append(-1 if c is None else int(c))
        if (si + 1) % 10 == 0 or si + 1 == len(shards):
            log(f"  ncit {si + 1}/{len(shards)}: {len(pm)} rows, RSS {peak_rss_gb():.2f} GB")
    pm_a = np.frombuffer(pm, dtype=np.int64)
    val_a = np.frombuffer(val, dtype=np.int64)
    n = int(pm_a.size)
    mx = int(pm_a.max()) if n else 0
    arr = np.full(mx + 1, -1, dtype=np.int32)
    arr[pm_a] = val_a.astype(np.int32)
    del pm_a, val_a, pm, val
    np.save(out / "ncit.npy", arr)
    known = int(np.count_nonzero(arr >= 0))
    nz = int(np.count_nonzero(arr > 0))
    mean = float(arr[arr >= 0].mean()) if known else 0.0
    del arr
    log(f"  ncit: {n} rows, {known} with a value, {nz} non-zero, mean {mean:.1f}, max pmid {mx}, "
        f"{time.time() - t0:.0f}s")
    return {"n_rows": n, "n_known": known, "n_nonzero": nz, "max_pmid": mx,
            "mean_when_known": round(mean, 2), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges", nargs="+", required=True)
    ap.add_argument("--meta", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None, help="scratch for the packed pairs (default <out>/_tmp)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.tmp) if args.tmp else out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    edge_shards = resolve(args.edges, "edges")
    meta_shards = resolve(args.meta, "meta") if args.meta else []
    if not edge_shards:
        sys.exit("no sealed edge shards matched --edges")
    log(f"edges: {len(edge_shards)} shard(s); meta: {len(meta_shards)} shard(s) -> {out}")
    t_start = time.time()

    log("stage 1/4: packing edges")
    pk = stage_pairs(edge_shards, tmp, args.resume)
    max_pmid = pk["max_pmid"]
    log("stage 2/4: forward CSR (refs of X)")
    fwd = build_direction(tmp / "pairs.i64", out, "fwd", max_pmid, swap_to=tmp / "swapped.i64")
    log("stage 3/4: reverse CSR (cited_by of X)")
    bwd = build_direction(tmp / "swapped.i64", out, "bwd", max_pmid)

    ncit = None
    if meta_shards:
        log("stage 4/4: n_citation")
        ncit = stage_ncit(meta_shards, out)
    else:
        log("stage 4/4: skipped (no --meta)")

    meta = {
        "schema_version": "pool_edges_db/1.0",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "built_by": "biolitbench/pool/build_edges_db.py",
        "structure": "dense CSR indexed by pmid, both directions, mmap .npy",
        "max_pmid": max_pmid,
        "n_edges_input": pk["n_edges"],
        "n_source_rows": pk["n_rows"],
        "forward": fwd, "reverse": bwd, "n_citation": ncit,
        "edge_shards": [{"name": Path(p).name, "bytes": os.stat(p).st_size, "sha256": sha256_file(p)}
                        for p in edge_shards],
        "meta_shards": [{"name": Path(p).name, "bytes": os.stat(p).st_size, "sha256": sha256_file(p)}
                        for p in meta_shards],
        "bytes": sum(f.stat().st_size for f in out.glob("*.npy")),
        "build_seconds": round(time.time() - t_start, 1),
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE: {meta['bytes'] / 1e9:.2f} GB, {meta['build_seconds'] / 60:.1f} min, "
        f"peak RSS {meta['peak_rss_gb']:.2f} GB")
    log(f"temp files in {tmp} hold "
        f"{sum(f.stat().st_size for f in tmp.iterdir()) / 1e9:.2f} GB -- delete to reclaim")


if __name__ == "__main__":
    main()
