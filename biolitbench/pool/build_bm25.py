#!/usr/bin/env python3
"""Builds the sharded BM25 index over the titles and abstracts of the retrieval pool. Usage: python
build_bm25.py --shards <glob> --out <dir> [--resume]."""

from __future__ import annotations

import argparse
import array
import glob as globmod
import gzip
import heapq
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_common import (
    B_DEFAULT, K1_DEFAULT, STEMMER, TOKENIZER_VERSION, UNDATED, doc_encode, idf_lucene, iter_shard,
    make_stemmer, sha256_file, term_hash, term_hash_many, tokenize,
)


def peak_rss_gb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def drop_cache(path):
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass

FIELD_TITLE_WEIGHT = 1
MIN_DF_DEFAULT = 1
DOCS_PER_BLOCK_DEFAULT = 100_000


def build_text(rec: dict) -> str:
    return ((rec.get("title") or "") + " \n " + (rec.get("abstract") or ""))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_shards(patterns) -> list:
    out = []
    for p in patterns:
        pth = Path(p)
        if pth.is_dir():
            out.extend(sorted(str(x) for x in pth.glob("*.jsonl.gz")))
        elif any(c in p for c in "*?["):
            out.extend(sorted(globmod.glob(p)))
        else:
            out.append(p)
    sealed = [s for s in out if s.endswith(".jsonl.gz") and Path(s).is_file()]
    dropped = [s for s in out if s not in sealed]
    if dropped:
        log(f"skipping {len(dropped)} unsealed/missing path(s): {[Path(d).name for d in dropped]}")
    seen, uniq = set(), []
    for s in sealed:
        r = str(Path(s).resolve())
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    if not uniq:
        sys.exit("no sealed corpus shards matched --shards")
    return uniq


def pass1_shard(path: str, tmp: Path, resume: bool) -> dict:
    name = Path(path).name.replace(".jsonl.gz", "")
    stat_p = tmp / f"{name}.stat.json"
    df_p = tmp / f"{name}.df.tsv.gz"
    if resume and stat_p.exists() and df_p.exists():
        return json.loads(stat_p.read_text())
    stemmer = make_stemmer()
    df = {}
    n_docs = 0
    sum_dl = 0
    t0 = time.time()
    for rec in iter_shard(path):
        toks = tokenize(build_text(rec), stemmer)
        n_docs += 1
        sum_dl += len(toks)
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    with gzip.open(df_p.with_suffix(".tmp"), "wt", encoding="utf-8") as f:
        for t in sorted(df):
            f.write(f"{t}\t{df[t]}\n")
    os.replace(df_p.with_suffix(".tmp"), df_p)
    drop_cache(path)
    drop_cache(df_p)
    st = {"name": name, "path": path, "n_docs": n_docs, "sum_dl": sum_dl,
          "n_terms_local": len(df), "pass1_seconds": round(time.time() - t0, 1)}
    stat_p.write_text(json.dumps(st))
    log(f"  pass1 {name}: {n_docs} docs, {len(df)} terms, {st['pass1_seconds']}s, "
        f"peak RSS {peak_rss_gb():.2f} GB")
    return st


def merge_vocab(tmp: Path, names, out: Path, n_docs: int, min_df: int, max_df_frac: float):
    max_df = int(max_df_frac * n_docs) if max_df_frac and max_df_frac < 1.0 else n_docs + 1
    files = [gzip.open(tmp / f"{n}.df.tsv.gz", "rt", encoding="utf-8") for n in names]

    def gen(fh):
        for line in fh:
            t, d = line.rstrip("\n").split("\t")
            yield t, int(d)

    terms_p = out / "vocab_terms.txt.gz"
    dfs = array.array("q")
    n_seen = n_kept = n_cut_min = n_cut_max = 0
    with gzip.open(str(terms_p) + ".tmp", "wt", encoding="utf-8") as tf:
        cur, acc = None, 0
        for term, d in heapq.merge(*[gen(f) for f in files], key=lambda x: x[0]):
            if term != cur:
                if cur is not None:
                    n_seen += 1
                    if acc < min_df:
                        n_cut_min += 1
                    elif acc > max_df:
                        n_cut_max += 1
                    else:
                        tf.write(cur + "\n")
                        dfs.append(acc)
                        n_kept += 1
                cur, acc = term, d
            else:
                acc += d
        if cur is not None:
            n_seen += 1
            if acc < min_df:
                n_cut_min += 1
            elif acc > max_df:
                n_cut_max += 1
            else:
                tf.write(cur + "\n")
                dfs.append(acc)
                n_kept += 1
    for f in files:
        f.close()
    os.replace(str(terms_p) + ".tmp", terms_p)
    df_arr = np.frombuffer(dfs, dtype=np.int64).copy()
    log(f"  vocab: {n_seen} distinct terms, kept {n_kept} "
        f"(dropped {n_cut_min} with df<{min_df}, {n_cut_max} with df>{max_df})")

    hashes = []
    chunk = []
    with gzip.open(terms_p, "rt", encoding="utf-8") as f:
        for line in f:
            chunk.append(line.rstrip("\n"))
            if len(chunk) >= (1 << 20):
                hashes.append(term_hash_many(chunk))
                chunk = []
    if chunk:
        hashes.append(term_hash_many(chunk))
    h = np.concatenate(hashes) if hashes else np.zeros(0, np.uint64)
    del hashes
    order = np.argsort(h, kind="stable")
    h_sorted = h[order]
    n_collide = int(np.count_nonzero(np.diff(h_sorted) == 0))
    if n_collide:
        log(f"  WARNING: {n_collide} 64-bit term-hash collision(s)")
    np.save(out / "vocab_hash.npy", h_sorted)
    np.save(out / "vocab_hash_order.npy", order.astype(np.int64))
    np.save(out / "df.npy", df_arr[order])
    np.save(out / "idf.npy", idf_lucene(df_arr[order], n_docs))
    return {"n_terms_seen": n_seen, "n_terms_kept": n_kept, "n_terms_dropped_min_df": n_cut_min,
            "n_terms_dropped_max_df": n_cut_max, "max_df_absolute": max_df,
            "n_hash_collisions": n_collide}


def stats_fingerprint(**kw) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:16]


def _write_npy(d: Path, name: str, arr: np.ndarray):
    np.save(d / name, arr)
    drop_cache(d / name)


def pass2_block(d: Path, docs, vocab_hash, idf, tok2gid, stemmer, avgdl, k1, b,
                fingerprint: str) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    key = array.array("q")
    dl = array.array("i")
    pmids, years = array.array("q"), array.array("h")
    doc_off = array.array("q", [0])
    V = len(vocab_hash)
    n = 0
    with open(d / "docs.bin.tmp", "wb") as db:
        for rec in docs:
            toks = tokenize(build_text(rec), stemmer)
            dl.append(len(toks))
            base = n
            for t in toks:
                g = tok2gid.get(t, -2)
                if g == -2:
                    h = np.uint64(term_hash(t))
                    pos = int(np.searchsorted(vocab_hash, h))
                    g = pos if pos < V and np.uint64(vocab_hash[pos]) == h else -1
                    tok2gid[t] = g
                if g >= 0:
                    key.append((g << 32) | base)
            pmids.append(int(rec["pmid"]))
            y = rec.get("year")
            years.append(int(y) if isinstance(y, int) else UNDATED)
            blob = doc_encode({"pmid": str(rec["pmid"]), "title": rec.get("title") or "",
                               "abstract": rec.get("abstract") or "",
                               "year": rec.get("year"), "doi": rec.get("doi") or ""})
            db.write(blob)
            doc_off.append(doc_off[-1] + len(blob))
            n += 1
        db.flush()
        os.fsync(db.fileno())
    os.replace(d / "docs.bin.tmp", d / "docs.bin")
    drop_cache(d / "docs.bin")
    _write_npy(d, "docs_off.npy", np.frombuffer(doc_off, dtype=np.int64).copy())
    _write_npy(d, "pmid.npy", np.frombuffer(pmids, dtype=np.int64).copy())
    year_arr = np.frombuffer(years, dtype=np.int16).copy()
    _write_npy(d, "year.npy", year_arr)
    dl_arr = np.frombuffer(dl, dtype=np.int32).astype(np.float32)
    del pmids, years, doc_off, dl

    k = np.frombuffer(key, dtype=np.int64).copy()
    del key
    k.sort()
    if k.size:
        newrun = np.empty(k.size, dtype=bool)
        newrun[0] = True
        np.not_equal(k[1:], k[:-1], out=newrun[1:])
        starts = np.flatnonzero(newrun)
        del newrun
        tf = np.diff(np.append(starts, k.size)).astype(np.float32)
        uk = k[starts]
        del starts, k
        gt = (uk >> 32).astype(np.int32)
        dc = (uk & 0xFFFFFFFF).astype(np.int32)
        del uk
        denom = tf + k1 * (1.0 - b + b * (dl_arr[dc] / avgdl))
        score = (np.asarray(idf[gt]) * (tf * (k1 + 1.0)) / denom).astype(np.float32)
        del tf, denom
        terms_present, first = np.unique(gt, return_index=True)
        indptr = np.append(first, gt.size).astype(np.int64)
        del gt, first
    else:
        terms_present = np.zeros(0, np.int32)
        indptr = np.zeros(1, np.int64)
        dc = np.zeros(0, np.int32)
        score = np.zeros(0, np.float32)
    _write_npy(d, "terms.npy", terms_present.astype(np.int32))
    _write_npy(d, "indptr.npy", indptr)
    _write_npy(d, "docid.npy", dc)
    _write_npy(d, "score.npy", score)
    return {"name": d.name, "n_docs": n, "nnz": int(score.size),
            "n_terms": int(terms_present.size), "stats_fingerprint": fingerprint,
            "docs_bin_bytes": (d / "docs.bin").stat().st_size,
            "year_min": int(year_arr[year_arr > UNDATED].min()) if n else None,
            "year_max": int(year_arr.max()) if n else None}


def pass2_shard(path: str, out: Path, vocab_hash: np.ndarray, idf: np.ndarray,
                avgdl: float, k1: float, b: float,
                resume: bool, fingerprint: str, block_docs: int) -> list:
    stem = Path(path).name.replace(".jsonl.gz", "")
    marker = out / "shards" / f"_{stem}.blocks.json"
    if resume and marker.exists():
        prev = json.loads(marker.read_text())
        if prev.get("stats_fingerprint") == fingerprint:
            return prev["blocks"]
        log(f"  pass2 {stem}: global statistics changed, rebuilding")
    stemmer = make_stemmer()
    tok2gid = {}
    t0 = time.time()
    blocks, buf, bi = [], [], 0
    for rec in iter_shard(path):
        buf.append(rec)
        if len(buf) >= block_docs:
            blocks.append(pass2_block(out / "shards" / f"{stem}.b{bi:03d}", buf, vocab_hash,
                                      idf, tok2gid, stemmer, avgdl, k1, b, fingerprint))
            buf, bi = [], bi + 1
    if buf:
        blocks.append(pass2_block(out / "shards" / f"{stem}.b{bi:03d}", buf, vocab_hash, idf,
                                  tok2gid, stemmer, avgdl, k1, b, fingerprint))
    drop_cache(path)
    el = round(time.time() - t0, 1)
    marker.write_text(json.dumps({"stats_fingerprint": fingerprint, "blocks": blocks,
                                  "pass2_seconds": el}))
    log(f"  pass2 {stem}: {sum(x['n_docs'] for x in blocks)} docs in {len(blocks)} block(s), "
        f"{sum(x['nnz'] for x in blocks)} nnz, {el}s, peak RSS {peak_rss_gb():.2f} GB")
    return blocks


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", nargs="+", required=True,
                    help="glob, directory, or explicit list of sealed corpus_*.jsonl.gz")
    ap.add_argument("--out", required=True, help="index directory, e.g. data/pool_index/bm25")
    ap.add_argument("--b", type=float, default=B_DEFAULT)
    ap.add_argument("--max-df-frac", type=float, default=0.30)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "shards").mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    shards = resolve_shards(args.shards)
    log(f"indexing {len(shards)} sealed shard(s) -> {out}")
    t_start = time.time()

    log("pass 1: global document frequency")
    t0 = time.time()
    stats = [pass1_shard(p, tmp, args.resume) for p in shards]
    n_docs = sum(s["n_docs"] for s in stats)
    sum_dl = sum(s["sum_dl"] for s in stats)
    if n_docs == 0:
        sys.exit("corpus is empty")
    avgdl = sum_dl / n_docs
    pass1_s = time.time() - t0
    log(f"pass 1 done: {n_docs} docs, avgdl={avgdl:.1f}, {pass1_s / 60:.1f} min")

    log("merging vocabularies")
    t0 = time.time()
    vstat = merge_vocab(tmp, [s["name"] for s in stats], out, n_docs,
                        MIN_DF_DEFAULT, args.max_df_frac)
    merge_s = time.time() - t0

    vocab_hash = np.load(out / "vocab_hash.npy", mmap_mode="r")
    idf = np.load(out / "idf.npy", mmap_mode="r")

    log("pass 2: per-shard postings with global statistics")
    t0 = time.time()
    fp = stats_fingerprint(n_docs=n_docs, avgdl=round(avgdl, 6), k1=K1_DEFAULT, b=args.b,
                           stemmer=STEMMER, min_df=MIN_DF_DEFAULT,
                           max_df_frac=args.max_df_frac,
                           n_terms=vstat["n_terms_kept"], tok=TOKENIZER_VERSION)
    per_corpus_shard = {p: pass2_shard(p, out, vocab_hash, idf, avgdl,
                                       K1_DEFAULT, args.b, args.resume, fp, DOCS_PER_BLOCK_DEFAULT)
                        for p in shards}
    smeta = [m for p in shards for m in per_corpus_shard[p]]
    pass2_s = time.time() - t0
    log(f"pass 2 done: {pass2_s / 60:.1f} min")

    log("building global pmid table")
    all_pmid, all_shard, all_idx = [], [], []
    for si, m in enumerate(smeta):
        pm = np.load(out / "shards" / m["name"] / "pmid.npy")
        all_pmid.append(pm)
        all_shard.append(np.full(pm.size, si, dtype=np.int16))
        all_idx.append(np.arange(pm.size, dtype=np.int32))
    pm = np.concatenate(all_pmid) if all_pmid else np.zeros(0, np.int64)
    sh = np.concatenate(all_shard) if all_shard else np.zeros(0, np.int16)
    ix = np.concatenate(all_idx) if all_idx else np.zeros(0, np.int32)
    order = np.argsort(pm, kind="stable")
    pm = pm[order].astype(np.uint64)
    n_dup = int(np.count_nonzero(np.diff(pm) == 0))
    if n_dup:
        log(f"  WARNING: {n_dup} duplicate pmid(s) across shards; first occurrence wins")
    np.save(out / "pmid_sorted.npy", pm)
    np.save(out / "pmid_shard.npy", sh[order])
    np.save(out / "pmid_idx.npy", ix[order])

    log("recording shard checksums")
    corpus_records = []
    for p in shards:
        st = os.stat(p)
        blocks = per_corpus_shard[p]
        rec = {"corpus_shard": Path(p).name, "path": p, "bytes": st.st_size,
               "mtime": int(st.st_mtime),
               "n_docs": sum(m["n_docs"] for m in blocks),
               "index_shards": [m["name"] for m in blocks]}
        rec["sha256"] = sha256_file(p)
        drop_cache(p)
        corpus_records.append(rec)

    nnz = sum(m["nnz"] for m in smeta)
    idx_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()
                    and "_tmp" not in f.parts)
    meta = {
        "schema_version": "pool_bm25_index/1.0",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "built_by": "biolitbench/pool/build_bm25.py",
        "tokenizer_version": TOKENIZER_VERSION,
        "stemmer": STEMMER,
        "text_fields": ["title", "abstract"],
        "title_weight": FIELD_TITLE_WEIGHT,
        "bm25": {"variant": "lucene", "k1": K1_DEFAULT, "b": args.b,
                 "idf": "log(1+(N-df+0.5)/(df+0.5))",
                 "doc_length": "token count after tokenisation, before vocabulary pruning"},
        "n_docs": n_docs, "n_shards": len(smeta),
        "n_corpus_shards": len(shards), "docs_per_block": DOCS_PER_BLOCK_DEFAULT,
        "avgdl": avgdl, "nnz": nnz,
        "nnz_per_doc": round(nnz / n_docs, 2),
        "vocabulary": {**vstat, "min_df": MIN_DF_DEFAULT, "max_df_frac": args.max_df_frac},
        "duplicate_pmids": n_dup,
        "stats_fingerprint": fp,
        "index_bytes": idx_bytes,
        "build_seconds": {"pass1": round(pass1_s, 1), "vocab_merge": round(merge_s, 1),
                          "pass2": round(pass2_s, 1), "total": round(time.time() - t_start, 1)},
        "corpus_shards": corpus_records,
        "shards": [{"name": m["name"], "n_docs": m["n_docs"], "nnz": m["nnz"],
                    "n_terms": m["n_terms"], "docs_bin_bytes": m["docs_bin_bytes"],
                    "year_min": m["year_min"], "year_max": m["year_max"]} for m in smeta],
        "corpus_complete": None,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE: {n_docs} docs in {len(smeta)} index shard(s), {nnz} nnz, "
        f"index {idx_bytes / 1e9:.2f} GB, {meta['build_seconds']['total'] / 60:.1f} min total, "
        f"peak RSS {peak_rss_gb():.2f} GB")
    log(f"temporary pass-1 files kept in {tmp} (delete to reclaim "
        f"{sum(f.stat().st_size for f in tmp.iterdir()) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
