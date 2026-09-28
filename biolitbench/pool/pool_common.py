#!/usr/bin/env python3
"""Shared code of the pool index: tokenizer, BM25 weights, record encoding, shard iteration, the
paper-id rule and a memory-mapped searcher that serves only dated papers published before the
cutoff year."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
import zlib
from pathlib import Path

import numpy as np

STOPWORDS = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",
    "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there",
    "these", "they", "this", "to", "was", "will", "with",
))

TOKENIZER_VERSION = "pool-tok/1.0"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


STEMMER = "english"
UNDATED = -32768


def make_stemmer():
    import Stemmer
    return Stemmer.Stemmer(STEMMER)


def tokenize(text: str, stemmer=None) -> list:
    if not text:
        return []
    toks = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]
    if stemmer is not None and toks:
        toks = stemmer.stemWords(toks)
    return toks


def resolve_id(raw: str) -> str:
    s = (raw or "").strip()
    if ":" in s:
        pre, _, rest = s.partition(":")
        if pre.lower() in ("pmid", "pubmed", "corpusid"):
            s = rest.strip()
        else:
            return None
    return s if s.isdigit() else None


def term_hash(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest(), "little")


def term_hash_many(terms) -> np.ndarray:
    b = hashlib.blake2b
    return np.fromiter(
        (int.from_bytes(b(t.encode("utf-8"), digest_size=8).digest(), "little") for t in terms),
        dtype=np.uint64, count=len(terms))


K1_DEFAULT = 0.9
B_DEFAULT = 0.4


def idf_lucene(df: np.ndarray, n_docs: int) -> np.ndarray:
    df = df.astype(np.float64)
    return np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)


class _Shard:
    __slots__ = ("name", "n_docs", "terms", "indptr", "docid", "score",
                 "pmid", "year", "docs_off", "_docs_path", "_docs_fh", "_buf")

    def __init__(self, d: Path, name: str, n_docs: int):
        self.name = name
        self.n_docs = n_docs
        m = lambda f: np.load(d / f, mmap_mode="r")
        self.terms = m("terms.npy")
        self.indptr = m("indptr.npy")
        self.docid = m("docid.npy")
        self.score = m("score.npy")
        self.pmid = m("pmid.npy")
        self.year = m("year.npy")
        self.docs_off = m("docs_off.npy")
        self._docs_path = d / "docs.bin"
        self._docs_fh = None
        self._buf = None

    def buf(self) -> np.ndarray:
        if self._buf is None:
            self._buf = np.zeros(self.n_docs, dtype=np.float32)
        return self._buf

    def doc(self, i: int) -> dict:
        if self._docs_fh is None:
            self._docs_fh = open(self._docs_path, "rb")
        lo, hi = int(self.docs_off[i]), int(self.docs_off[i + 1])
        self._docs_fh.seek(lo)
        return json.loads(zlib.decompress(self._docs_fh.read(hi - lo)))


class PoolIndex:

    def __init__(self, index_dir):
        self.dir = Path(index_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        if self.meta.get("tokenizer_version") != TOKENIZER_VERSION:
            raise RuntimeError(
                f"index was built with tokenizer {self.meta.get('tokenizer_version')!r} but this "
                f"code is {TOKENIZER_VERSION!r}; rebuild rather than serve mismatched tokens")
        if self.meta.get("stemmer") != STEMMER:
            raise RuntimeError(f"index stemmer {self.meta.get('stemmer')!r} != {STEMMER!r}; rebuild the index")
        self.stemmer = make_stemmer()
        self.vocab_hash = np.load(self.dir / "vocab_hash.npy", mmap_mode="r")
        self.idf = np.load(self.dir / "idf.npy", mmap_mode="r")
        self.n_docs = int(self.meta["n_docs"])
        self.shards = [_Shard(self.dir / "shards" / s["name"], s["name"], int(s["n_docs"]))
                       for s in self.meta["shards"]]
        self.pmid_sorted = np.load(self.dir / "pmid_sorted.npy", mmap_mode="r")
        self.pmid_shard = np.load(self.dir / "pmid_shard.npy", mmap_mode="r")
        self.pmid_idx = np.load(self.dir / "pmid_idx.npy", mmap_mode="r")
        self._lock = threading.Lock()

    def _term_ids(self, query: str):
        toks = tokenize(query, self.stemmer)
        if not toks:
            return None, None
        h = term_hash_many(toks)
        uniq, counts = np.unique(h, return_counts=True)
        pos = np.searchsorted(self.vocab_hash, uniq)
        pos_c = np.clip(pos, 0, len(self.vocab_hash) - 1)
        ok = np.asarray(self.vocab_hash[pos_c]) == uniq
        ok &= pos < len(self.vocab_hash)
        if not ok.any():
            return None, None
        return pos[ok].astype(np.int64), counts[ok].astype(np.float32)

    def search(self, query: str, cutoff: int, k: int, year_min: int = None,
               year_max: int = None, exclude=()):
        tids, qtf = self._term_ids(query)
        if tids is None:
            return [], 0
        hi_year = int(cutoff) - 1
        if year_max is not None:
            hi_year = min(hi_year, int(year_max))
        lo_year = UNDATED + 1 if year_min is None else max(UNDATED + 1, int(year_min))
        excl = np.asarray(sorted(int(x) for x in exclude), dtype=np.int64)

        out = []
        n_cand = 0
        with self._lock:
            for si, sh in enumerate(self.shards):
                sp = np.searchsorted(sh.terms, tids)
                sp_c = np.clip(sp, 0, len(sh.terms) - 1)
                present = (np.asarray(sh.terms[sp_c]) == tids) & (sp < len(sh.terms))
                if not present.any():
                    continue
                scores = sh.buf()
                touched = []
                for j in np.flatnonzero(present):
                    p = int(sp[j])
                    lo, hi = int(sh.indptr[p]), int(sh.indptr[p + 1])
                    if hi <= lo:
                        continue
                    d = np.asarray(sh.docid[lo:hi])
                    s = np.asarray(sh.score[lo:hi])
                    w = float(qtf[j])
                    if w == 1.0:
                        scores[d] += s
                    else:
                        scores[d] += s * w
                    touched.append(d)
                if not touched:
                    continue
                cand = touched[0] if len(touched) == 1 else np.unique(np.concatenate(touched))
                yr = np.asarray(sh.year[cand])
                keep = (yr <= hi_year) & (yr >= lo_year)
                if excl.size:
                    keep &= ~np.isin(np.asarray(sh.pmid[cand]).astype(np.int64), excl)
                sel = cand[keep]
                n_cand += int(sel.size)
                if sel.size:
                    sc = scores[sel]
                    if sel.size > k:
                        part = np.argpartition(-sc, k - 1)[:k]
                        sel, sc = sel[part], sc[part]
                    pm = np.asarray(sh.pmid[sel])
                    out.extend(zip((-sc).tolist(), pm.tolist(), [si] * len(sel), sel.tolist()))
                for d in touched:
                    scores[d] = 0.0
        out.sort(key=lambda r: (r[0], r[1]))
        return ([{"pmid": str(p), "score": float(-ns), "_shard": si, "_idx": ix}
                 for ns, p, si, ix in out[:k]], n_cand)

    def get(self, pmid) -> dict:
        try:
            key = np.uint64(int(pmid))
        except (TypeError, ValueError):
            return None
        i = int(np.searchsorted(self.pmid_sorted, key))
        if i >= len(self.pmid_sorted) or int(self.pmid_sorted[i]) != int(key):
            return None
        return self.doc_at(int(self.pmid_shard[i]), int(self.pmid_idx[i]))

    def year_of(self, pmid) -> int:
        try:
            key = np.uint64(int(pmid))
        except (TypeError, ValueError):
            return None
        i = int(np.searchsorted(self.pmid_sorted, key))
        if i >= len(self.pmid_sorted) or int(self.pmid_sorted[i]) != int(key):
            return None
        y = int(self.shards[int(self.pmid_shard[i])].year[int(self.pmid_idx[i])])
        return None if y == UNDATED else y

    def years_of_many(self, pmids) -> np.ndarray:
        keys = np.asarray(pmids).astype(np.uint64)
        out = np.full(keys.size, UNDATED, dtype=np.int32)
        if keys.size == 0:
            return out
        n = len(self.pmid_sorted)
        idx = np.searchsorted(self.pmid_sorted, keys)
        idxc = np.clip(idx, 0, n - 1)
        found = (np.asarray(self.pmid_sorted[idxc]) == keys) & (idx < n)
        f = np.flatnonzero(found)
        if f.size:
            pos = idxc[f]
            sh = np.asarray(self.pmid_shard[pos])
            ix = np.asarray(self.pmid_idx[pos])
            for s in np.unique(sh):
                m = sh == s
                out[f[m]] = np.asarray(self.shards[int(s)].year[ix[m]])
        return out

    def doc_at(self, shard_i: int, idx: int) -> dict:
        with self._lock:
            return self.shards[shard_i].doc(idx)


def doc_encode(record: dict) -> bytes:
    return zlib.compress(json.dumps(record, ensure_ascii=False).encode("utf-8"), 6)


def sha256_file(path, chunk=8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return "sha256:" + h.hexdigest()


def iter_shard(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
