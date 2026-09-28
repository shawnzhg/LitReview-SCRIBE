#!/usr/bin/env python3
"""Offline HTTP service that serves, per task, only the pool papers published before the cutoff and
never the evaluated review, logging every call. Usage: python pool_service.py --port <port> --index
<bm25 dir> --cutoffs <json> --gold <dir> --log <file>."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pool_common import UNDATED, PoolIndex, resolve_id, tokenize

MAX_LIMIT = int(os.environ.get("POOL_MAX_LIMIT", "100"))
MAX_BATCH = int(os.environ.get("POOL_MAX_BATCH", "500"))
BING_SNIPPET_CHARS = int(os.environ.get("POOL_BING_SNIPPET_CHARS", "320"))
EDGES_PATH = os.environ.get("POOL_EDGES", "data/pool_index/edges")
ALLOWLIST_DIR = os.environ.get("POOL_ALLOWLIST_DIR")
ALLOWLIST_CAND = int(os.environ.get("POOL_ALLOWLIST_CAND", "5000"))
EDGE_SCAN_CAP = int(os.environ.get("POOL_EDGE_SCAN_CAP", "50000"))


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def load_cutoffs(path) -> dict:
    p = Path(path)
    if not p.exists():
        sys.exit(f"POOL_CUTOFFS not found: {p}")
    text = p.read_text().strip()
    if not text:
        sys.exit(f"POOL_CUTOFFS is empty: {p}")

    def from_rows(rows):
        out = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            tid = r.get("task_id") or r.get("task")
            cut = r.get("cutoff", r.get("publication_cutoff"))
            if tid is not None and cut is not None:
                out[str(tid)] = int(cut)
        return out

    obj = None
    if text[0] in "{[":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
    if isinstance(obj, dict):
        if "task_id" in obj:
            cuts = from_rows([obj])
        else:
            cuts = {str(k): int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        cuts = from_rows(obj)
    else:
        cuts = from_rows(json.loads(ln) for ln in text.splitlines() if ln.strip())
    if not cuts:
        sys.exit(f"POOL_CUTOFFS parsed to zero tasks: {p}")
    for t, y in cuts.items():
        if not (1900 <= y <= 2100):
            sys.exit(f"implausible cutoff for {t}: {y}")
    return cuts


def load_reviews(gold_dir, tasks) -> dict:
    out = {}
    for t in tasks:
        p = Path(gold_dir) / f"{t}.json"
        rp = json.loads(p.read_text()).get("review_pmid") if p.is_file() else None
        if not rp:
            sys.exit(f"no review_pmid for task {t} in {p}: the evaluated review cannot be excluded")
        out[t] = str(rp)
    return out


class Edges:

    EMPTY = np.zeros(0, dtype=np.uint32)

    def __init__(self, path):
        self.path = Path(path)
        self.has_n_citation = False
        self._nc = None
        mj = self.path / "meta.json"
        if not mj.is_file():
            raise RuntimeError(f"POOL_EDGES={self.path} has no meta.json; build it with "
                               f"biolitbench/pool/build_edges_db.py")
        self.meta = json.loads(mj.read_text())
        m = lambda f: np.load(self.path / f, mmap_mode="r")
        self._fi, self._fv = m("fwd_indptr.npy"), m("fwd_val.npy")
        self._bi, self._bv = m("bwd_indptr.npy"), m("bwd_val.npy")
        nc = self.path / "ncit.npy"
        if nc.exists():
            self._nc = np.load(nc, mmap_mode="r")
            self.has_n_citation = True
        self.kind = "dense-csr"
        self.present = True

    def _slice(self, indptr, val, pmid):
        try:
            p = int(pmid)
        except (TypeError, ValueError):
            return self.EMPTY
        if p < 0 or p + 1 >= len(indptr):
            return self.EMPTY
        lo, hi = int(indptr[p]), int(indptr[p + 1])
        return self.EMPTY if hi <= lo else np.asarray(val[lo:hi])

    def refs(self, pmid) -> np.ndarray:
        return self._slice(self._fi, self._fv, pmid)

    def cited_by(self, pmid) -> np.ndarray:
        return self._slice(self._bi, self._bv, pmid)

    @staticmethod
    def _count(indptr, pmid) -> int:
        try:
            p = int(pmid)
        except (TypeError, ValueError):
            return 0
        if p < 0 or p + 1 >= len(indptr):
            return 0
        return int(indptr[p + 1]) - int(indptr[p])

    def n_refs(self, pmid) -> int:
        return self._count(self._fi, pmid)

    def n_cited_by(self, pmid) -> int:
        return self._count(self._bi, pmid)

    def n_citation(self, pmid):
        if self._nc is None:
            return None
        try:
            p = int(pmid)
        except (TypeError, ValueError):
            return None
        if p < 0 or p >= len(self._nc):
            return None
        v = int(self._nc[p])
        return None if v < 0 else v


class PoolLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self.n = 0

    def write(self, **rec):
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            self.n += 1


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


class State:
    def __init__(self, index_dir, cutoffs_path, gold_dir, log_path, base_url):
        self.index = PoolIndex(index_dir)
        self.cutoffs = load_cutoffs(cutoffs_path)
        self.reviews = load_reviews(gold_dir, self.cutoffs)
        self.log = PoolLog(log_path)
        self.base_url = base_url.rstrip("/") if base_url else None
        self.edges = Edges(EDGES_PATH)
        self._allow = {}
        self._allow_lock = threading.Lock()
        self._cache = {}
        self._cache_lock = threading.Lock()

    def allowlist(self, task: str):
        if not ALLOWLIST_DIR:
            return None
        with self._allow_lock:
            if task in self._allow:
                return self._allow[task]
        p = Path(ALLOWLIST_DIR) / f"{task}.json"
        if not p.exists():
            raise HTTPException(status_code=400, detail=f"POOL_ALLOWLIST_DIR is set but has no allowlist for task {task!r}")
        with open(p) as f:
            allow = {str(x) for x in json.load(f)}
        with self._allow_lock:
            self._allow[task] = allow
        return allow

    def visible(self, task: str, doc) -> bool:
        if doc is None or doc.get("year") is None or int(doc["year"]) >= self.cutoffs[task]:
            return False
        if str(doc["pmid"]) == self.reviews[task]:
            return False
        a = self.allowlist(task)
        return a is None or str(doc["pmid"]) in a

    def ranked(self, task: str, query: str, k: int, ymin=None, ymax=None):
        cutoff = self.cutoffs[task]
        review = self.reviews[task]
        allow = self.allowlist(task)
        key = (task, query, k, ymin, ymax)
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        if allow is None:
            val = self.index.search(query, cutoff=cutoff, k=k, year_min=ymin, year_max=ymax, exclude=(review,))
        else:
            want = max(k, len(allow))
            hits, _ = self.index.search(query, cutoff=cutoff, k=max(want, ALLOWLIST_CAND), year_min=ymin,
                                        year_max=ymax, exclude=(review,))
            kept = [h for h in hits if str(h["pmid"]) in allow]
            seen = {str(h["pmid"]) for h in kept}
            ix = self.index
            for pm in sorted(allow - seen - {review}, key=lambda x: int(x)):
                try:
                    pk = np.uint64(int(pm))
                except (TypeError, ValueError):
                    continue
                i = int(np.searchsorted(ix.pmid_sorted, pk))
                if i >= len(ix.pmid_sorted) or int(ix.pmid_sorted[i]) != int(pk):
                    continue
                yr = ix.year_of(pm)
                if yr is None or yr >= cutoff:
                    continue
                if ymin is not None and yr < ymin or ymax is not None and yr > ymax:
                    continue
                kept.append({"pmid": str(pm), "score": 0.0,
                             "_shard": int(ix.pmid_shard[i]), "_idx": int(ix.pmid_idx[i])})
            val = (kept[:want], len(kept))
        with self._cache_lock:
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = val
        return val

    def cutoff_or_400(self, task: str) -> int:
        try:
            return self.cutoffs[task]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=(f"unknown task_id {task!r}: not in POOL_CUTOFFS. There is no default "
                        f"cutoff, because a default would serve post-cutoff literature."))


ST: State = None


def visible_doc(task: str, pmid):
    doc = ST.index.get(pmid) if pmid else None
    return doc if ST.visible(task, doc) else None


def base_url(request: Request) -> str:
    if ST.base_url:
        return ST.base_url
    u = request.url
    return f"{u.scheme}://{u.netloc}"


def bibkey(pmid: str) -> str:
    return f"pmid{pmid}"


def bibtex(doc: dict) -> str:
    title = (doc.get("title") or "").replace("{", "").replace("}", "")
    return ("@article{%s,\n  title={%s},\n  author={Unknown},\n  year={%s},\n"
            "  doi={%s},\n  note={PMID: %s}\n}" %
            (bibkey(doc["pmid"]), title, doc.get("year") or "", doc.get("doi") or "",
             doc["pmid"]))


MAX_EDGE_STUBS = int(os.environ.get("POOL_MAX_EDGE_STUBS", "100"))


def citation_count(pmid: str) -> int:
    n = ST.edges.n_citation(pmid)
    return n if n is not None else ST.edges.n_cited_by(pmid)


def visible_neighbours(task: str, neigh: np.ndarray, limit) -> np.ndarray:
    if neigh.size == 0:
        return neigh
    scan = neigh[:EDGE_SCAN_CAP]
    years = ST.index.years_of_many(scan)
    ok = scan[(years != UNDATED) & (years < ST.cutoffs[task]) & (scan.astype(np.int64) != int(ST.reviews[task]))]
    return ok if limit is None or ok.size <= limit else ok[:limit]


def edge_stubs(task: str, neigh: np.ndarray) -> list:
    return [{"paperId": f"PMID:{int(n)}"} for n in visible_neighbours(task, neigh, MAX_EDGE_STUBS)]


def s2_paper(doc: dict, req_base: str, task: str, fields=None) -> dict:
    pmid = doc["pmid"]
    ext = {"PubMed": pmid, "CorpusId": int(pmid)}
    if doc.get("doi"):
        ext["DOI"] = doc["doi"]
    oa = {"url": f"{req_base}/t/{task}/pdf/{pmid}", "status": "GREEN"}
    full = {
        "paperId": f"PMID:{pmid}",
        "corpusId": int(pmid),
        "externalIds": ext,
        "url": f"{req_base}/t/{task}/page/{pmid}",
        "title": doc.get("title") or "",
        "abstract": doc.get("abstract") or "",
        "venue": "",
        "publicationVenue": None,
        "year": doc.get("year"),
        "referenceCount": ST.edges.n_refs(pmid),
        "citationCount": citation_count(pmid),
        "influentialCitationCount": 0,
        "isOpenAccess": True,
        "openAccessPdf": oa,
        "fieldsOfStudy": None,
        "s2FieldsOfStudy": None,
        "publicationTypes": None,
        "publicationDate": None,
        "journal": None,
        "citationStyles": {"bibtex": bibtex(doc)},
        "authors": [],
    }
    keep = None if not fields else {"paperId"} | {f.strip() for f in fields if f.strip()}
    if keep is None or "references" in keep:
        full["references"] = edge_stubs(task, ST.edges.refs(pmid))
    if keep is None or "citations" in keep:
        full["citations"] = edge_stubs(task, ST.edges.cited_by(pmid))
    if keep is None:
        return full
    return {k: v for k, v in full.items() if k in keep}


def parse_fields(fields: str):
    return [f for f in (fields or "").split(",") if f.strip()] or None


def parse_year_param(year: str):
    if not year:
        return None, None
    y = year.strip()
    if "-" not in y:
        try:
            v = int(y)
            return v, v
        except ValueError:
            return None, None
    lo, _, hi = y.partition("-")
    try:
        lo_i = int(lo) if lo.strip() else None
    except ValueError:
        lo_i = None
    try:
        hi_i = int(hi) if hi.strip() else None
    except ValueError:
        hi_i = None
    return lo_i, hi_i


def docs_for_hits(hits):
    return [ST.index.doc_at(h["_shard"], h["_idx"]) for h in hits]


def logged(task, route, params, cutoff, returned, ms, **extra):
    ST.log.write(ts=utcnow(), task=task, route=route, params=params, cutoff=cutoff,
                 n_returned=len(returned), returned=returned, latency_ms=round(ms, 2), **extra)


app = FastAPI(title="pool retrieval service", docs_url=None, redoc_url=None)


@app.get("/healthz")
def healthz():
    m = ST.index.meta
    return {"ok": True, "n_docs": m["n_docs"], "n_shards": m["n_shards"],
            "index_dir": str(ST.index.dir), "built_at": m.get("built_at"),
            "stats_fingerprint": m.get("stats_fingerprint"),
            "corpus_complete": m.get("corpus_complete"),
            "n_tasks": len(ST.cutoffs), "log": str(ST.log.path), "log_lines": ST.log.n,
            "edges_present": ST.edges.present, "edges_kind": ST.edges.kind,
            "edges_path": str(ST.edges.path), "edges_n_input": ST.edges.meta.get("n_edges_input"),
            "has_n_citation": ST.edges.has_n_citation,
            "citation_count_mode": "global", "s2_pdf_mode": "local"}


@app.get("/t/{task}/graph/v1/paper/search")
def s2_search(task: str, request: Request,
              query: str = Query("", alias="query"),
              fields: str = Query(""),
              limit: int = Query(10),
              offset: int = Query(0),
              year: str = Query("")):
    cutoff = ST.cutoff_or_400(task)
    limit = max(0, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))
    ymin, ymax = parse_year_param(year)
    with Timer() as t:
        hits, total = ST.ranked(task, query, offset + limit, ymin, ymax)
        page = hits[offset:offset + limit]
        docs = docs_for_hits(page)
        bu = base_url(request)
        data = [s2_paper(d, bu, task, parse_fields(fields)) for d in docs]
    body = {"total": total, "offset": offset, "data": data}
    if offset + len(data) < total:
        body["next"] = offset + len(data)
    logged(task, "s2_search",
           {"query": query, "fields": fields, "limit": limit, "offset": offset, "year": year},
           cutoff, [{"pmid": d["pmid"], "score": h["score"]} for d, h in zip(docs, page)],
           t.ms, total=total)
    return body


@app.post("/t/{task}/graph/v1/paper/batch")
async def s2_batch(task: str, request: Request, fields: str = Query("")):
    cutoff = ST.cutoff_or_400(task)
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    ids = payload.get("ids") or []
    if not isinstance(ids, list):
        ids = []
    truncated = len(ids) > MAX_BATCH
    ids = [str(i) for i in ids[:MAX_BATCH]]
    with Timer() as t:
        bu = base_url(request)
        keep = parse_fields(fields)
        out, got = [], []
        for raw in ids:
            doc = visible_doc(task, resolve_id(raw))
            if doc is None:
                out.append(None)
            else:
                out.append(s2_paper(doc, bu, task, keep))
                got.append(doc["pmid"])
    logged(task, "s2_batch", {"n_ids": len(ids), "fields": fields, "truncated": truncated},
           cutoff, [{"pmid": p, "score": None} for p in got], t.ms,
           n_null=sum(1 for o in out if o is None))
    return JSONResponse(out)


def _edge_page(task: str, request: Request, paper_id: str, fields: str, limit: int,
               offset: int, direction: str):
    cutoff = ST.cutoff_or_400(task)
    pmid = resolve_id(paper_id)
    neigh = (ST.edges.cited_by(pmid) if direction == "citations" else ST.edges.refs(pmid)) \
        if pmid else Edges.EMPTY
    wrapper = "citingPaper" if direction == "citations" else "citedPaper"
    limit = max(0, min(int(limit), MAX_LIMIT))
    offset = max(0, int(offset))
    with Timer() as t:
        bu = base_url(request)
        keep = parse_fields(fields)
        visible = visible_neighbours(task, neigh, None)
        data, got = [], []
        for n in visible[offset:offset + limit].tolist():
            doc = visible_doc(task, n)
            if doc is None:
                continue
            data.append({wrapper: s2_paper(doc, bu, task, keep)})
            got.append(doc["pmid"])
    logged(task, f"s2_{direction}",
           {"paper_id": paper_id, "fields": fields, "limit": limit, "offset": offset},
           cutoff, [{"pmid": p, "score": None} for p in got], t.ms,
           edges_present=ST.edges.present, n_neighbours=int(neigh.size),
           n_visible=int(visible.size))
    body = {"offset": offset, "data": data}
    if offset + len(data) < int(visible.size):
        body["next"] = offset + len(data)
    return body


@app.get("/t/{task}/graph/v1/paper/{paper_id}/citations")
def s2_citations(task: str, request: Request, paper_id: str, fields: str = Query(""),
                 limit: int = Query(100), offset: int = Query(0)):
    return _edge_page(task, request, paper_id, fields, limit, offset, "citations")


@app.get("/t/{task}/graph/v1/paper/{paper_id}/references")
def s2_references(task: str, request: Request, paper_id: str, fields: str = Query(""),
                  limit: int = Query(100), offset: int = Query(0)):
    return _edge_page(task, request, paper_id, fields, limit, offset, "references")


@app.get("/t/{task}/graph/v1/paper/{paper_id}")
def s2_paper_by_id(task: str, request: Request, paper_id: str, fields: str = Query("")):
    cutoff = ST.cutoff_or_400(task)
    with Timer() as t:
        doc = visible_doc(task, resolve_id(paper_id))
        body = None if doc is None else s2_paper(doc, base_url(request), task, parse_fields(fields))
    logged(task, "s2_paper", {"paper_id": paper_id, "fields": fields}, cutoff,
           ([{"pmid": doc["pmid"], "score": None}] if doc else []), t.ms)
    if doc is None:
        return JSONResponse({"error": "Paper not found"}, status_code=404)
    return body


SNIPPET_CHARS = int(os.environ.get("POOL_SNIPPET_CHARS", "500"))
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_spans(text: str):
    out, i = [], 0
    for part in _SENT_SPLIT.split(text):
        if not part:
            continue
        j = text.find(part, i)
        if j < 0:
            j = i
        out.append((j, j + len(part)))
        i = j + len(part)
    return out or [(0, len(text))]


def best_window(abstract: str, qterms: set):
    spans = sentence_spans(abstract)
    scored = []
    for k, (a, b) in enumerate(spans):
        toks = set(tokenize(abstract[a:b], ST.index.stemmer))
        scored.append((len(toks & qterms), -k, k))
    best = max(scored)[2]
    lo = hi = best
    start, end = spans[best]
    while True:
        grew = False
        if hi + 1 < len(spans) and spans[hi + 1][1] - start <= SNIPPET_CHARS:
            hi += 1
            end = spans[hi][1]
            grew = True
        if lo - 1 >= 0 and end - spans[lo - 1][0] <= SNIPPET_CHARS:
            lo -= 1
            start = spans[lo][0]
            grew = True
        if not grew:
            break
    return start, end, spans[lo:hi + 1]


@app.get("/t/{task}/graph/v1/snippet/search")
def s2_snippet_search(task: str, request: Request,
                      query: str = Query(""),
                      limit: int = Query(10),
                      year: str = Query(""),
                      paperIds: str = Query(""),
                      venue: str = Query("")):
    cutoff = ST.cutoff_or_400(task)
    limit = max(0, min(int(limit), MAX_LIMIT))
    ymin, ymax = parse_year_param(year)
    want = None
    if paperIds.strip():
        want = {resolve_id(x) for x in paperIds.split(",")}
        want.discard(None)
    with Timer() as t:
        depth = limit if want is None else max(limit * 20, 200)
        hits, total = ST.ranked(task, query, depth, ymin, ymax)
        if want is not None:
            hits = [h for h in hits if h["pmid"] in want]
        hits = hits[:limit]
        docs = docs_for_hits(hits)
        qterms = set(tokenize(query, ST.index.stemmer))
        data = []
        for d, h in zip(docs, hits):
            abstract = (d.get("abstract") or "").strip()
            start, end, spans = best_window(abstract, qterms) if abstract else (0, 0, [])
            text = abstract[start:end]
            data.append({
                "snippet": {
                    "text": text,
                    "snippetKind": "abstract",
                    "section": "abstract",
                    "snippetOffset": {"start": start, "end": end},
                    "annotations": {
                        "sentences": [{"start": a - start, "end": b - start} for a, b in spans],
                        "refMentions": [],
                    },
                },
                "score": h["score"],
                "paper": {
                    "corpusId": str(d["pmid"]),
                    "title": d.get("title") or "",
                    "authors": [],
                    "openAccessInfo": {"license": "", "status": "CLOSED",
                                       "disclaimer": "Served from the local evaluation pool "
                                                     "(title+abstract only)."},
                },
            })
    logged(task, "s2_snippet_search",
           {"query": query, "limit": limit, "year": year, "paperIds": paperIds, "venue": venue},
           cutoff, [{"pmid": d["pmid"], "score": h["score"]} for d, h in zip(docs, hits)],
           t.ms, total=total)
    return {"data": data}


@app.get("/t/{task}/v7.0/search")
def bing_search(task: str, request: Request,
                q: str = Query(""),
                count: int = Query(10),
                offset: int = Query(0),
                mkt: str = Query("en-US"),
                responseFilter: str = Query("")):
    cutoff = ST.cutoff_or_400(task)
    count = max(0, min(int(count), 50))
    offset = max(0, int(offset))
    with Timer() as t:
        hits, total = ST.ranked(task, q, offset + count)
        page = hits[offset:offset + count]
        docs = docs_for_hits(page)
        bu = base_url(request)
        value = []
        for i, d in enumerate(docs):
            snip = (d.get("abstract") or "").strip().replace("\n", " ")
            if len(snip) > BING_SNIPPET_CHARS:
                snip = snip[:BING_SNIPPET_CHARS].rsplit(" ", 1)[0] + " ..."
            url = f"{bu}/t/{task}/page/{d['pmid']}"
            value.append({
                "id": f"{bu}/t/{task}/v7.0/search#WebPages.{offset + i}",
                "name": d.get("title") or f"PMID {d['pmid']}",
                "url": url,
                "displayUrl": url,
                "snippet": snip,
                "dateLastCrawled": f"{d.get('year') or cutoff}-01-01T00:00:00.0000000Z",
                "language": "en",
                "isFamilyFriendly": True,
                "isNavigational": False,
            })
    body = {"_type": "SearchResponse",
            "queryContext": {"originalQuery": q},
            "webPages": {"webSearchUrl": f"{bu}/t/{task}/v7.0/search?q={q}",
                         "totalEstimatedMatches": total,
                         "value": value}}
    logged(task, "bing_search", {"q": q, "count": count, "offset": offset, "mkt": mkt},
           cutoff, [{"pmid": d["pmid"], "score": h["score"]} for d, h in zip(docs, page)],
           t.ms, total=total)
    return body


_PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<meta name="citation_pmid" content="{pmid}">
<meta name="citation_publication_date" content="{year}">
<meta name="description" content="{desc}">
</head><body>
<article>
<h1>{title}</h1>
<p><strong>PMID:</strong> {pmid} &middot; <strong>Year:</strong> {year}{doi_html}</p>
<h2>Abstract</h2>
<p>{abstract}</p>
</article>
</body></html>
"""


def _esc(s: str) -> str:
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


@app.get("/t/{task}/page/{pmid}", response_class=HTMLResponse)
def page(task: str, pmid: str):
    cutoff = ST.cutoff_or_400(task)
    with Timer() as t:
        doc = visible_doc(task, resolve_id(pmid))
    logged(task, "page", {"pmid": pmid}, cutoff,
           ([{"pmid": doc["pmid"], "score": None}] if doc else []), t.ms)
    if doc is None:
        raise HTTPException(status_code=404, detail="not in pool, not before the task cutoff, or the evaluated review")
    abstract = _esc(doc.get("abstract"))
    doi = doc.get("doi")
    return HTMLResponse(_PAGE_HTML.format(
        title=_esc(doc.get("title")), pmid=_esc(doc["pmid"]), year=_esc(doc.get("year")),
        desc=abstract[:300], abstract=abstract,
        doi_html=(f' &middot; <strong>DOI:</strong> {_esc(doi)}' if doi else "")))


@app.get("/t/{task}/pdf/{pmid}")
def pdf(task: str, pmid: str):
    cutoff = ST.cutoff_or_400(task)
    with Timer() as t:
        doc = visible_doc(task, resolve_id(pmid))
    logged(task, "pdf", {"pmid": pmid}, cutoff,
           ([{"pmid": doc["pmid"], "score": None}] if doc else []), t.ms)
    if doc is None:
        raise HTTPException(status_code=404, detail="not in pool, not before the task cutoff, or the evaluated review")
    return Response(content=_minimal_pdf(doc), media_type="application/pdf")


def _minimal_pdf(doc: dict) -> bytes:
    def esc(s):
        return (str(s or "").replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)"))

    words, lines, cur = (str(doc.get("title") or "") + " . " +
                         str(doc.get("abstract") or "")).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 95:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    lines = [f"PMID {doc['pmid']} ({doc.get('year')})"] + lines
    stream = ("BT /F1 9 Tf 36 756 Td 11 TL\n" +
              "".join(f"({esc(l)}) Tj T*\n" for l in lines[:66]) + "ET")
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1", "replace")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
            ).encode()
    return bytes(out)


@app.get("/t/{task}/search")
def plain_search(task: str, q: str = Query(""), k: int = Query(20)):
    cutoff = ST.cutoff_or_400(task)
    k = max(0, min(int(k), 1000))
    with Timer() as t:
        hits, total = ST.ranked(task, q, k)
        docs = docs_for_hits(hits)
        out = [{"pmid": d["pmid"], "title": d.get("title") or "",
                "abstract": d.get("abstract") or "", "year": d.get("year"),
                "score": round(h["score"], 6)} for d, h in zip(docs, hits)]
    logged(task, "plain_search", {"q": q, "k": k}, cutoff,
           [{"pmid": r["pmid"], "score": r["score"]} for r in out], t.ms, total=total)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8931)
    ap.add_argument("--index", default=os.environ.get("POOL_INDEX", "data/pool_index/bm25"))
    ap.add_argument("--cutoffs", default=os.environ.get("POOL_CUTOFFS"))
    ap.add_argument("--gold", default=os.environ.get("POOL_GOLD_DIR"))
    ap.add_argument("--log", default=os.environ.get("POOL_LOG"))
    args = ap.parse_args()
    if not args.cutoffs:
        sys.exit("POOL_CUTOFFS (or --cutoffs) is required: task_id -> publication cutoff year")
    if not args.gold:
        sys.exit("POOL_GOLD_DIR (or --gold) is required: the gold records naming each task's review_pmid")
    if not args.log:
        sys.exit("POOL_LOG (or --log) is required: every request must be attributable")

    global ST
    base = os.environ.get("POOL_BASE_URL") or f"http://{args.host}:{args.port}"
    ST = State(args.index, args.cutoffs, args.gold, args.log, base)
    print(f"[pool] index={args.index} docs={ST.index.meta['n_docs']} "
          f"shards={ST.index.meta['n_shards']} tasks={len(ST.cutoffs)} "
          f"edges={ST.edges.kind} n_citation={'yes' if ST.edges.has_n_citation else 'no'}", flush=True)
    print(f"[pool] listening on http://{args.host}:{args.port}  base_url={base}", flush=True)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
