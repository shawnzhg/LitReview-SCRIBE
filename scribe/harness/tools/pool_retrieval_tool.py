#!/usr/bin/env python3
"""Retrieval client of the pool service: BM25 search under the task's cutoff and document opens with
call and document caps; it fails closed and logs every call."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PIN_META_SHA16 = "263f4ee078ad2e00"
PIN_STATS_FINGERPRINT = "e8e89d5ae21f6626"
PIN_N_DOCS = 26569243
POOL_SNAPSHOT_ID = f"pool-bm25-{PIN_META_SHA16}"
ROUTE_SERVED = "lexical_bm25_pool"
BACKEND_VERSION = "pool/1.1"
SERVICE_K_MAX = 1000
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

RETRIES = 4
TIMEOUT_S = 180.0
BACKOFF_S = 2.0


class BudgetExhausted(RuntimeError):
    pass


class PoolServiceDead(BaseException):
    pass


class PoolContractViolation(BaseException):
    pass


_DEAD = threading.Event()
_DEAD_WHY: list = []


def mark_dead(why: str):
    if not _DEAD.is_set():
        _DEAD_WHY.append(why)
        _DEAD.set()


def dead_reason():
    return _DEAD_WHY[0] if _DEAD_WHY else None


class PoolConfig:

    def __init__(self, url=None, cutoffs_path=None, index_dir=None, boundary_path=None):
        self.url = (url or os.environ.get("POOL_URL") or "").rstrip("/")
        if not self.url:
            raise PoolContractViolation("POOL_URL is not set: the pool backend has no service to talk to")
        cp = cutoffs_path or os.environ.get("POOL_CUTOFFS")
        if not cp or not Path(cp).is_file():
            raise PoolContractViolation(f"POOL_CUTOFFS missing or not a file: {cp!r}")
        self.cutoffs_path = str(Path(cp).resolve())
        raw = Path(cp).read_bytes()
        self.cutoffs_sha16 = hashlib.sha256(raw).hexdigest()[:16]
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise PoolContractViolation(f"{cp}: expected the {{task: year}} dict the service was started with")
        self.cutoffs = {str(k): int(v) for k, v in obj.items()}
        self.index_dir = index_dir or os.environ.get("POOL_INDEX")
        bp = boundary_path or os.environ.get("POOL_BOUNDARY")
        self.boundary, self.boundary_path = None, None
        if bp:
            try:
                self.boundary = json.loads(Path(bp).read_text())
            except (OSError, ValueError) as e:
                raise PoolContractViolation(f"POOL_BOUNDARY={bp!r} is set but unreadable ({type(e).__name__}: {e}); "
                                            f"the per-task cutoff negative case cannot run")
            if not isinstance(self.boundary, dict) or not isinstance(self.boundary.get("by_year"), dict):
                raise PoolContractViolation(f"POOL_BOUNDARY={bp!r} has no 'by_year' map")
            self.boundary_path = bp

    def cutoff_for(self, task_id: str) -> int:
        if task_id not in self.cutoffs:
            raise PoolContractViolation(f"task {task_id!r} is not in {self.cutoffs_path}: the service would "
                                        f"answer 400 (no default cutoff by design)")
        return self.cutoffs[task_id]


_CFG_LOCK = threading.Lock()
_CFG: dict = {}


def config() -> PoolConfig:
    with _CFG_LOCK:
        if "c" not in _CFG:
            _CFG["c"] = PoolConfig()
        return _CFG["c"]


def http_get(url: str, timeout: float = None, retries: int = None):
    timeout = TIMEOUT_S if timeout is None else timeout
    retries = RETRIES if retries is None else retries
    last = None
    t0 = time.time()
    for attempt in range(1, max(1, retries) + 1):
        if _DEAD.is_set():
            raise PoolServiceDead(f"pool service already marked dead: {dead_reason()}")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                body = r.read()
                return r.status, (json.loads(body) if body else None), round((time.time() - t0) * 1000, 2), attempt
        except urllib.error.HTTPError as e:
            if 400 <= e.code < 500:
                try:
                    body = json.loads(e.read() or b"null")
                except Exception:
                    body = None
                return e.code, body, round((time.time() - t0) * 1000, 2), attempt
            last = f"HTTP {e.code}"
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, http.client.HTTPException) as e:
            last = f"{type(e).__name__}: {e}"
        except json.JSONDecodeError as e:
            last = f"unparseable body: {e}"
        if attempt < retries:
            time.sleep(BACKOFF_S * (2 ** (attempt - 1)))
    why = f"GET {url} failed {retries}x (last: {last})"
    mark_dead(why)
    raise PoolServiceDead(why)


def sentences_of(pmid: str, title, abstract) -> list:
    out = []
    if title and str(title).strip():
        out.append({"locator": f"pmid{pmid}_0", "text": str(title).strip()})
    k = 1
    for part in SENTENCE_SPLIT.split((abstract or "").strip()):
        part = part.strip()
        if not part:
            continue
        out.append({"locator": f"pmid{pmid}_{k}", "text": part})
        k += 1
    out.sort(key=lambda s: s["locator"])
    return out


def _year(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class PoolRetrievalTool:

    def __init__(self, snapshot_id, cutoff_year, budget, trace_path=None, task_id=None, calls_path=None,
                 cfg: PoolConfig = None):
        if _DEAD.is_set():
            raise PoolServiceDead(f"pool service already marked dead: {dead_reason()}")
        if not task_id:
            raise PoolContractViolation("PoolRetrievalTool needs the task id (the service's /t/{task}/ prefix "
                                        "is what applies the cutoff); construct it through runners/pool_backend")
        if snapshot_id != POOL_SNAPSHOT_ID:
            raise PoolContractViolation(f"snapshot_id {snapshot_id!r} != {POOL_SNAPSHOT_ID!r}: a pool-backed "
                                        f"artifact carries the pool snapshot id")
        self.cfg = cfg or config()
        self.task_id = str(task_id)
        self.snapshot_id = snapshot_id
        self.cutoff = int(cutoff_year)
        svc_cut = self.cfg.cutoff_for(self.task_id)
        if svc_cut != self.cutoff:
            raise PoolContractViolation(f"{self.task_id}: TaskSpec cutoff {self.cutoff} != the service's "
                                        f"cutoff {svc_cut} ({self.cfg.cutoffs_path}); parity needs the same cutoff")
        self.budget = dict(budget)
        self.used = {"search_calls": 0, "document_opens": 0, "embed_tokens": 0,
                     "wall_ms": 0, "usd": 0.0, "post_cutoff_suppressed": 0,
                     "docs_read": 0, "rows_capped_per_call": 0, "docs_cap_dropped": 0}
        rpc, dcap = self.budget.get("max_results_per_call"), self.budget.get("max_docs_read")
        self.max_rpc = int(rpc) if rpc is not None else None
        self.max_docs = int(dcap) if dcap is not None else None
        if (self.max_rpc is not None and self.max_rpc < 1) or (self.max_docs is not None and self.max_docs < 1):
            raise PoolContractViolation(f"{task_id}: a budget cap below 1 ({self.max_rpc}, {self.max_docs})")
        self.delivered: set = set()
        self.exclude_ids: set = set()
        self.trace = []
        self.trace_path = Path(trace_path) if trace_path else None
        self.calls_path = Path(calls_path) if calls_path else None
        self._calls_lock = threading.Lock()
        self._searched = set()
        self.n_http = 0
        self.route_requested = {}

    def _spend(self, kind, n=1):
        key = {"search": "search_calls", "open": "document_opens"}[kind]
        cap = self.budget.get({"search": "max_search_calls", "open": "max_document_opens"}[kind])
        if cap is not None and self.used[key] + n > cap:
            raise BudgetExhausted(f"{key} budget {cap} exhausted")
        self.used[key] += n

    def _event(self, action, tool, t0, effect):
        self.trace.append({
            "t": len(self.trace), "stage": "acquisition", "action_class": action,
            "tool_class": tool,
            "resource_bin": {"in_tokens": 0, "out_tokens": 0, "calls": 1,
                             "wall_ms": int((time.time() - t0) * 1000),
                             "usd": round(self.used["usd"], 6)},
            "retry_or_failure": None, "observable_effect": effect})

    def flush(self):
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a") as f:
                for e in self.trace:
                    f.write(json.dumps(e) + "\n")
            self.trace = []

    def _record(self, rec: dict):
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z",
               "task": self.task_id, "cutoff": self.cutoff, "cutoffs_sha16": self.cfg.cutoffs_sha16,
               "service": self.cfg.url, "backend": BACKEND_VERSION, **rec}
        if self.calls_path is None:
            return
        with self._calls_lock:
            self.calls_path.parent.mkdir(parents=True, exist_ok=True)
            with self.calls_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _url(self, path: str, params: dict) -> str:
        q = urllib.parse.urlencode(params)
        return f"{self.cfg.url}/t/{urllib.parse.quote(self.task_id, safe='')}{path}" + (f"?{q}" if q else "")

    def search(self, query, k=50, route="both"):
        t0 = time.time()
        if route not in ("both", "lexical", "dense"):
            raise ValueError(f"unknown route {route!r}")
        if route == "dense":
            raise PoolContractViolation("route='dense' has no counterpart in the frozen pool (BM25 only); "
                                        "refusing rather than silently serving lexical results")
        if self.max_docs is not None and len(self.delivered) >= self.max_docs:
            self._record({"kind": "budget_refusal", "quantity": "docs_read", "query": query, "k_requested": k,
                          "docs_read": len(self.delivered), "cap": self.max_docs})
            raise BudgetExhausted(f"docs_read budget {self.max_docs} exhausted")
        self._spend("search")
        k_served = int(k) if self.max_rpc is None else min(int(k), self.max_rpc)
        k_served = min(k_served, SERVICE_K_MAX)
        if k_served < int(k):
            self.used["rows_capped_per_call"] += 1
        depth = min(max(k_served * 3, 100), SERVICE_K_MAX)
        url = self._url("/search", {"q": query, "k": depth})
        status, body, ms, attempts = http_get(url)
        self.n_http += 1
        self.route_requested[route] = self.route_requested.get(route, 0) + 1
        if status == 400:
            self._record({"kind": "search", "route": "/t/{task}/search", "url": url, "query": query,
                          "k_requested": k, "service_limit": depth, "status": status, "error": body})
            raise PoolContractViolation(f"{self.task_id}: the service answered 400 to a search ({body})")
        if status != 200 or not isinstance(body, list):
            self._record({"kind": "search", "route": "/t/{task}/search", "url": url, "query": query,
                          "k_requested": k, "service_limit": depth, "status": status, "error": body})
            raise PoolContractViolation(f"{self.task_id}: unexpected search answer HTTP {status}")
        raw = []
        for rk, r in enumerate(body, 1):
            pid = str(r.get("pmid") or "")
            y = _year(r.get("year"))
            raw.append({"pmid": pid, "rank": rk, "score": r.get("score"), "year": y})
            if y is None or y >= self.cutoff:
                self._record({"kind": "search", "route": "/t/{task}/search", "url": url, "query": query,
                              "status": status, "returned": raw, "violation": f"year {y} not before the cutoff"})
                raise PoolContractViolation(f"{self.task_id}: the service returned pmid {pid} of year {y}, not "
                                            f"before the cutoff {self.cutoff}; the cutoff gate is broken")
        pool = {}
        mx = max((r.get("score") or 0) for r in body) if body else 1.0
        for r in body:
            pid = str(r.get("pmid") or "")
            if not pid:
                continue
            sc = 0.9 * (r.get("score") or 0) / (mx or 1.0)
            if pid not in pool or sc > pool[pid]["score"]:
                pool[pid] = {"paper_id": pid, "title": r.get("title"), "abstract": r.get("abstract"),
                             "doi": None, "year": _year(r.get("year")), "score": sc, "route": "lexical"}
        out = sorted(pool.values(), key=lambda r: -r["score"])[:k_served]
        self._searched.update(p["paper_id"] for p in out)
        dropped = []
        if self.max_docs is not None:
            room, kept_rows = self.max_docs - len(self.delivered), []
            for p in out:
                pid = p["paper_id"]
                if pid in self.delivered or pid in self.exclude_ids:
                    kept_rows.append(p)
                elif room > 0:
                    kept_rows.append(p)
                    room -= 1
                else:
                    dropped.append(pid)
            out = kept_rows
            self.used["docs_cap_dropped"] += len(dropped)
        self.delivered.update(p["paper_id"] for p in out if p["paper_id"] not in self.exclude_ids)
        self.used["docs_read"] = len(self.delivered)
        for i, p in enumerate(out, 1):
            p["rank"] = i
        self._record({"kind": "search", "route": "/t/{task}/search", "url": url, "query": query,
                      "route_requested": route, "route_served": ROUTE_SERVED, "k_requested": k,
                      "k_served": k_served, "rpc_cap": self.max_rpc, "docs_cap": self.max_docs,
                      "service_limit": depth, "status": status, "latency_ms": ms, "attempts": attempts,
                      "n_returned": len(body), "returned": raw,
                      "kept": [{"pmid": p["paper_id"], "rank": p["rank"], "score": p["score"]} for p in out],
                      "docs_cap_dropped": dropped, "docs_read_after": len(self.delivered),
                      "post_cutoff_seen": 0})
        self._event("search", "pool_search", t0,
                    {"query": query, "k": k, "route": route, "route_served": ROUTE_SERVED, "n": len(out),
                     "post_cutoff_suppressed": 0})
        return out

    def open_document(self, pmid, sentences=True):
        t0 = time.time()
        pid = str(pmid)
        if (self.max_docs is not None and pid not in self.delivered and pid not in self.exclude_ids
                and len(self.delivered) >= self.max_docs):
            self._record({"kind": "budget_refusal", "quantity": "docs_read", "pmid": pid,
                          "docs_read": len(self.delivered), "cap": self.max_docs})
            raise BudgetExhausted(f"docs_read budget {self.max_docs} exhausted (open of an unseen pmid)")
        self._spend("open")
        url = self._url(f"/graph/v1/paper/PMID:{urllib.parse.quote(pid, safe='')}",
                        {"fields": "title,abstract,year"})
        status, body, ms, attempts = http_get(url)
        self.n_http += 1
        if status == 400:
            self._record({"kind": "open", "route": "/t/{task}/graph/v1/paper/{id}", "url": url, "pmid": pid,
                          "status": status, "error": body})
            raise PoolContractViolation(f"{self.task_id}: the service answered 400 to an open ({body})")
        if status == 404:
            self._record({"kind": "open", "route": "/t/{task}/graph/v1/paper/{id}", "url": url, "pmid": pid,
                          "status": status, "latency_ms": ms, "found": False})
            if pid in self._searched:
                raise PoolContractViolation(f"{self.task_id}: pmid {pid} was returned by search but the by-id "
                                            f"route answers 404; the service is inconsistent")
            self._event("open_document", "pool_fetch", t0, {"pmid": pid, "refused": "not_in_pool_or_post_cutoff"})
            return {"paper_id": pid, "refused": "not_in_pool_or_post_cutoff"}
        if status != 200 or not isinstance(body, dict):
            self._record({"kind": "open", "route": "/t/{task}/graph/v1/paper/{id}", "url": url, "pmid": pid,
                          "status": status, "error": body})
            raise PoolContractViolation(f"{self.task_id}: unexpected open answer HTTP {status}")
        y = _year(body.get("year"))
        if y is None or y >= self.cutoff:
            raise PoolContractViolation(f"{self.task_id}: by-id route served pmid {pid} of year {y}, not before the cutoff")
        doc = {"pmid": pid, "title": body.get("title") or None, "abstract": body.get("abstract") or None,
               "doi": None, "year": y}
        if pid not in self.exclude_ids:
            self.delivered.add(pid)
            self.used["docs_read"] = len(self.delivered)
        if sentences:
            doc["sentences"] = sentences_of(pid, doc["title"], doc["abstract"])
        self._record({"kind": "open", "route": "/t/{task}/graph/v1/paper/{id}", "url": url, "pmid": pid,
                      "status": status, "latency_ms": ms, "attempts": attempts, "found": True, "year": y,
                      "n_sentences": len(doc.get("sentences") or []), "sentence_source": "pool_regex_split"})
        self._event("open_document", "pool_fetch", t0, {"pmid": pid, "n_sentences": len(doc.get("sentences", []))})
        return doc

    def report(self):
        return {"snapshot_id": self.snapshot_id, "cutoff_year": self.cutoff,
                "budget": self.budget, "used": self.used,
                "backend": {"name": BACKEND_VERSION, "service": self.cfg.url, "task_route": f"/t/{self.task_id}/",
                            "cutoffs_path": self.cfg.cutoffs_path, "cutoffs_sha16": self.cfg.cutoffs_sha16,
                            "index_meta_sha16_pin": PIN_META_SHA16, "stats_fingerprint_pin": PIN_STATS_FINGERPRINT,
                            "route_served": ROUTE_SERVED, "route_requested": dict(self.route_requested),
                            "caps": {"max_search_calls": self.budget.get("max_search_calls"),
                                     "max_results_per_call": self.max_rpc, "max_docs_read": self.max_docs,
                                     "service_k_max": SERVICE_K_MAX},
                            "dense_route": "unavailable on the frozen pool (BM25 only)",
                            "sentence_source": "pool_regex_split (_0 title, _k abstract sentences; locator order)",
                            "n_http": self.n_http, "calls_path": str(self.calls_path) if self.calls_path else None}}


_META_SHA: dict = {}


def index_meta(index_dir) -> dict:
    p = Path(index_dir) / "meta.json"
    key = str(p)
    if key not in _META_SHA:
        raw = p.read_bytes()
        m = json.loads(raw)
        _META_SHA[key] = {"bm25_meta_sha256_16": hashlib.sha256(raw).hexdigest()[:16],
                          "stats_fingerprint": m.get("stats_fingerprint"), "n_docs": m.get("n_docs"),
                          "n_shards": m.get("n_shards"), "built_at": m.get("built_at"), "bm25": m.get("bm25")}
    return dict(_META_SHA[key])


def task_start_provenance(task_id: str, cfg: PoolConfig = None) -> dict:
    cfg = cfg or config()
    cut = cfg.cutoff_for(task_id)
    status, hz, ms, _ = http_get(f"{cfg.url}/healthz")
    if status != 200 or not isinstance(hz, dict) or not hz.get("ok"):
        raise PoolContractViolation(f"/healthz answered HTTP {status}: {hz}")
    prov = {"schema": "pool_provenance/1.0", "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task": task_id, "cutoff": cut, "cutoffs_path": cfg.cutoffs_path, "cutoffs_sha16": cfg.cutoffs_sha16,
            "service": cfg.url, "healthz": hz, "healthz_ms": ms,
            "pins": {"bm25_meta_sha256_16": PIN_META_SHA16, "stats_fingerprint": PIN_STATS_FINGERPRINT,
                     "n_docs": PIN_N_DOCS}}
    bad = []
    if hz.get("stats_fingerprint") != PIN_STATS_FINGERPRINT:
        bad.append(f"service stats_fingerprint {hz.get('stats_fingerprint')} != pin {PIN_STATS_FINGERPRINT}")
    if int(hz.get("n_docs") or -1) != PIN_N_DOCS:
        bad.append(f"service n_docs {hz.get('n_docs')} != pin {PIN_N_DOCS}")
    if cfg.index_dir:
        im = index_meta(cfg.index_dir)
        prov["index_dir"] = str(cfg.index_dir)
        prov.update({k: im[k] for k in ("bm25_meta_sha256_16", "stats_fingerprint", "n_docs", "n_shards", "bm25")})
        if im["bm25_meta_sha256_16"] != PIN_META_SHA16:
            bad.append(f"index meta sha16 {im['bm25_meta_sha256_16']} != pin {PIN_META_SHA16}")
        if Path(str(hz.get("index_dir") or "")).resolve() != Path(str(cfg.index_dir)).resolve():
            bad.append(f"service index_dir {hz.get('index_dir')} != POOL_INDEX {cfg.index_dir}")
    else:
        bad.append("POOL_INDEX not set: the on-disk index hash cannot be recorded")
    probe = None
    if cfg.boundary is not None:
        b = (cfg.boundary.get("by_year") or {})
        before, at = b.get(str(cut - 1)), b.get(str(cut))
        probe = {"boundary_file": cfg.boundary_path}
        for name, pm, want in (("before_cutoff", before, 200), ("at_cutoff", at, 404)):
            if pm is None:
                probe[name] = {"pmid": None, "note": f"no pool document of year {cut - 1 if want == 200 else cut}"}
                continue
            url = f"{cfg.url}/t/{urllib.parse.quote(task_id, safe='')}/graph/v1/paper/PMID:{pm}?fields=year"
            st, bd, pms, _ = http_get(url)
            probe[name] = {"pmid": str(pm), "status": st, "want": want,
                           "year": (bd or {}).get("year") if isinstance(bd, dict) else None}
            if st != want:
                bad.append(f"cutoff probe {name}: pmid {pm} answered {st}, want {want}")
        if before is None and at is None:
            bad.append(f"boundary file has no pmid for year {cut - 1} or {cut}")
    prov["cutoff_probe"] = probe
    prov["ok"] = not bad
    prov["problems"] = bad
    if bad:
        raise PoolContractViolation(f"{task_id}: pool provenance failed: " + "; ".join(bad))
    return prov
