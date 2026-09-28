#!/usr/bin/env python3
"""Wraps the retrieval tool of the acquisition window: enforces the budget, excludes the evaluated
review, logs the ranked results and caches abstracts and sentences."""

from __future__ import annotations
import sys
import threading
import traceback
from pathlib import Path

from calllog import Timer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from pool_retrieval_tool import BudgetExhausted

_CACHE_LOCK = threading.Lock()


class MeteredRetrieval:

    def __init__(self, snapshot_id, cutoff_year, budget, calllog, doccache, window="acquisition",
                 trace_path=None, exclude_ids=None):
        self.tool = self._build_tool(snapshot_id, cutoff_year, budget, trace_path)
        self.log = calllog
        self.window = window
        self.snapshot_id = snapshot_id
        self.cutoff_year = int(cutoff_year)
        self.docs = doccache
        self.pool: dict = {}
        self.opened: set = set()
        self.exhausted = {"search": False, "open": False}
        self.exclude_ids = {str(x) for x in (exclude_ids or set())}
        self.n_excluded = 0

    def _build_tool(self, snapshot_id, cutoff_year, budget, trace_path):
        raise NotImplementedError("the pool backend (pool_backend.install_acquisition) builds the retrieval tool")

    def search(self, query, k=50, route="both", step=None):
        with Timer() as t:
            try:
                rows = self.tool.search(query, k=k, route=route)
            except BudgetExhausted as e:
                self.exhausted["search"] = True
                self.log.failure(self.window, t.started, "BudgetExhausted", str(e),
                                 kind="budget_refusal", wall_ms=0)
                return None
            except Exception as e:
                self.log.failure(self.window, t.started, type(e).__name__, str(e),
                                 traceback=traceback.format_exc()[:4000])
                return None

        if self.exclude_ids:
            before = len(rows)
            rows = [r for r in rows if str(r["paper_id"]) not in self.exclude_ids]
            self.n_excluded += before - len(rows)
            for i, r in enumerate(rows, 1):
                r["rank"] = i

        added = []
        for r in rows:
            pid = str(r["paper_id"])
            self.docs.put({"paper_id": pid, "title": r.get("title"), "abstract": r.get("abstract"),
                           "doi": r.get("doi"), "year": r.get("year")})
            prev = self.pool.get(pid)
            if prev is None or (r.get("score") or 0) > (prev.get("score") or 0):
                r = dict(r)
                r["first_seen_step"] = prev["first_seen_step"] if prev else step
                r["query"] = query if prev is None else prev.get("query", query)
                self.pool[pid] = r
            if prev is None:
                added.append(pid)

        self.log.retrieval_search(
            self.window, t.started, t.ms, query=query, route=route, k_requested=k,
            cutoff_year=self.cutoff_year, snapshot_id=self.snapshot_id,
            returned=[{"paper_id": str(r["paper_id"]), "rank": r["rank"],
                       "score": float(r.get("score") or 0.0), "route": r.get("route")}
                      for r in rows],
            post_cutoff_suppressed=self.tool.used["post_cutoff_suppressed"],
            n_new=len(added),
            artifact_delta={"papers_added": added, "relations_added": 0})
        return rows

    def open_document(self, pmid, step=None):
        pmid = str(pmid)
        if pmid in self.exclude_ids:
            self.n_excluded += 1
            self.log.failure(self.window, __import__("common").utcnow(), "ExcludedByScope",
                             f"refused to open {pmid}: excluded by TaskSpec.scope.exclusion",
                             recovered=True)
            return None
        with Timer() as t:
            try:
                doc = self.tool.open_document(pmid, sentences=True)
            except BudgetExhausted as e:
                self.exhausted["open"] = True
                self.log.failure(self.window, t.started, "BudgetExhausted", str(e),
                                 kind="budget_refusal")
                return None
            except Exception as e:
                self.log.failure(self.window, t.started, type(e).__name__, str(e),
                                 traceback=traceback.format_exc()[:4000])
                return None
        if doc.get("refused"):
            self.log.retrieval_open(self.window, t.started, t.ms, pmid, 0, self.snapshot_id,
                                    self.cutoff_year, refused=doc["refused"])
            return doc
        rec = self.docs.put({"paper_id": pmid, "title": doc.get("title"),
                             "abstract": doc.get("abstract"), "doi": doc.get("doi"),
                             "year": doc.get("year"), "sentences": doc.get("sentences") or []})
        self.opened.add(pmid)
        self.log.retrieval_open(self.window, t.started, t.ms, pmid,
                                len(rec.get("sentences") or []), self.snapshot_id,
                                self.cutoff_year)
        return rec

    def report(self):
        r = self.tool.report()
        r["exhausted"] = dict(self.exhausted)
        r["pool_size"] = len(self.pool)
        r["n_opened"] = len(self.opened)
        r["n_excluded_by_scope"] = self.n_excluded
        r["exclude_ids"] = sorted(self.exclude_ids)
        return r

    def remaining(self, kind):
        cap = self.tool.budget.get({"search": "max_search_calls", "open": "max_document_opens"}[kind])
        used = self.tool.used[{"search": "search_calls", "open": "document_opens"}[kind]]
        return None if cap is None else max(0, cap - used)

    def flush(self):
        with _CACHE_LOCK:
            self.tool.flush()
