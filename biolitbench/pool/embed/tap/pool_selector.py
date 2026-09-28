"""Restricts the FAISS searches of AutoSurvey and SurveyForge to the row ids in
POOL_SELECTOR_CUTOFF_NPY: papers published before the task cutoff, without the evaluated review."""

from __future__ import annotations

import json
import os
import sys


def batch_index_filter(arxivid_to_index, results_arxivid):
    import faiss
    import numpy as np

    n = len(results_arxivid)
    results_index = [0] * n
    for i in range(n):
        results_index[i] = arxivid_to_index[results_arxivid[i]]
    sel = faiss.IDSelectorBatch(np.asarray(results_index, dtype="int64"))
    out = {"id_selector": sel}
    return out


batch_index_filter._pool_selector = True


def patch_module(mod) -> int:
    n = 0
    fn = getattr(mod, "get_index_filter", None)
    if callable(fn) and not getattr(fn, "_pool_selector", False):
        mod.get_index_filter = batch_index_filter
        n += 1
    if n:
        sys.stderr.write("[pool_selector] IDSelectorArray -> IDSelectorBatch in %s\n" % mod.__name__)
    return n


CUTOFF_ENV = "POOL_SELECTOR_CUTOFF_NPY"
_CUTOFF_SEL = None


def _cutoff_selector():
    global _CUTOFF_SEL
    if _CUTOFF_SEL is not None:
        return _CUTOFF_SEL
    path = os.environ.get(CUTOFF_ENV, "")
    if not path or not os.path.isfile(path):
        raise RuntimeError("%s=%r is not a file" % (CUTOFF_ENV, path))
    import faiss
    import numpy as np

    ids = np.load(path).astype("int64", copy=False)
    _CUTOFF_SEL = faiss.IDSelectorBatch(ids)
    sys.stderr.write("[pool_selector] cutoff selector: %d ids from %s\n"
                     % (ids.size, os.path.basename(path)))
    return _CUTOFF_SEL


def _cutoff_year():
    head = os.path.basename(os.environ.get(CUTOFF_ENV, "")).split(".")[0]
    return int(head) if head.isdigit() and len(head) == 4 else None


_REVIEW = {}


def _review_pmid():
    path = os.environ.get(CUTOFF_ENV, "")
    if path in _REVIEW:
        return _REVIEW[path]
    parts = os.path.basename(path).split(".")
    if _cutoff_year() is None or len(parts) != 3:
        raise RuntimeError("%s=%r is not a <cutoff>.<task>.npy cutoff view" % (CUTOFF_ENV, path))
    with open(os.path.join(os.path.dirname(path), "task_to_cutoff.json"), "r", encoding="utf-8") as f:
        ent = json.load(f)["tasks"][parts[1]]
    if int(ent["cutoff"]) != _cutoff_year() or not ent.get("review_pmid"):
        raise RuntimeError("task_to_cutoff.json disagrees with %s: %r" % (path, ent))
    _REVIEW[path] = str(ent["review_pmid"])
    return _REVIEW[path]


def _id_year(sid):
    pre = str(sid).split(".")[0]
    return int(pre[:4]) if len(pre) == 6 and pre.isdigit() else None


def _visible(sid, cut, review):
    y = _id_year(sid)
    return y is not None and y < cut and str(sid).split(".", 1)[-1] != review


def _survey_selector(inst, cut):
    cache = getattr(inst, "_pool_survey_sel", None)
    if cache is not None and cache[0] == cut:
        return cache[1]
    import faiss
    import numpy as np

    review = _review_pmid()
    allowed = [pos for pos, sid in inst.index_to_id.items() if _visible(sid, cut, review)]
    sel = faiss.IDSelectorBatch(np.asarray(allowed, dtype="int64"))
    inst._pool_survey_sel = (cut, sel)
    sys.stderr.write("[pool_selector] survey-db selector: %d/%d ids before %d\n"
                     % (len(allowed), len(inst.index_to_id), cut))
    return sel


def patch_surveyforge_retrieve(cls) -> int:
    import functools

    fn = getattr(cls, "retrieve", None)
    if not callable(fn) or getattr(fn, "_pool_cutoff", False):
        return 0

    @functools.wraps(fn)
    def wrapper(self, query, search_type='similarity', top_k=10, filter=None,
                fetch_k=20, __fn=fn, **kwargs):
        cut = _cutoff_year()
        if cut is None:
            return __fn(self, query, search_type=search_type, top_k=top_k,
                        filter=filter, fetch_k=fetch_k, **kwargs)
        review = _review_pmid()
        out = None
        for mult in (5, 25):
            res = __fn(self, query, search_type=search_type, top_k=top_k * mult,
                       filter=filter, fetch_k=max(fetch_k, top_k * mult), **kwargs)
            out = []
            short = False
            for row in res:
                keep = [d for d in row if _visible(getattr(d, "metadata", {}).get("id"), cut, review)]
                if len(keep) < top_k and len(row) >= top_k * mult:
                    short = True
                out.append(keep[:top_k])
            if not short:
                break
        return out

    wrapper._pool_cutoff = True
    cls.retrieve = wrapper
    sys.stderr.write("[pool_selector] cutoff post-filter applied to %s.%s.retrieve\n"
                     % (cls.__module__, cls.__name__))
    return 1


def patch_unfiltered_search(cls) -> int:
    import functools

    n = 0
    for name in ("search", "batch_search"):
        fn = getattr(cls, name, None)
        if not callable(fn) or getattr(fn, "_pool_cutoff", False):
            continue

        @functools.wraps(fn)
        def wrapper(self, *a, __fn=fn, **k):
            if type(self).__name__ == "database_survey":
                cut = _cutoff_year()
                sel = _survey_selector(self, cut) if cut is not None else None
            else:
                sel = _cutoff_selector()
            if sel is None:
                return __fn(self, *a, **k)
            import faiss
            import numpy as np

            qv = np.array(a[0] if a else k.get("query_vector", k.get("query_vectors"))).astype("float32")
            if qv.ndim == 1:
                qv = qv[None, :]
            top_k = k.get("top_k", a[1] if len(a) > 1 else 1)
            title = k.get("title", a[2] if len(a) > 2 else False)
            index = self.title_loaded_index if title else self.abs_loaded_index
            p = faiss.SearchParametersIVF()
            p.sel = sel
            D, I = index.search(qv, top_k, params=p)
            if __fn.__name__ == "search":
                return [self.index_to_id[i] for i in I[0] if i != -1]
            return [[self.index_to_id[i] for i in row if i != -1] for row in I]

        wrapper._pool_cutoff = True
        setattr(cls, name, wrapper)
        n += 1
    if n:
        sys.stderr.write("[pool_selector] cutoff selector applied to %s.%s\n"
                         % (cls.__module__, cls.__name__))
    return n


def patch_database_module(mod) -> int:
    if not any(isinstance(getattr(mod, c, None), type) for c in ("database", "database_survey", "GeneralRAG_langchain")):
        return 0
    _cutoff_selector()
    if _cutoff_year() is not None:
        _review_pmid()
    n = 0
    for cname in ("database", "database_survey"):
        cls = getattr(mod, cname, None)
        if isinstance(cls, type):
            n += patch_unfiltered_search(cls)
    cls = getattr(mod, "GeneralRAG_langchain", None)
    if isinstance(cls, type):
        n += patch_surveyforge_retrieve(cls)
    return n
