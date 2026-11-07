"""Loaded through PYTHONPATH in the AutoSurvey and SurveyForge runs: always installs the SQLite
docstore and the row-id selector, logs every retrieval call to RETRIEVAL_TAP_LOG, and ends the
process on any failure."""

from __future__ import annotations

import functools
import json
import os
import sys
import threading
import time

_LOG = os.environ.get("RETRIEVAL_TAP_LOG", "")
_LOCK = threading.Lock()
_DEPTH = threading.local()
FATAL_EXIT = 86

_TARGET_MODULES = {"src.database", "src.rag", "src.utils", "database", "rag", "utils",
                   "src.agents.writer", "src.agents.outline_writer",
                   "agents.writer", "agents.outline_writer"}

_TARGET_METHODS = {
    "database": ("get_ids_from_query", "get_ids_from_queries", "batch_search"),
    "database_survey": ("get_ids_from_query", "get_ids_from_queries", "batch_search"),
    "GeneralRAG_langchain": ("retrieve_id", "retrieve_id4citation"),
}

_MAX_IDS = int(os.environ.get("RETRIEVAL_TAP_MAX_IDS", "2000"))
_MAX_QUERY_CHARS = int(os.environ.get("RETRIEVAL_TAP_MAX_QUERY_CHARS", "4000"))


def _fatal(why: str) -> None:
    try:
        sys.stderr.write("[retrieval_tap] FATAL: %s\n" % why)
        sys.stderr.flush()
    finally:
        os._exit(FATAL_EXIT)


def _emit(record: dict) -> None:
    if not _LOG:
        return
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _LOCK:
            d = os.path.dirname(_LOG)
            if d and not os.path.isdir(d):
                os.makedirs(d, exist_ok=True)
            with open(_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception as e:
        _fatal("cannot write %s (%s)" % (_LOG, e))


def _short_text(v):
    s = v if isinstance(v, str) else str(v)
    return s[:_MAX_QUERY_CHARS]


def _jsonable_ids(result):
    if result is None:
        return None, 0
    if isinstance(result, (str, bytes)):
        return [_short_text(result)], 1
    if isinstance(result, (list, tuple)):
        flat = []
        for item in result:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        out = [x if isinstance(x, (str, int, float, bool)) else _short_text(x) for x in flat[:_MAX_IDS]]
        return out, len(flat)
    return None, 0


def _describe_args(method, args, kwargs):
    info = {}
    pos = list(args)

    def take(name, idx):
        if name in kwargs:
            return kwargs[name]
        return pos[idx] if len(pos) > idx else None

    if method == "get_ids_from_query":
        info["query"] = _short_text(take("query", 0))
        info["num"] = take("num", 1)
    elif method in ("get_ids_from_queries", "retrieve_id", "retrieve_id4citation"):
        q = take("queries", 0) if method == "get_ids_from_queries" else take("query", 0)
        if isinstance(q, (list, tuple)):
            info["queries"] = [_short_text(x) for x in list(q)[:64]]
            info["n_queries"] = len(q)
        else:
            info["query"] = _short_text(q)
            info["n_queries"] = 1
        info["num"] = kwargs.get("num", kwargs.get("top_k", take("num", 1)))
    elif method == "batch_search":
        qv = take("query_vectors", 0)
        info["n_queries"] = len(qv) if qv is not None else None
        info["dim"] = int(getattr(qv, "shape", (0, 0))[1]) if getattr(qv, "ndim", 0) == 2 else None
        info["num"] = kwargs.get("top_k", take("top_k", 1))
        info["title"] = kwargs.get("title", take("title", 2))
    return info


def _wrap(cls_name, method_name, func):
    if getattr(func, "_retrieval_tap", False):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        depth = getattr(_DEPTH, "n", 0)
        _DEPTH.n = depth + 1
        t0 = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            _DEPTH.n = depth
        if _LOG:
            try:
                ids, n_ids = _jsonable_ids(result)
                rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)), "pid": os.getpid(),
                       "cls": cls_name, "method": method_name, "depth": depth,
                       "ms": round((time.time() - t0) * 1000.0, 1), "n_ids": n_ids, "ids": ids}
                rec.update(_describe_args(method_name, args[1:], kwargs))
            except Exception as e:
                _fatal("hook error in %s.%s (%s)" % (cls_name, method_name, e))
            _emit(rec)
        return result

    wrapper._retrieval_tap = True
    return wrapper


def _docstore_patch(mod) -> int:
    import pool_docstore

    return pool_docstore.patch_module(mod)


def _selector_patch(mod) -> int:
    import pool_selector

    return pool_selector.patch_module(mod) + pool_selector.patch_database_module(mod)


def _patch_module(mod) -> int:
    _docstore_patch(mod)
    _selector_patch(mod)
    patched = 0
    for cls_name, methods in _TARGET_METHODS.items():
        cls = getattr(mod, cls_name, None)
        if not isinstance(cls, type):
            continue
        for m in methods:
            func = getattr(cls, m, None)
            if callable(func) and not getattr(func, "_retrieval_tap", False):
                setattr(cls, m, _wrap(cls_name, m, func))
                patched += 1
    if patched:
        sys.stderr.write("[retrieval_tap] patched %d method(s) in %s -> %s\n" % (patched, mod.__name__, _LOG))
    return patched


class _TapFinder:


    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGET_MODULES:
            return None
        try:
            for finder in sys.meta_path:
                if finder is self:
                    continue
                fs = getattr(finder, "find_spec", None)
                if fs is None:
                    continue
                spec = fs(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _TapLoader(spec.loader)
                    return spec
        except Exception as e:
            _fatal("finder error for %s (%s)" % (fullname, e))
        return None


class _TapLoader:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _patch_module(module)
        except Exception as e:
            _fatal("patch error in %s (%s)" % (module.__name__, e))


def install() -> bool:
    if any(isinstance(f, _TapFinder) for f in sys.meta_path):
        return True
    sys.meta_path.insert(0, _TapFinder())
    for name in list(_TARGET_MODULES):
        mod = sys.modules.get(name)
        if mod is not None:
            _patch_module(mod)
    return True


try:
    install()
except Exception as e:
    _fatal("install failed (%s)" % e)
