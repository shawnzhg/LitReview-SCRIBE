#!/usr/bin/env python3
"""Routes the acquisition window to the pool service: installs the pool retrieval tool with a
per-task document cache, call log, provenance record and budget, and makes generation read texts
from that cache."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_CTX = threading.local()
_INSTALLED: dict = {}
_BUDGET_REC: dict = {}
PER_TASK_CACHE = "pool_doccache.jsonl"
PER_TASK_CALLS = "pool_calls.jsonl"
PER_TASK_PROV = "pool_provenance.json"
PER_TASK_BUDGET = "retrieval_budget.json"
BUDGET_LABEL = "cap"
RETRIEVAL_MODULES = ("metered_tool", "pool_retrieval_tool")


class BudgetConfigError(RuntimeError):
    pass


def budget_config(env=None):
    import hashlib
    env = os.environ if env is None else env
    path = env.get("SCRIBE_BUDGET_CAPS")
    if not path or not Path(path).is_file():
        raise BudgetConfigError(f"SCRIBE_BUDGET_CAPS={path!r} is not a file (the measured per-task caps)")
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    pin = env.get("SCRIBE_BUDGET_CAPS_SHA256")
    if pin and pin != sha:
        raise BudgetConfigError(f"{path}: sha256 {sha} != pinned SCRIBE_BUDGET_CAPS_SHA256 {pin}")
    doc = json.loads(raw)
    if doc.get("schema") != "budget_caps/1.0" or not isinstance(doc.get("caps"), dict):
        raise BudgetConfigError(f"{path}: not a budget_caps/1.0 file")
    return {"path": str(Path(path).resolve()), "sha256": sha, "caps": doc["caps"],
            "effective_file": doc.get("effective") or {}, "raw_sources_md5": doc.get("raw_sources_md5")}


def stamp_manifests(R, caps_sha256):
    if getattr(R.manifest, "_budget_stamped", False):
        return
    orig = R.manifest

    def manifest(*a, **k):
        extra = dict(k.get("extra") or {})
        extra.setdefault("retrieval_budget", BUDGET_LABEL)
        extra.setdefault("budget_caps_sha256", caps_sha256)
        extra.setdefault("retrieval_backend", "pool (frozen static pool, per-task cutoffs)")
        k["extra"] = extra
        return orig(*a, **k)
    manifest._budget_stamped = True
    R.manifest = manifest


def set_unit_dir(d):
    _CTX.run_dir = Path(d) if d is not None else None


def unit_dir():
    return getattr(_CTX, "run_dir", None)


def install_acquisition(budget_for):
    if _INSTALLED.get("acq"):
        return _INSTALLED["acq"]
    if "metered_tool" in sys.modules:
        raise RuntimeError("metered_tool was imported before the pool backend was installed; "
                           "install_acquisition() must run first in the process")
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import pool_retrieval_tool as PRT
    try:
        bcfg = budget_config()
    except BudgetConfigError as e:
        raise PRT.PoolContractViolation(f"retrieval budget: {e}")
    import metered_tool as MT
    from doccache import DocCache
    import runner as R
    if MT.BudgetExhausted is not PRT.BudgetExhausted:
        raise RuntimeError("metered_tool did not bind the pool backend's BudgetExhausted")
    if R.SNAPSHOT != PRT.POOL_SNAPSHOT_ID:
        raise PRT.PoolContractViolation(f"runner.SNAPSHOT {R.SNAPSHOT!r} != the pool's {PRT.POOL_SNAPSHOT_ID!r}")

    class MeteredPoolRetrieval(MT.MeteredRetrieval):

        def __init__(self, snapshot_id, cutoff_year, budget, calllog, exclude_ids=None):
            task = str(calllog.task_id)
            run_dir = Path(calllog.path).parent
            if snapshot_id != PRT.POOL_SNAPSHOT_ID:
                raise PRT.PoolContractViolation(f"snapshot {snapshot_id!r} is not the pool's")
            cache = run_dir / PER_TASK_CACHE
            if cache.exists() and not (run_dir / "evidence_bundle.json").exists():
                raise PRT.PoolContractViolation(f"{cache} already exists without an evidence bundle: a partial "
                                                 f"earlier attempt; re-run at a FRESH level (no cache reuse)")
            if cache.exists():
                raise PRT.PoolContractViolation(f"{cache} exists: this unit was already acquired (run_campaign "
                                                 f"--redo on a pool level is refused; use a fresh level)")
            self._pool_task, self._pool_dir = task, run_dir
            brec = _BUDGET_REC.get(task)
            if brec is None or {k: budget.get(k) for k in brec["effective"]} != brec["effective"]:
                raise PRT.PoolContractViolation(f"{task}: the acquisition budget {budget} is not the effective pool "
                                                 f"budget (runner.load_spec was not wrapped by install_acquisition)")
            prov = PRT.task_start_provenance(task)
            (run_dir / PER_TASK_PROV).write_text(json.dumps(prov, indent=1))
            super().__init__(snapshot_id, cutoff_year, budget, calllog, DocCache(PRT.POOL_SNAPSHOT_ID, path=cache),
                             exclude_ids=exclude_ids)
            self.tool.exclude_ids = set(self.exclude_ids)
            (run_dir / PER_TASK_BUDGET).write_text(json.dumps(dict(brec, exclude_ids=sorted(self.exclude_ids)),
                                                              indent=1))

        def _build_tool(self, snapshot_id, cutoff_year, budget, trace_path):
            return PRT.PoolRetrievalTool(snapshot_id, cutoff_year, budget, trace_path=trace_path,
                                         task_id=self._pool_task, calls_path=self._pool_dir / PER_TASK_CALLS)

    orig_load_spec = R.load_spec

    def load_spec_budgeted(task_id, split=None):
        spec = orig_load_spec(task_id, split)
        try:
            eff, rec = budget_for(str(task_id), spec["budget"], bcfg)
        except BudgetConfigError as e:
            raise PRT.PoolContractViolation(f"retrieval budget: {e}")
        rec["spec_content_hash"] = spec.get("content_hash")
        _BUDGET_REC[str(task_id)] = rec
        spec = dict(spec, budget=eff, retrieval_budget=rec)
        return spec

    MT.MeteredRetrieval = MeteredPoolRetrieval
    R.load_spec = load_spec_budgeted
    stamp_manifests(R, bcfg["sha256"])
    _INSTALLED["acq"] = PRT
    _INSTALLED["MeteredPoolRetrieval"] = MeteredPoolRetrieval
    _INSTALLED["budget_config"] = bcfg
    return PRT


def pool_texts_from_cache(bundle):
    from doccache import DocCache
    import runner as R
    d = unit_dir()
    if d is None:
        raise RuntimeError("pool_texts_from_cache: no unit dir bound to this thread (run_campaign_pool.py binds it)")
    if str(bundle.get("task_id")) != d.parent.name:
        raise RuntimeError(f"bundle task {bundle.get('task_id')} is not this unit's task {d.parent.name}")
    cache = d / PER_TASK_CACHE
    if not cache.is_file():
        raise RuntimeError(f"{cache} missing: this unit was not acquired by the pool backend")
    dc = DocCache(R.SNAPSHOT, path=cache)
    out, missing = {}, []
    for p in bundle["papers"]:
        rec = dc.get(p["paper_id"])
        if not rec:
            missing.append(p["paper_id"])
            continue
        out[p["paper_id"]] = {
            "abstract": rec.get("abstract") or "",
            "sentences": {s["locator"]: s["text"] for s in (rec.get("sentences") or [])}}
    if missing:
        raise RuntimeError(f"{len(missing)} bundle papers missing from {cache}: {missing[:5]}")
    return out


def install_generation():
    if _INSTALLED.get("gen"):
        return
    for m in RETRIEVAL_MODULES:
        if m in sys.modules:
            raise RuntimeError(f"{m} is loaded in the generation process: the late windows must hold no route "
                               f"to any corpus")
    import runner as R
    R.texts_from_cache = pool_texts_from_cache
    stamp_manifests(R, os.environ.get("SCRIBE_BUDGET_CAPS_SHA256"))
    _INSTALLED["gen"] = True


def assert_generation_clean():
    return [m for m in RETRIEVAL_MODULES if m in sys.modules]
