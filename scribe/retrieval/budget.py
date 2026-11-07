#!/usr/bin/env python3
"""Budget extension of the pool backend: sets each task's effective budget from the K_cap and caps
files and records whether the ranking's batch-order fallback fired."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNERS = HERE.parent / "harness" / "runners"
for _p in (str(RUNNERS), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pool_backend as PB

KCAP_SCHEMA = "kcap/1.0"
RECORD = "ranking_fallback_record.json"
RECORD_SCHEMA = "ranking_fallback_record/1.0"
CAP_RULE = "K_cap"
RANK_BATCH = 120
EFF_KEYS = ("max_search_calls", "max_ranked_output_K", "max_document_opens", "max_results_per_call",
            "max_docs_read")
ROWS_RULE = "min(CAP.rpc, 1000)"
KCAP_PINNED_KEYS = tuple(k for k in EFF_KEYS if k != "max_results_per_call")
_INSTALLED: dict = {}
_RANK = threading.local()


class KcapConfigError(RuntimeError):
    pass


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_kcap(env=None) -> dict:
    env = os.environ if env is None else env
    path, pin = env.get("SCRIBE_KCAP"), env.get("SCRIBE_KCAP_SHA256")
    if not path or not Path(path).is_file():
        raise KcapConfigError(f"SCRIBE_KCAP={path!r} is not a file (the per-task K_cap file)")
    if not pin:
        raise KcapConfigError("SCRIBE_KCAP_SHA256 is not set: the K_cap file must be pinned")
    raw = Path(path).read_bytes()
    sha = sha256_hex(raw)
    if sha != pin:
        raise KcapConfigError(f"{path}: sha256 {sha} != pinned SCRIBE_KCAP_SHA256 {pin}")
    doc = json.loads(raw)
    if doc.get("schema") != KCAP_SCHEMA or not isinstance(doc.get("tasks"), dict):
        raise KcapConfigError(f"{path}: not a {KCAP_SCHEMA} file")
    return {"path": str(Path(path).resolve()), "sha256": sha, "doc": doc, "tasks": doc["tasks"],
            "caps_sha256": (doc.get("inputs") or {}).get("caps_sha256"),
            "service_k_max": int(doc.get("service_k_max", 1000)), "native_opens": int(doc.get("native_opens", 60))}


def kcap_effective(task_id, spec_budget, cfg, kc):
    t = str(task_id)
    ent = kc["tasks"].get(t)
    if not ent:
        raise KcapConfigError(f"{t}: no K_cap in {kc['path']}")
    if kc["caps_sha256"] != cfg["sha256"]:
        raise KcapConfigError(f"the K_cap file was built from caps sha256 {kc['caps_sha256']}, the run loads "
                              f"{cfg['sha256']} ({cfg['path']})")
    c = cfg["caps"].get(t)
    if not c or any(not isinstance(c.get(q), dict) for q in ("search_calls", "docs_read")):
        raise KcapConfigError(f"{t}: no measured cap in {cfg['path']}")
    K_cap = int(ent["K_cap"])
    if not 1 <= K_cap <= kc["service_k_max"]:
        raise KcapConfigError(f"{t}: K_cap {K_cap} outside 1..{kc['service_k_max']}")
    C_calls, C_docs = int(c["search_calls"]["value"]), int(c["docs_read"]["value"])
    if min(C_calls, C_docs) < 1:
        raise KcapConfigError(f"{t}: a measured cap below 1: {c}")
    rpc = c.get("results_per_call_max")
    if not isinstance(rpc, dict) or int(rpc.get("value") or 0) < 1:
        raise KcapConfigError(f"{t}: no measured rows-per-call cap (results_per_call_max) in {cfg['path']}")
    C_rpc = int(rpc["value"])
    R_call = min(C_rpc, kc["service_k_max"])
    O0 = int(spec_budget["max_document_opens"])
    eff = dict(spec_budget, max_search_calls=C_calls, max_ranked_output_K=K_cap,
               max_document_opens=max(O0, K_cap), max_results_per_call=R_call,
               max_docs_read=C_docs)
    want = ent.get("effective") or {}
    if any(int(want.get(k, -1)) != int(eff[k]) for k in KCAP_PINNED_KEYS):
        raise KcapConfigError(f"{t}: effective budget {({k: eff[k] for k in KCAP_PINNED_KEYS})} != the K_cap file's "
                              f"{({k: want.get(k) for k in KCAP_PINNED_KEYS})} (spec budget or caps changed since the build?)")
    want_rpc = ((cfg.get("effective_file") or {}).get("cap") or {}).get(t)
    if want_rpc is not None and int(want_rpc.get("max_results_per_call", -1)) != R_call:
        raise KcapConfigError(f"{t}: rows per call {R_call} = min(CAP.rpc {C_rpc}, {kc['service_k_max']}) != the caps "
                              f"file's 'cap' effective {want_rpc.get('max_results_per_call')}")
    part = {"cap_rule": CAP_RULE, "rows_rule": ROWS_RULE, "kcap_file": kc["path"], "kcap_sha256": kc["sha256"],
            "K_cap": K_cap, "K_cap_arm": ent.get("arm"), "K_cap_P_by_arm": ent.get("P_by_arm"),
            "caps": {"search_calls": c["search_calls"], "docs_read": c["docs_read"],
                     "results_per_call": {"value": R_call, "measured": C_rpc, "arm": rpc.get("arm"),
                                          "rule": "min(CAP.rpc, 1000)"},
                     "K": {"value": K_cap, "arm": ent.get("arm"), "rule": "max baseline |P|, <= 1000"}}}
    return eff, part


def budget_for_kcap_factory(kc):
    def budget_for_kcap(task_id, spec_budget, cfg):
        try:
            eff, part = kcap_effective(task_id, spec_budget, cfg, kc)
        except KcapConfigError as e:
            raise PB.BudgetConfigError(str(e))
        rec = {"schema": "retrieval_budget/1.0", "task": str(task_id), "caps_file": cfg["path"],
               "caps_sha256": cfg["sha256"], "caps_raw_sources_md5": cfg["raw_sources_md5"],
               "spec_budget": dict(spec_budget), "effective": eff, **part}
        return eff, rec
    return budget_for_kcap


def stamp_kcap(R, kc):
    if getattr(R.manifest, "_cap_budget_stamped", False):
        return
    orig = R.manifest

    def manifest(*a, **k):
        extra = dict(k.get("extra") or {})
        extra.setdefault("cap_rule", CAP_RULE)
        extra.setdefault("cap_rows_rule", ROWS_RULE)
        extra.setdefault("kcap_sha256", kc["sha256"])
        k["extra"] = extra
        return orig(*a, **k)
    manifest._cap_budget_stamped = True
    manifest._budget_stamped = True
    R.manifest = manifest


class _RecordingClient:

    def __init__(self, inner, sink):
        self._inner = inner
        self._sink = sink

    def chat(self, messages, *a, **k):
        t0 = time.time()
        d = self._inner.chat(messages, *a, **k)
        prompt = getattr(d, "prompt", "") or ""
        self._sink.append({
            "finish_reason": getattr(d, "finish_reason", None), "error": getattr(d, "error", None),
            "n_prompt_tokens": getattr(d, "n_prompt_tokens", None),
            "n_completion_tokens": getattr(d, "n_completion_tokens", None),
            "max_tokens": k.get("max_tokens"), "seed": k.get("seed"),
            "prompt_chars": len(prompt), "prompt_sha256": "sha256:" + sha256_hex(prompt.encode("utf-8")),
            "completion_chars": len(getattr(d, "completion", "") or ""), "wall_s": round(time.time() - t0, 3)})
        return d

    def __getattr__(self, name):
        return getattr(self._inner, name)


def classify_final(final):
    at = final.get("attempts") or []
    errs = [x.get("error") for x in at if x.get("error")]
    if any("HTTP Error 400" in str(e) for e in errs):
        return "http_400"
    if errs:
        return "llm_error"
    if at and all(x.get("finish_reason") == "length" for x in at):
        return "completion_truncated"
    if final.get("n_fabricated"):
        return "only_unknown_ids"
    return "unparseable_or_empty"


def build_record(spec, st, res, exc, unit_dir):
    K = int(spec["budget"]["max_ranked_output_K"])
    b = res[0] if isinstance(res, tuple) and res else None
    v = (b or {}).get("validation") or {}
    calls = st["calls"]
    batches = [c for c in calls if c["role"] == "batch"]
    finals = [c for c in calls if c["role"] == "final"]
    survivors = [pid for c in batches for pid in c["selected"]]
    pool_size = v.get("pool_size")
    problems = []
    nb = math.ceil(pool_size / RANK_BATCH) if pool_size else 0
    if b is not None:
        if len(batches) != nb:
            problems.append(f"{len(batches)} batch ranking calls recorded, the pool of {pool_size} needs {nb}")
        if [c["batch_index"] for c in batches] != list(range(len(batches))):
            problems.append(f"batch indices {[c['batch_index'] for c in batches][:10]} are not 0..n-1 in order")
        if len(finals) > 1:
            problems.append(f"{len(finals)} final ranking calls recorded (the agent makes at most one)")
        if finals and calls[-1]["role"] != "final":
            problems.append("a batch ranking call was recorded after the final one")
        need_final = len(survivors) > K and nb > 1
        if need_final != bool(finals):
            problems.append(f"final call {'missing' if need_final else 'unexpected'}: survivors {len(survivors)}, "
                            f"K {K}, batches {nb}")
    final = finals[0] if finals else None
    fallback = bool(final is not None and final["n_selected"] == 0)
    bundle_ids = [str(p["paper_id"]) for p in (b or {}).get("papers") or []]
    order_ok = None
    if b is not None and fallback:
        order_ok = bundle_ids == survivors[:K]
        if not order_ok:
            problems.append("fallback recorded but the bundle is not the survivors in batch order")
    if b is not None and final is not None and not fallback:
        if bundle_ids != final["selected"][:K]:
            problems.append("final ranking answered but the bundle is not its selection")
    ranking = v.get("ranking")
    rec = {"schema": RECORD_SCHEMA, "task": str(spec.get("task_id")), "unit_dir": str(unit_dir),
           "unit_seed": st["unit_seed"], "K": K,
           "acquisition_returned": b is not None, "acquisition_exception": (f"{type(exc).__name__}: {exc}"[:500]
                                                                            if exc is not None else None),
           "pool_size": pool_size, "n_batches": len(batches), "n_batches_expected": nb,
           "n_batch_calls_empty": sum(1 for c in batches if c["n_selected"] == 0),
           "n_survivors": len(survivors), "final_called": final is not None,
           "final_n_candidates": final["n_candidates"] if final else None,
           "final_n_selected": final["n_selected"] if final else None,
           "fallback_fired": fallback, "fallback_cause": classify_final(final) if fallback else None,
           "batch_order_confirmed": order_ok, "bundle_ranking": ranking,
           "tool_score_fallback": ranking == "tool_score_fallback", "n_bundle_papers": len(bundle_ids),
           "calls": [{k: (x if k != "selected" else len(x)) for k, x in c.items()} for c in calls],
           "survivors": survivors, "final_selected": final["selected"] if final else None,
           "problems": problems, "ok": not problems,
           "definition": ("fallback_fired = the final ranking call over the batch survivors was made (survivors > K, "
                          "> 1 batch) and yielded no id, so the bundle is the survivors in batch order (windows."
                          "acquisition: selected = finals or survivors); causes: http_400 (prompt + max_tokens over "
                          "max-model-len), completion_truncated (every attempt cut at max_tokens), llm_error, "
                          "only_unknown_ids, unparseable_or_empty. tool_score_fallback = every batch failed.")}
    return rec


def install_rank_recorder():
    if _INSTALLED.get("rank"):
        return
    import windows as W
    orig_acq, orig_rank = W.acquisition, W._rank_once

    def _rank_once(client, log, spec, cands, K, seed, known):
        st = getattr(_RANK, "state", None)
        if st is None:
            return orig_rank(client, log, spec, cands, K, seed, known)
        attempts = []
        t0 = time.time()
        out, fab, comp, err = orig_rank(_RecordingClient(client, attempts), log, spec, cands, K, seed, known)
        us = st["unit_seed"]
        role = "final" if seed == us + 1 else "batch"
        st["calls"].append({"role": role, "batch_index": (seed - us - 100) if role == "batch" else None,
                            "seed": seed, "K_arg": K, "n_candidates": len(cands), "n_known": len(known),
                            "n_selected": len(out), "selected": list(out), "n_fabricated": len(fab),
                            "err": err, "attempts": attempts, "wall_s": round(time.time() - t0, 3)})
        return out, fab, comp, err

    def acquisition(spec, tool, client, log, seed=0):
        st = {"unit_seed": seed, "calls": []}
        unit_dir = Path(log.path).parent
        _RANK.state = st
        res, exc = None, None
        try:
            res = orig_acq(spec, tool, client, log, seed=seed)
            return res
        except BaseException as e:
            exc = e
            raise
        finally:
            _RANK.state = None
            try:
                rec = build_record(spec, st, res, exc, unit_dir)
            except Exception as e:
                rec = {"schema": RECORD_SCHEMA, "task": str(spec.get("task_id")), "ok": False,
                       "problems": [f"record build failed: {type(e).__name__}: {e}"], "fallback_fired": None}
            try:
                (unit_dir / RECORD).write_text(json.dumps(rec, indent=1))
            except OSError as e:
                if exc is None:
                    import pool_retrieval_tool as PRT
                    raise PRT.PoolContractViolation(f"cannot write {unit_dir / RECORD}: {e}")

    W._rank_once = _rank_once
    W.acquisition = acquisition
    _INSTALLED["rank"] = True


def install_acquisition_kcap(kc):
    if _INSTALLED.get("acq"):
        return _INSTALLED["acq"]
    PRT = PB.install_acquisition(budget_for_kcap_factory(kc))
    import runner as R
    install_rank_recorder()
    stamp_kcap(R, kc)
    _INSTALLED["acq"] = PRT
    return PRT


def install_generation_kcap(kc):
    PB.install_generation()
    import runner as R
    stamp_kcap(R, kc)
    _INSTALLED["gen"] = True
