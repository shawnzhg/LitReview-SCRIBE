#!/usr/bin/env python3
"""Append-only writer of a run's call log: one JSONL event per model call, retrieval call,
validation or failure, with running token, time and cost totals."""

from __future__ import annotations
import json
import time
from pathlib import Path

from common import sha256_str, utcnow

LABEL_SPACE_VERSION = "label_space/1.0"

WINDOWS = ("acquisition", "synthesis", "planning", "writing")
KINDS = ("llm", "retrieval_search", "retrieval_open", "tool", "validate",
         "internal_verifier", "failure", "retry", "budget_refusal")


class CallLog:
    def __init__(self, path, run_id: str, task_id: str, autoflush: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id, self.task_id = run_id, task_id
        self.seq = 0
        self.autoflush = autoflush
        self._buf: list = []
        self._fh = self.path.open("a")
        self.totals = {"in_tokens": 0, "out_tokens": 0, "wall_ms": 0, "usd": 0.0,
                       "gpu_seconds": 0.0, "n_events": 0,
                       "n_llm": 0, "n_search": 0, "n_open": 0, "n_failure": 0}

    def event(self, window: str, kind: str, started_at: str, resources: dict,
              host_class: str, branch_label: dict = None, **payload) -> dict:
        assert window in WINDOWS, f"bad window {window}"
        assert kind in KINDS, f"bad kind {kind}"
        rec = {"seq": self.seq, "run_id": self.run_id, "task_id": self.task_id,
               "label_space_version": LABEL_SPACE_VERSION,
               "window": window, "kind": kind,
               "started_at": started_at, "ended_at": utcnow(),
               "resources": resources, "host_class": host_class}
        if branch_label:
            rec["branch_label"] = branch_label
        rec.update({k: v for k, v in payload.items() if v is not None})
        self.seq += 1
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self.autoflush:
            self._fh.flush()
        t = self.totals
        t["n_events"] += 1
        for k in ("in_tokens", "out_tokens", "wall_ms"):
            t[k] += int(resources.get(k) or 0)
        t["usd"] += float(resources.get("usd") or 0.0)
        t["gpu_seconds"] += float(resources.get("gpu_seconds") or 0.0)
        t["n_llm"] += kind == "llm"
        t["n_search"] += kind == "retrieval_search"
        t["n_open"] += kind == "retrieval_open"
        t["n_failure"] += kind in ("failure", "budget_refusal")
        return rec

    def llm(self, window, started_at, wall_ms, model, prompt, completion, params,
            finish_reason=None, n_prompt_tokens=None, n_completion_tokens=None,
            model_revision=None, role_sequence=None, logprob_sum=None,
            branch_label=None, artifact_delta=None, gpu_seconds=None):
        return self.event(
            window, "llm", started_at,
            {"in_tokens": n_prompt_tokens or 0, "out_tokens": n_completion_tokens or 0,
             "wall_ms": wall_ms, "usd": 0.0, "gpu_seconds": gpu_seconds},
            host_class="gpu",
            branch_label=branch_label or {
                "stage": window, "action_class": "generate", "tool_class": "policy_llm",
                "resource_bin": _bin(n_completion_tokens or 0),
                "retry_or_failure": None if finish_reason == "stop" else (finish_reason or "unknown"),
                "observable_effect": _effect("llm", ok=finish_reason == "stop")},
            llm={"model": model, "model_revision": model_revision,
                 "role_sequence": role_sequence, "prompt": prompt,
                 "prompt_sha256": sha256_str(prompt), "completion": completion,
                 "finish_reason": finish_reason, "params": params,
                 "logprob_sum": logprob_sum,
                 "n_prompt_tokens": n_prompt_tokens,
                 "n_completion_tokens": n_completion_tokens},
            artifact_delta=artifact_delta)

    def retrieval_search(self, window, started_at, wall_ms, query, route, k_requested,
                         cutoff_year, snapshot_id, returned, post_cutoff_suppressed,
                         kept_by_agent=None, usd=0.0, artifact_delta=None, n_new=None):
        return self.event(
            window, "retrieval_search", started_at,
            {"in_tokens": 0, "out_tokens": 0, "wall_ms": wall_ms, "usd": usd},
            host_class="login",
            branch_label={"stage": window, "action_class": "search", "tool_class": "pool_search",
                          "resource_bin": _bin(len(returned)),
                          "retry_or_failure": None,
                          "observable_effect": _effect("search", len(returned), k=k_requested),
                          "novelty": _novelty(n_new if n_new is not None else len(returned),
                                              len(returned))},
            retrieval={"query": query, "route": route, "k_requested": k_requested,
                       "cutoff_year": cutoff_year, "snapshot_id": snapshot_id,
                       "returned": returned, "post_cutoff_suppressed": post_cutoff_suppressed,
                       "kept_by_agent": kept_by_agent},
            artifact_delta=artifact_delta)

    def retrieval_open(self, window, started_at, wall_ms, pmid, n_sentences, snapshot_id,
                       cutoff_year, refused=None, artifact_delta=None):
        return self.event(
            window, "retrieval_open", started_at,
            {"in_tokens": 0, "out_tokens": 0, "wall_ms": wall_ms, "usd": 0.0},
            host_class="login",
            branch_label={"stage": window, "action_class": "open_document",
                          "tool_class": "pool_fetch", "resource_bin": _bin(n_sentences),
                          "retry_or_failure": refused,
                          "observable_effect": _effect("open", n_sentences)},
            retrieval={"snapshot_id": snapshot_id, "cutoff_year": cutoff_year,
                       "opened": [str(pmid)]},
            artifact_delta=artifact_delta)

    def failure(self, window, started_at, ftype, message, traceback=None, recovered=False,
                kind="failure", host=None, wall_ms=0):
        return self.event(
            window, kind, started_at,
            {"in_tokens": 0, "out_tokens": 0, "wall_ms": wall_ms, "usd": 0.0},
            host_class=host or "login",
            branch_label={"stage": window, "action_class": kind, "tool_class": "none",
                          "resource_bin": "0", "retry_or_failure": ftype,
                          "observable_effect": "recovered" if recovered else "aborted"},
            failure={"type": ftype, "message": str(message)[:4000],
                     "traceback": traceback, "recovered": recovered})

    def validate_event(self, window, started_at, artifact, errors, host=None):
        return self.event(
            window, "validate", started_at,
            {"in_tokens": 0, "out_tokens": 0, "wall_ms": 0, "usd": 0.0},
            host_class=host or "login",
            branch_label={"stage": window, "action_class": "validate", "tool_class": "jsonschema",
                          "resource_bin": "0",
                          "retry_or_failure": None if not errors else "schema_invalid",
                          "observable_effect": _effect("validate", ok=not errors)},
            validation={"artifact": artifact, "n_errors": len(errors), "errors": errors[:20]})

    def close(self):
        self._fh.flush()
        self._fh.close()


def _novelty(n_new: int, n_returned: int) -> str:
    if n_returned <= 0 or n_new <= 0:
        return "novelty_none"
    return "novelty_some" if n_new * 2 <= n_returned else "novelty_most"


def _effect(kind: str, n: int = 0, ok: bool = True, k: int = None) -> str:
    if kind == "llm":
        return "completion" if ok else "truncated"
    if kind == "search":
        K = k or 50
        return ("results_none" if n == 0 else "results_few" if n <= 0.2 * K else
                "results_some" if n <= 0.8 * K else "results_many")
    if kind == "open":
        return ("sentences_none" if n == 0 else "sentences_few" if n <= 8 else "sentences_many")
    if kind == "validate":
        return "valid" if ok else "invalid"
    return "recovered" if ok else "aborted"


def _bin(n: int) -> str:
    n = int(n or 0)
    for hi, name in ((0, "0"), (16, "1-16"), (64, "17-64"), (256, "65-256"), (1024, "257-1024")):
        if n <= hi:
            return name
    return ">1024"


class Timer:
    def __enter__(self):
        self.started = utcnow()
        self._t0 = time.time()
        return self

    def __exit__(self, *a):
        self.ms = int((time.time() - self._t0) * 1000)
        return False
