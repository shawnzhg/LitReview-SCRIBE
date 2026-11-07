#!/usr/bin/env python3
"""Model client of SCRIBE-Luna: sends every SCRIBE call to gpt-5.6-luna through the recording proxy
and labels each call and window manifest with that backend."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNERS = REPO / "scribe" / "harness" / "runners"
BACKEND = "openai-api"
HARNESS_MODULES = ("run_campaign", "runner", "windows", "llm", "common", "calllog", "integrity", "doccache",
                   "run_campaign_pool", "run_budgeted_campaign", "pool_backend", "budget",
                   "metered_tool", "query_agent", "selection_rule", "pool_retrieval_tool", "ranker", "ranker_audit",
                   "writing_levers")


def load_llm():
    if str(RUNNERS) not in sys.path:
        sys.path.insert(0, str(RUNNERS))
    import llm
    if Path(llm.__file__).resolve().parent != RUNNERS:
        raise SystemExit(f"luna: llm was imported from {llm.__file__}, not {RUNNERS}")
    return llm


def foreign_modules(modules=None, repo=REPO):
    modules = sys.modules if modules is None else modules
    out = {}
    for name in HARNESS_MODULES:
        f = getattr(modules.get(name), "__file__", None)
        if f and Path(repo).resolve() not in Path(f).resolve().parents:
            out[name] = f
    return out


def rendezvous_problem(rv):
    if not isinstance(rv, dict) or rv.get("backend") != BACKEND:
        return "the rendezvous must describe the recording proxy (backend openai-api)"
    for k in ("url", "served_model", "service_tier", "reasoning_effort"):
        if not rv.get(k):
            return f"the rendezvous has no {k}"
    if not str(rv["url"]).rstrip("/").endswith("/v1"):
        return f"the rendezvous url {rv['url']} does not end in /v1"
    return None


def proxy_health(url, timeout=15):
    base = url.rstrip("/").rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(f"{base}/__proxy_health", timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def make_client_class(base):

    class LunaClient(base):
        _luna = True

        def __init__(self, url=None, model=None, timeout=None, retries=3, rendezvous=None):
            why = rendezvous_problem(rendezvous)
            if why:
                raise SystemExit(f"luna: {why} (set SCRIBE_RENDEZVOUS)")
            bad = foreign_modules()
            if bad:
                raise SystemExit(f"luna: harness modules loaded from outside {REPO}: {bad}")
            super().__init__(timeout=float(timeout or rendezvous.get("client_timeout_s") or 1800),
                             retries=retries, rendezvous=rendezvous)
            self.rendezvous = dict(rendezvous)
            self.service_tier = rendezvous["service_tier"]
            self.reasoning_effort = rendezvous["reasoning_effort"]

        def describe(self):
            blob = json.dumps(self.rendezvous, sort_keys=True)
            return {"backend": BACKEND, "url": self.url, "served_model": self.model,
                    "service_tier": self.service_tier, "reasoning_effort": self.reasoning_effort,
                    "max_model_len": self.rendezvous.get("max_model_len"),
                    "rendezvous_sha256": hashlib.sha256(blob.encode()).hexdigest()[:16]}

        def chat(self, *a, **k):
            d = super().chat(*a, **k)
            d.params.update(backend=BACKEND, dtype=None, model_path=None, service_tier=self.service_tier,
                            reasoning_effort=self.reasoning_effort)
            d.model = self.model
            return d

        def health(self):
            h = proxy_health(self.url)
            return bool(h) and h.get("mode") == "api" and h.get("force_model") == self.model \
                and h.get("service_tier") == self.service_tier and h.get("reasoning_effort") == self.reasoning_effort

    return LunaClient


def install(llm=None):
    llm = llm or load_llm()
    if getattr(llm.VLLMClient, "_luna", False):
        return llm.VLLMClient
    cls = make_client_class(llm.VLLMClient)
    llm.VLLMClient = cls
    return cls
