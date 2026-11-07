#!/usr/bin/env python3
"""Model client of the harness: sends greedy chat requests (temperature 0) round-robin to the vLLM
servers named in the rendezvous, returns each draw with its prompt, completion and parameters, and
parses JSON blocks from replies."""

from __future__ import annotations
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import itertools
import threading

TEMPERATURE = 0.0
TOP_P = 1.0


@dataclass
class Draw:
    prompt: str
    completion: str
    finish_reason: str
    params: dict
    n_prompt_tokens: int = 0
    n_completion_tokens: int = 0
    wall_ms: int = 0
    model: str = ""
    role_sequence: list = field(default_factory=list)
    error: str = None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class VLLMClient:

    def __init__(self, url=None, model=None, timeout=900, retries=3, rendezvous=None):
        if rendezvous is not None:
            rvs = [rendezvous]
        elif url:
            rvs = [{"url": u.strip(), "served_model": model or "qwen38"}
                   for u in url.split(",") if u.strip()]
        else:
            raise ValueError("VLLMClient needs a rendezvous (SCRIBE_RENDEZVOUS) or a url")
        self.endpoints = [{"url": r["url"].rstrip("/"),
                           "model": model or r.get("served_model", "qwen38")} for r in rvs]
        self._rr = itertools.cycle(range(len(self.endpoints)))
        self._rr_lock = threading.Lock()
        rv = rvs[0]
        self.url = self.endpoints[0]["url"]
        self.model = self.endpoints[0]["model"]
        self.model_path = rv.get("model_path")
        self.max_model_len = rv.get("max_model_len")
        self.timeout = timeout
        self.retries = max(1, retries)
        self.fingerprints = {}
        self._rv_raw = rvs

    def describe(self) -> dict:
        import hashlib
        cfgs = []
        for r in self._rv_raw:
            blob = json.dumps(r, sort_keys=True)
            cfgs.append({"url": r.get("url"), "host": r.get("host"), "port": r.get("port"),
                         "served_model": r.get("served_model"),
                         "model_path": r.get("model_path"),
                         "max_model_len": r.get("max_model_len"),
                         "job_id": r.get("job_id"),
                         "rendezvous_sha256": hashlib.sha256(blob.encode()).hexdigest()[:16],
                         "vllm_fingerprint": self.fingerprints.get((r.get("url") or "").rstrip("/"))})
        return {"backend": "vllm-openai", "endpoints": cfgs, "prefix_caching": False}

    def _next(self):
        with self._rr_lock:
            return self.endpoints[next(self._rr)]

    def chat(self, messages, max_tokens=2048, seed=None, stop=None) -> Draw:
        params = {"temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": max_tokens,
                  "seed": seed, "stop": stop, "backend": "vllm", "dtype": "bfloat16",
                  "served_model": self.model, "model_path": self.model_path}
        ep = self._next()
        params["endpoint"] = ep["url"]
        body = {"model": ep["model"], "messages": messages, "temperature": TEMPERATURE,
                "top_p": TOP_P, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        if stop:
            body["stop"] = stop
        flat = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)

        t0 = time.time()
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    f"{ep['url']}/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read())
                ch = d["choices"][0]
                u = d.get("usage") or {}
                fp = d.get("system_fingerprint")
                if fp:
                    self.fingerprints[ep["url"]] = fp
                return Draw(prompt=flat, completion=ch["message"].get("content") or "",
                            finish_reason=ch.get("finish_reason"), params=params,
                            n_prompt_tokens=u.get("prompt_tokens", 0),
                            n_completion_tokens=u.get("completion_tokens", 0),
                            wall_ms=int((time.time() - t0) * 1000),
                            model=self.model_path or self.model,
                            role_sequence=[m["role"] for m in messages])
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
                    KeyError, json.JSONDecodeError) as e:
                last = f"{type(e).__name__}: {e}"
                if attempt < self.retries - 1:
                    time.sleep(2 ** attempt)
        return Draw(prompt=flat, completion="", finish_reason="error", params=params,
                    wall_ms=int((time.time() - t0) * 1000),
                    model=self.model_path or self.model,
                    role_sequence=[m["role"] for m in messages], error=last)

    def health(self) -> bool:
        alive = []
        for ep in self.endpoints:
            try:
                base = ep["url"].rsplit("/v1", 1)[0]
                with urllib.request.urlopen(f"{base}/health", timeout=15) as r:
                    if r.status == 200:
                        alive.append(ep)
            except Exception:
                pass
        if alive and len(alive) != len(self.endpoints):
            print(f"[llm] dropping {len(self.endpoints) - len(alive)} unhealthy endpoint(s)")
            self.endpoints = alive
            self._rr = itertools.cycle(range(len(alive)))
        return bool(alive)


def parse_json_block(text: str):
    if not text:
        return None, "empty completion"
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                s = p
                break
    i = min([x for x in (s.find("{"), s.find("[")) if x >= 0], default=-1)
    if i < 0:
        return None, "no JSON object found in completion"
    opener = s[i]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[i:j + 1]), None
                except json.JSONDecodeError as e:
                    return None, f"JSONDecodeError: {e}"
    return None, "unterminated JSON (completion likely truncated -- check finish_reason)"


def salvage_truncated_arrays(text: str, keys):
    out = {}
    for key in keys:
        i = text.find(f'"{key}"')
        if i < 0:
            continue
        j = text.find("[", i)
        if j < 0:
            continue
        items, depth, start, in_str, esc = [], 0, None, False, False
        for k in range(j + 1, len(text)):
            c = text[k]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                if depth == 0:
                    start = k
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        items.append(json.loads(text[start:k + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
            elif c == "]" and depth == 0:
                break
        if items:
            out[key] = items
    return out
