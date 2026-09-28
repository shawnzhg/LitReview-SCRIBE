#!/usr/bin/env python3
"""Final acquisition ranking: keeps the single ranking call when the prompt fits the context and
otherwise ranks chunks and merges them by quota; writes ranking_record.json per unit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

VERSION = "ranker/1.0"
SCHEMA = "ranking_record/1.0"
RECORD = "ranking_record.json"
RULE = "single final call when it fits, else chunk tournament merged by quota"

TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
TEMPLATE_SHA256 = "178e41d35fcbd2412afb7d0cc83acb1a04fc39242c74afd639a8ee6bcbd95c00"

SINGLE_MAX_TOKENS = 900
RETRY_MULT, RETRY_CAP = 3, 12000
FINAL_SEED_OFFSET = 1
BATCH_SEED_OFFSET = 100
BATCH, PER_BATCH_MIN = 120, 20

CHUNK_SEED_OFFSET = 50000
TPI_HI = 13.0
JSON_OVERHEAD = 24
RETRY_EXTRA = 800
MARGIN = 256
CHUNK_MAX = 120
MAX_CHUNKS = 400

_STATE = threading.local()
_INSTALLED: dict = {}
_LOCK = threading.Lock()


class RankerConfigError(RuntimeError):
    pass


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def rank_listing(W, cands):
    return "\n".join(
        f"  {p['paper_id']} [{p.get('year')}] {W.clip(p.get('title') or '', 120, '...')}"
        for p in cands)


def rank_messages(W, spec, cands, K):
    return [{"role": "system", "content": W.ACQ_SYS},
            {"role": "user", "content": W.ACQ_RANK.format(
                question=spec["question"], cutoff=spec["publication_cutoff"], K=K,
                candidates=rank_listing(W, cands))}]


def flat_prompt(messages):
    return "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)


def messages_from_flat(flat):
    import re
    parts = re.split(r"(?:^|\n)<\|(system|user|assistant|tool)\|>\n", flat)
    msgs = []
    for i in range(1, len(parts) - 1, 2):
        msgs.append({"role": parts[i], "content": parts[i + 1]})
    return msgs


def parse_json(text):
    from llm import parse_json_block
    return parse_json_block(text)


class PromptCounter:

    def __init__(self, model_dir=None, check_pins=True):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        import tokenizers
        d = model_dir or os.environ.get("SCRIBE_RANKING_TOKENIZER")
        if not d:
            raise RankerConfigError("SCRIBE_RANKING_TOKENIZER is not set")
        d = Path(d)
        tj, ct = d / "tokenizer.json", d / "chat_template.jinja"
        if not tj.is_file() or not ct.is_file():
            raise RankerConfigError(f"{d}: tokenizer.json / chat_template.jinja missing")
        tb, cb = tj.read_bytes(), ct.read_bytes()
        self.tokenizer_sha256, self.template_sha256 = sha256_hex(tb), sha256_hex(cb)
        if check_pins and (self.tokenizer_sha256 != TOKENIZER_SHA256 or self.template_sha256 != TEMPLATE_SHA256):
            raise RankerConfigError(f"{d}: tokenizer/template sha256 {self.tokenizer_sha256[:16]}/{self.template_sha256[:16]} "
                                 f"!= pins {TOKENIZER_SHA256[:16]}/{TEMPLATE_SHA256[:16]}")
        self.model_dir = str(d)
        self.tok = tokenizers.Tokenizer.from_str(tb.decode("utf-8"))

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        self.template = env.from_string(cb.decode("utf-8"))

    def render(self, messages):
        return self.template.render(messages=messages, add_generation_prompt=True, bos_token=None,
                                    eos_token="<|im_end|>", pad_token="<|endoftext|>", unk_token=None)

    def count(self, messages):
        return len(self.tok.encode(self.render(messages), add_special_tokens=False).ids)


def ids_budget(n_ids):
    return int(math.ceil(TPI_HI * n_ids)) + JSON_OVERHEAD


def fits(P, max_tokens, maxlen):
    first = P + max_tokens + MARGIN <= maxlen
    retry = P + RETRY_EXTRA + min(RETRY_MULT * max_tokens, RETRY_CAP) + MARGIN <= maxlen
    return first, retry


def decide_single(P, K, maxlen):
    first, retry = fits(P, SINGLE_MAX_TOKENS, maxlen)
    ids_fit = ids_budget(K) <= SINGLE_MAX_TOKENS
    return {"prompt_tokens": P, "max_model_len": maxlen, "max_tokens": SINGLE_MAX_TOKENS, "fits_first": first,
            "fits_retry": retry, "ids_fit": ids_fit, "ids_budget_K": ids_budget(K),
            "mode": "single" if (first and retry and ids_fit) else "tournament"}


def even_sizes(N, n):
    return [N // n + (1 if i < N % n else 0) for i in range(n)]


def quota(K, size, N):
    return int(math.ceil(K * size / N))


def plan_chunks(W, spec, cands, K, counter, maxlen):
    N = len(cands)
    if N == 0:
        return []
    n0 = max(2, int(math.ceil(N / CHUNK_MAX)))
    for n in range(n0, min(N, MAX_CHUNKS) + 1):
        plan, start, ok = [], 0, True
        for i, sz in enumerate(even_sizes(N, n)):
            chunk = cands[start:start + sz]
            q = min(quota(K, sz, N), sz)
            mt = ids_budget(q)
            P = counter.count(rank_messages(W, spec, chunk, q))
            f1, f2 = fits(P, mt, maxlen)
            if not (f1 and f2):
                ok = False
                break
            plan.append({"index": i, "start": start, "end": start + sz, "n_candidates": sz, "q": q,
                         "max_tokens": mt, "prompt_tokens": P, "retry_max_tokens": min(RETRY_MULT * mt, RETRY_CAP)})
            start += sz
        if ok:
            return plan
    raise RankerConfigError(f"no chunking of {N} survivors fits max_model_len {maxlen} (K {K})")


def merge(lists, quotas, K):
    keyed = []
    for i, (ids, q) in enumerate(zip(lists, quotas)):
        for r, pid in enumerate(ids, 1):
            keyed.append((Fraction(r, q), i, r, pid))
    keyed.sort(key=lambda x: (x[0], x[1]))
    out, seen = [], set()
    for _, i, r, pid in keyed:
        if pid in seen:
            continue
        seen.add(pid)
        out.append((pid, i, r))
    return out[:K]


class _Rec:

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


def _cause(attempts, n_fab, n_foreign, err):
    errs = [x.get("error") for x in attempts if x.get("error")]
    if any("HTTP Error 400" in str(e) for e in errs):
        return "http_400"
    if errs:
        return "llm_error"
    if attempts and all(x.get("finish_reason") == "length" for x in attempts):
        return "completion_truncated"
    if n_fab or n_foreign:
        return "only_unknown_ids"
    return "unparseable_or_empty"


def batch_max_tokens(K_arg, n_cand):
    need = ids_budget(min(int(K_arg), int(n_cand)))
    return need if need > SINGLE_MAX_TOKENS else None


def rank_once_mt(W, client, log, spec, cands, K, seed, known, max_tokens):
    obj, draw, err = W.ask_json(client, log, "acquisition", rank_messages(W, spec, cands, K),
                                max_tokens=max_tokens, seed=seed)
    out, seen, fabricated = [], set(), []
    if obj:
        for pid in (obj.get("selected") or []):
            pid = str(pid).strip()
            if pid not in known:
                fabricated.append(pid)
            elif pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out, fabricated, (draw.completion if draw else ""), err


def chunk_call(W, client, log, spec, chunk, q, seed, max_tokens):
    attempts = []
    obj, draw, err = W.ask_json(_Rec(client, attempts), log, "acquisition", rank_messages(W, spec, chunk, q),
                                max_tokens=max_tokens, seed=seed)
    own = [p["paper_id"] for p in chunk]
    own_set = set(own)
    raw, got, seen, fab, foreign = [], [], set(), [], []
    if obj:
        for pid in (obj.get("selected") or []):
            pid = str(pid).strip()
            raw.append(pid)
            if pid not in own_set:
                fab.append(pid)
            elif pid not in seen:
                seen.add(pid)
                got.append(pid)
    return got, raw, fab, attempts, err


def tournament(W, client, log, spec, cands, K, unit_seed, known, counter, maxlen, rec):
    N = len(cands)
    plan = plan_chunks(W, spec, cands, K, counter, maxlen)
    lists, quotas, chunks_rec, all_fab = [], [], [], []
    for c in plan:
        chunk = cands[c["start"]:c["end"]]
        seed = unit_seed + CHUNK_SEED_OFFSET + c["index"]
        t0 = time.time()
        got, raw, fab, attempts, err = chunk_call(W, client, log, spec, chunk, c["q"], seed, c["max_tokens"])
        kept = got[:c["q"]]
        failed = not kept
        fb = [p["paper_id"] for p in chunk][:c["q"]] if failed else []
        lists.append(kept if not failed else fb)
        quotas.append(c["q"])
        windows_fab = [x for x in fab if x not in known]
        all_fab += windows_fab
        chunks_rec.append(dict(c, seed=seed, attempts=attempts, n_returned_raw=len(raw), n_own_unique=len(got),
                               n_kept=len(kept), n_over_quota=max(0, len(got) - c["q"]),
                               n_short=max(0, c["q"] - len(kept)) if not failed else 0,
                               n_fabricated=len(windows_fab), n_foreign=len(fab) - len(windows_fab),
                               failed=failed, fail_cause=(_cause(attempts, len(windows_fab), len(fab) - len(windows_fab), err)
                                                          if failed else None),
                               err=(str(err)[:300] if err else None), n_fallback_ids=len(fb),
                               kept=kept, wall_s=round(time.time() - t0, 3)))
    merged = merge(lists, quotas, K)
    out = [pid for pid, _, _ in merged]
    failed_idx = {c["index"] for c in chunks_rec if c["failed"]}
    n_fb = sum(1 for _, i, _ in merged if i in failed_idx)
    rec.update({"final_mode": "tournament", "n_chunks": len(plan), "chunks": chunks_rec,
                "n_chunks_failed": len(failed_idx), "chunk_seeds": [c["seed"] for c in chunks_rec],
                "merged": out, "merged_from_chunk": [i for _, i, _ in merged],
                "n_final": len(out), "n_from_fallback": n_fb,
                "share_from_fallback": (n_fb / len(out)) if out else None})
    return out, all_fab, "", (f"{len(failed_idx)} of {len(plan)} chunks failed" if failed_idx else None)


def _maxlen_from_env():
    v = os.environ.get("SCRIBE_RANKING_MAX_MODEL_LEN")
    if not v:
        raise RankerConfigError("SCRIBE_RANKING_MAX_MODEL_LEN is not set (the launcher exports MAXLEN)")
    ml = int(v)
    rv = os.environ.get("SCRIBE_RENDEZVOUS")
    if rv:
        try:
            r = json.loads(rv)
        except json.JSONDecodeError:
            r = {}
        if r.get("max_model_len") is not None and int(r["max_model_len"]) != ml:
            raise RankerConfigError(f"SCRIBE_RANKING_MAX_MODEL_LEN {ml} != the rendezvous max_model_len {r['max_model_len']}")
    return ml


def install(W=None, counter=None, maxlen=None):
    if _INSTALLED.get("rank"):
        return _INSTALLED["cfg"]
    if W is None:
        import windows as W
    if getattr(W._rank_once, "_cap_budget", False) or getattr(W.acquisition, "_cap_budget", False) or \
            "budget" in sys.modules and getattr(sys.modules["budget"], "_INSTALLED", {}).get("rank"):
        raise RankerConfigError("the cap-budget rank recorder is already installed: the ranker must be installed first (under it)")
    counter = counter or PromptCounter()
    maxlen = maxlen or _maxlen_from_env()
    orig_rank, orig_acq = W._rank_once, W.acquisition
    W._RANKER_ORIG_RANK, W._RANKER_ORIG_ACQ = orig_rank, orig_acq
    cfg = {"version": VERSION, "module": str(Path(__file__).resolve()),
           "module_md5": hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
           "tokenizer_dir": counter.model_dir, "tokenizer_sha256": counter.tokenizer_sha256,
           "template_sha256": counter.template_sha256, "max_model_len": maxlen,
           "constants": {"SINGLE_MAX_TOKENS": SINGLE_MAX_TOKENS, "RETRY_MULT": RETRY_MULT, "RETRY_CAP": RETRY_CAP,
                         "TPI_HI": TPI_HI, "JSON_OVERHEAD": JSON_OVERHEAD, "RETRY_EXTRA": RETRY_EXTRA,
                         "MARGIN": MARGIN, "CHUNK_MAX": CHUNK_MAX, "CHUNK_SEED_OFFSET": CHUNK_SEED_OFFSET,
                         "FINAL_SEED_OFFSET": FINAL_SEED_OFFSET, "BATCH_SEED_OFFSET": BATCH_SEED_OFFSET}}

    def _rank_once(client, log, spec, cands, K, seed, known):
        st = getattr(_STATE, "state", None)
        if st is None or seed != st["unit_seed"] + FINAL_SEED_OFFSET:
            off = None if st is None else seed - st["unit_seed"]
            if off is not None and BATCH_SEED_OFFSET <= off < CHUNK_SEED_OFFSET:
                mt = batch_max_tokens(K, len(cands))
                if mt is not None:
                    st["batch_resized"].append({"batch_index": off - BATCH_SEED_OFFSET, "seed": seed, "K_arg": K,
                                                "n_candidates": len(cands), "max_tokens": mt})
                    return rank_once_mt(W, client, log, spec, cands, K, seed, known, mt)
            if st is not None:
                st["n_pass_through"] += 1
            return orig_rank(client, log, spec, cands, K, seed, known)
        if st.get("final") is not None:
            raise RankerConfigError("a second final ranking call in one unit (windows makes at most one)")
        P = counter.count(rank_messages(W, spec, cands, K))
        dec = decide_single(P, K, maxlen)
        rec = {"called": True, "K": K, "n_candidates": len(cands), "seed": seed, "decision": dec}
        st["final"] = rec
        t0 = time.time()
        if dec["mode"] == "single":
            attempts = []
            out = orig_rank(_Rec(client, attempts), log, spec, cands, K, seed, known)
            rec.update({"final_mode": "single", "attempts": attempts, "n_selected": len(out[0]),
                        "selected": list(out[0]), "n_fabricated": len(out[1]),
                        "single_fallback": not out[0],
                        "single_fallback_cause": _cause(attempts, len(out[1]), 0, out[3]) if not out[0] else None,
                        "wall_s": round(time.time() - t0, 3)})
            return out
        out = tournament(W, client, log, spec, cands, K, st["unit_seed"], known, counter, maxlen, rec)
        rec["wall_s"] = round(time.time() - t0, 3)
        return out

    def acquisition(spec, tool, client, log, seed=0):
        st = {"unit_seed": seed, "final": None, "n_pass_through": 0, "batch_resized": []}
        unit_dir = Path(log.path).parent
        _STATE.state = st
        res, exc = None, None
        try:
            res = orig_acq(spec, tool, client, log, seed=seed)
            return res
        except BaseException as e:
            exc = e
            raise
        finally:
            _STATE.state = None
            try:
                rec = build_record(spec, st, res, exc, unit_dir, cfg)
            except Exception as e:
                rec = {"schema": SCHEMA, "task": str(spec.get("task_id")), "ok": False,
                       "problems": [f"record build failed: {type(e).__name__}: {e}"]}
            try:
                (unit_dir / RECORD).write_text(json.dumps(rec, indent=1))
            except OSError as e:
                if exc is None:
                    raise RankerConfigError(f"cannot write {unit_dir / RECORD}: {e}")

    _rank_once._ranker = True
    acquisition._ranker = True
    W._rank_once = _rank_once
    W.acquisition = acquisition
    _INSTALLED.update(rank=True, cfg=cfg)
    return cfg


def build_record(spec, st, res, exc, unit_dir, cfg):
    b = res[0] if isinstance(res, tuple) and res else None
    v = (b or {}).get("validation") or {}
    K = int(spec["budget"]["max_ranked_output_K"])
    bundle_ids = [str(p["paper_id"]) for p in (b or {}).get("papers") or []]
    fin = st.get("final")
    problems = []
    mode = (fin or {}).get("final_mode")
    whole_task_fallback = False
    n_fb = 0
    if fin is not None and mode == "single":
        whole_task_fallback = bool(fin.get("single_fallback"))
        if b is not None and not whole_task_fallback and bundle_ids != fin["selected"][:K]:
            problems.append("single final answered but the bundle is not its selection")
        n_fb = len(bundle_ids) if whole_task_fallback else 0
    if fin is not None and mode == "tournament":
        n_fb = fin.get("n_from_fallback") or 0
        if b is not None and bundle_ids != fin["merged"][:K]:
            problems.append("tournament ran but the bundle is not its merged order")
    ranking = v.get("ranking")
    tool_fb = ranking == "tool_score_fallback"
    if tool_fb:
        n_fb = len(bundle_ids)
    task_fallback = bool(whole_task_fallback or tool_fb or n_fb > 0)
    overflow = []
    for at in ((fin or {}).get("attempts") or []) + [a for c in ((fin or {}).get("chunks") or []) for a in c["attempts"]]:
        if "HTTP Error 400" in str(at.get("error") or ""):
            overflow.append(at.get("seed"))
    rec = {"schema": SCHEMA, "version": VERSION, "rule": RULE, "task": str(spec.get("task_id")),
           "unit_dir": str(unit_dir), "unit_seed": st["unit_seed"], "K": K,
           "acquisition_returned": b is not None,
           "acquisition_exception": (f"{type(exc).__name__}: {exc}"[:500] if exc is not None else None),
           "pool_size": v.get("pool_size"), "bundle_ranking": ranking, "n_bundle_papers": len(bundle_ids),
           "n_pass_through_rank_calls": st["n_pass_through"],
           "n_batch_resized": len(st["batch_resized"]), "batch_resized": st["batch_resized"],
           "final_called": fin is not None, "final_mode": mode,
           "final": fin,
           "n_chunks": (fin or {}).get("n_chunks") if mode == "tournament" else (1 if mode == "single" else 0),
           "n_chunks_failed": (fin or {}).get("n_chunks_failed", 0) if mode == "tournament" else 0,
           "whole_task_fallback": whole_task_fallback, "tool_score_fallback": tool_fb,
           "n_final_from_fallback": n_fb,
           "share_from_fallback": (n_fb / len(bundle_ids)) if bundle_ids else None,
           "task_fallback": task_fallback, "n_context_overflow_attempts": len(overflow),
           "config": cfg, "problems": problems, "ok": not problems,
           "definition": ("task_fallback = some of the final K came from a fallback: a tournament chunk that yielded no "
                          "id (its own survivor order), the single final call yielding no id (windows: selected = "
                          "finals or survivors), or every batch failing (tool_score_fallback). n_context_overflow_"
                          "attempts = final-stage attempts refused with HTTP 400.")}
    return rec
