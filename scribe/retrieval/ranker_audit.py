#!/usr/bin/env python3
"""Recomputes a unit's final acquisition ranking from trace.jsonl and checks it against the ranking
records and the bundle; used by ranking_budget_audit.py and selection_audit.py."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ranker as CR

RANK_MARK = "You retrieved these candidate papers"
LINE = re.compile(r"^  (\S+) \[.*$", re.M)
ASK = re.compile(r"Select at most (\d+)\. Do not invent")


def _runners():
    p = str(HERE.parent / "harness" / "runners")
    if p not in sys.path:
        sys.path.insert(0, p)


def jl(p):
    p = Path(p)
    if not p.exists():
        return None
    return [json.loads(l) for l in p.read_text(encoding="utf-8").split("\n") if l.strip()]


def rank_calls(rows):
    calls = {}
    order = []
    for i, e in enumerate(rows):
        if e.get("kind") != "llm" or e.get("window") != "acquisition":
            continue
        L = e.get("llm") or {}
        prompt = L.get("prompt") or ""
        if RANK_MARK not in prompt:
            continue
        err = None
        if L.get("finish_reason") == "error":
            for f in rows[i + 1:i + 4]:
                if f.get("kind") in ("failure", "retry") and (f.get("failure") or {}).get("type") == "LLMTransportError":
                    err = (f.get("failure") or {}).get("message")
                    break
            err = err or "error (no LLMTransportError event)"
        p = L.get("params") or {}
        s = p.get("seed")
        if s not in calls:
            order.append(s)
        calls.setdefault(s, []).append({
            "prompt": prompt, "completion": L.get("completion") or "", "finish_reason": L.get("finish_reason"),
            "max_tokens": p.get("max_tokens"), "temperature": p.get("temperature"), "seed": s,
            "n_prompt_tokens": L.get("n_prompt_tokens"), "prompt_sha256": L.get("prompt_sha256"), "error": err})
    return calls, order


def acq_overflows(rows):
    rank, step = 0, 0
    for i, e in enumerate(rows):
        if e.get("kind") != "llm" or e.get("window") != "acquisition":
            continue
        L = e.get("llm") or {}
        if L.get("finish_reason") != "error":
            continue
        msg = ""
        for f in rows[i + 1:i + 4]:
            if f.get("kind") in ("failure", "retry") and (f.get("failure") or {}).get("type") == "LLMTransportError":
                msg = (f.get("failure") or {}).get("message") or ""
                break
        if "HTTP Error 400" not in msg:
            continue
        if RANK_MARK in (L.get("prompt") or ""):
            rank += 1
        else:
            step += 1
    return rank, step


def replay(attempts):
    probs = []
    for i, a in enumerate(attempts):
        if a["finish_reason"] == "error":
            if i != len(attempts) - 1:
                probs.append("attempts continue after a transport error")
            return None, i, probs
        obj, _ = CR.parse_json(a["completion"])
        if isinstance(obj, dict):
            if i != len(attempts) - 1:
                probs.append("attempts continue after a parsed answer")
            return obj, i, probs
    if len(attempts) > 2:
        probs.append(f"{len(attempts)} attempts (ask_json makes at most 2)")
    return None, len(attempts) - 1, probs


def user_of(prompt):
    msgs = CR.messages_from_flat(prompt)
    return msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else None, \
        next((m["content"] for m in msgs if m["role"] == "user"), "")


LIST_START = "repeat an id.\n\n"
LIST_END = "\n\nReply with ONLY one JSON object, on a SINGLE LINE"
ENTRY = re.compile(r"^  (\S+) \[")


def listing_block(user):
    i = user.find(LIST_START)
    j = user.rfind(LIST_END)
    if i < 0 or j < 0 or j < i:
        return None
    return user[i + len(LIST_START):j]


def lines_of(prompt):
    _, u = user_of(prompt)
    blk = listing_block(u)
    if blk is None:
        return [], []
    ids, entries = [], []
    for ln in blk.split("\n"):
        m = ENTRY.match(ln)
        if m:
            ids.append(m.group(1))
            entries.append(ln)
        elif entries:
            entries[-1] += "\n" + ln
    return ids, entries


def asked_of(prompt):
    _, u = user_of(prompt)
    m = ASK.search(u)
    return int(m.group(1)) if m else None


def select(obj, allowed):
    out, seen, raw = [], set(), []
    if obj:
        for pid in (obj.get("selected") or []):
            pid = str(pid).strip()
            raw.append(pid)
            if pid in allowed and pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out, raw


def rebuild(W, spec, K, lines):
    return [{"role": "system", "content": W.ACQ_SYS},
            {"role": "user", "content": W.ACQ_RANK.format(question=spec["question"], cutoff=spec["publication_cutoff"],
                                                          K=K, candidates="\n".join(lines))}]


def is_400(a):
    return "HTTP Error 400" in str(a.get("error") or "")


def unit_audit(unit_dir, unit_seed=0, counter=None, maxlen=None):
    _runners()
    import windows as W
    d = Path(unit_dir)
    probs = []
    rec = json.loads((d / CR.RECORD).read_text()) if (d / CR.RECORD).exists() else None
    brec = json.loads((d / "ranking_fallback_record.json").read_text()) if (d / "ranking_fallback_record.json").exists() else None
    b = json.loads((d / "evidence_bundle.json").read_text())
    spec = json.loads((d / "task_spec.json").read_text())
    rows = jl(d / "trace.jsonl") or []
    v = b.get("validation") or {}
    K = int(v["K"])
    bundle = [str(p["paper_id"]) for p in b.get("papers") or []]
    known = set(str(x) for x in (b.get("discovered_union") or []))
    pool_size = int(v.get("pool_size") or 0)
    u = int(unit_seed)
    maxlen = maxlen or ((rec or {}).get("config") or {}).get("max_model_len") or 65536
    calls, order = rank_calls(rows)
    nb = -(-pool_size // CR.BATCH) if pool_size else 0
    per_batch = max(K // 2, CR.PER_BATCH_MIN)
    batch_seeds = [u + CR.BATCH_SEED_OFFSET + i for i in range(nb)]
    chunk_seeds = sorted(s for s in calls if s is not None and s >= u + CR.CHUNK_SEED_OFFSET)
    other = [s for s in calls if s not in batch_seeds and s != u + CR.FINAL_SEED_OFFSET and s not in chunk_seeds]
    if other:
        probs.append(f"ranking calls with unexpected seeds {other[:5]}")
    survivors, n_resized, n_empty, lines_by_id = [], 0, 0, {}
    for i, s in enumerate(batch_seeds):
        at = calls.get(s)
        if not at:
            probs.append(f"batch {i} (seed {s}) missing from the trace")
            continue
        ids, lines = lines_of(at[0]["prompt"])
        for pid, ln in zip(ids, lines):
            lines_by_id[pid] = ln
        karg = asked_of(at[0]["prompt"])
        if karg != per_batch:
            probs.append(f"batch {i}: asked {karg} != max(K//2, 20) = {per_batch}")
        want_mt = CR.batch_max_tokens(per_batch, len(ids)) or CR.SINGLE_MAX_TOKENS
        if at[0]["max_tokens"] != want_mt:
            probs.append(f"batch {i}: first-attempt max_tokens {at[0]['max_tokens']} != {want_mt}")
        if want_mt != CR.SINGLE_MAX_TOKENS:
            n_resized += 1
        obj, _, p = replay(at)
        probs += [f"batch {i}: {x}" for x in p]
        sel, _ = select(obj, known)
        if not sel:
            n_empty += 1
        survivors += sel
    n_dup_survivors = len(survivors) - len(set(survivors))
    need_final = len(survivors) > K and nb > 1
    has_single = (u + CR.FINAL_SEED_OFFSET) in calls
    mode = "single" if has_single else ("tournament" if chunk_seeds else None)
    if has_single and chunk_seeds:
        probs.append("both a single final call and chunk calls in one unit")
    if need_final != (mode is not None):
        probs.append(f"final {'missing' if need_final else 'unexpected'}: survivors {len(survivors)} K {K} batches {nb}")
    expected = survivors[:K]
    single_fb, n_fb, failed, n_chunks, overflow, merged = False, 0, [], 0, 0, None
    decision_ok = None
    if mode == "single":
        at = calls[u + CR.FINAL_SEED_OFFSET]
        ids, lines = lines_of(at[0]["prompt"])
        if ids != survivors:
            probs.append("single final: candidates != survivors")
        if asked_of(at[0]["prompt"]) != K or at[0]["max_tokens"] != CR.SINGLE_MAX_TOKENS:
            probs.append("single final: asked / max_tokens differ from windows' (K, 900)")
        if counter is not None:
            P = counter.count(CR.messages_from_flat(at[0]["prompt"]))
            decision_ok = CR.decide_single(P, K, maxlen)["mode"] == "single"
            if not decision_ok:
                probs.append(f"single final ran although it does not fit (P {P}, K {K})")
        obj, _, p = replay(at)
        probs += [f"single final: {x}" for x in p]
        sel, _ = select(obj, set(survivors))
        single_fb = not sel
        expected = (sel or survivors)[:K]
        n_fb = len(expected) if single_fb else 0
        overflow = sum(1 for a in at if is_400(a))
        n_chunks = 1
    elif mode == "tournament":
        n = len(chunk_seeds)
        n_chunks = n
        if chunk_seeds != [u + CR.CHUNK_SEED_OFFSET + i for i in range(n)]:
            probs.append(f"chunk seeds {chunk_seeds[:5]}.. are not u+{CR.CHUNK_SEED_OFFSET}+0..{n - 1}")
        N = len(survivors)
        sizes = CR.even_sizes(N, n) if n else []
        cat, lists, quotas, start = [], [], [], 0
        for i, s in enumerate(chunk_seeds):
            at = calls[s]
            ids, lines = lines_of(at[0]["prompt"])
            cat += ids
            q = asked_of(at[0]["prompt"])
            sz = sizes[i] if i < len(sizes) else None
            if len(ids) != sz:
                probs.append(f"chunk {i}: {len(ids)} candidates != even split size {sz}")
            want_q = min(CR.quota(K, len(ids), N), len(ids)) if N else None
            if q != want_q:
                probs.append(f"chunk {i}: quota {q} != ceil(K*{len(ids)}/{N}) = {want_q}")
            if at[0]["max_tokens"] != CR.ids_budget(q or 0):
                probs.append(f"chunk {i}: max_tokens {at[0]['max_tokens']} != ids_budget({q})")
            reb = CR.flat_prompt(rebuild(W, spec, q, lines))
            if reb != at[0]["prompt"]:
                probs.append(f"chunk {i}: the traced prompt is not the ACQ_RANK template over its listing (K := {q})")
            obj, _, p = replay(at)
            probs += [f"chunk {i}: {x}" for x in p]
            sel, _ = select(obj, set(ids))
            kept = sel[:q or 0]
            if not kept:
                failed.append(i)
            lists.append(kept or ids[:q or 0])
            quotas.append(q or 1)
            overflow += sum(1 for a in at if is_400(a))
            start += len(ids)
        if cat != survivors:
            probs.append("chunk candidates concatenated != survivors (batch order)")
        m = CR.merge(lists, quotas, K)
        merged = [pid for pid, _, _ in m]
        n_fb = sum(1 for _, i, _ in m if i in set(failed))
        expected = merged
        if counter is not None and N:
            full = [lines_by_id.get(x) for x in survivors]
            if any(x is None for x in full):
                probs.append("a survivor has no listing line in any batch prompt")
            else:
                P = counter.count(rebuild(W, spec, K, full))
                decision_ok = CR.decide_single(P, K, maxlen)["mode"] == "tournament"
                if not decision_ok:
                    probs.append(f"tournament ran although the single call fits (P {P}, K {K})")
                n0 = max(2, -(-N // CR.CHUNK_MAX))
                if n < n0:
                    probs.append(f"{n} chunks < the minimum {n0} (CHUNK_MAX {CR.CHUNK_MAX})")
                for i, s in enumerate(chunk_seeds):
                    Pi = counter.count(CR.messages_from_flat(calls[s][0]["prompt"]))
                    f1, f2 = CR.fits(Pi, calls[s][0]["max_tokens"], maxlen)
                    if not (f1 and f2):
                        probs.append(f"chunk {i}: counted prompt {Pi} + max_tokens does not fit {maxlen}")
                for n2 in range(n0, n):
                    st, fit_all = 0, True
                    for sz in CR.even_sizes(N, n2):
                        q2 = min(CR.quota(K, sz, N), sz)
                        P2 = counter.count(rebuild(W, spec, q2, full[st:st + sz]))
                        a, b2 = CR.fits(P2, CR.ids_budget(q2), maxlen)
                        st += sz
                        if not (a and b2):
                            fit_all = False
                            break
                    if fit_all:
                        probs.append(f"{n2} chunks would have fit: {n} is not the fewest")
                        break
    tool_fb = v.get("ranking") == "tool_score_fallback"
    if tool_fb:
        expected = [x for x in bundle]
        n_fb = len(bundle)
    if bundle != expected[:K] and not tool_fb:
        probs.append(f"bundle ({len(bundle)} papers) != the selection recomputed from the trace ({len(expected[:K])})")
    task_fb = bool(single_fb or tool_fb or n_fb > 0)
    rank_400_all, step_400 = acq_overflows(rows)
    all_final = ([a for a in calls.get(u + 1, [])] if mode == "single" else
                 [a for s in chunk_seeds for a in calls[s]])
    trace = {"final_called": mode is not None, "final_mode": mode, "n_chunks": n_chunks, "failed_chunks": failed,
             "whole_task_fallback": single_fb, "tool_score_fallback": tool_fb, "n_final_from_fallback": n_fb,
             "task_fallback": task_fb, "n_context_overflow_attempts": overflow, "n_batch_resized": n_resized,
             "n_batches": nb, "n_batch_empty": n_empty, "n_survivors": len(survivors),
             "n_final_attempts": len(all_final), "n_rank_overflow_attempts": rank_400_all,
             "n_query_step_overflow_attempts": step_400, "n_duplicate_survivors": n_dup_survivors}
    agree_probs = []
    if rec is None:
        agree_probs.append(f"{CR.RECORD} missing")
    else:
        if not rec.get("ok"):
            agree_probs.append(f"record problems: {rec.get('problems')}")
        rf = rec.get("final") or {}
        rfailed = sorted(c["index"] for c in (rf.get("chunks") or []) if c.get("failed"))
        pairs = [("final_called", rec.get("final_called"), trace["final_called"]),
                 ("final_mode", rec.get("final_mode"), mode),
                 ("n_chunks", rec.get("n_chunks"), n_chunks),
                 ("failed_chunks", rfailed, sorted(failed)),
                 ("whole_task_fallback", rec.get("whole_task_fallback"), single_fb),
                 ("tool_score_fallback", rec.get("tool_score_fallback"), tool_fb),
                 ("n_final_from_fallback", rec.get("n_final_from_fallback"), n_fb),
                 ("task_fallback", rec.get("task_fallback"), task_fb),
                 ("n_context_overflow_attempts", rec.get("n_context_overflow_attempts"), overflow),
                 ("n_batch_resized", rec.get("n_batch_resized"), n_resized),
                 ("n_bundle_papers", rec.get("n_bundle_papers"), len(bundle))]
        if mode == "tournament":
            pairs.append(("merged", rf.get("merged"), merged))
            pairs.append(("chunk_seeds", rf.get("chunk_seeds"), chunk_seeds))
        for k, a, t in pairs:
            if a != t:
                agree_probs.append(f"record {k} {str(a)[:80]} != trace {str(t)[:80]}")
        shas = {a["prompt_sha256"] for at in calls.values() for a in at}
        att = (rf.get("attempts") or []) + [a for c in (rf.get("chunks") or []) for a in c.get("attempts") or []]
        miss = [a.get("prompt_sha256") for a in att if a.get("prompt_sha256") not in shas]
        if miss:
            agree_probs.append(f"{len(miss)} recorded final-stage attempts are not in the trace")
    if brec is None:
        agree_probs.append("ranking_fallback_record.json missing")
    else:
        if not brec.get("ok"):
            agree_probs.append(f"fallback record problems: {brec.get('problems')}")
        if brec.get("final_called") != trace["final_called"]:
            agree_probs.append("fallback record final_called != trace")
        if brec.get("n_survivors") != len(survivors):
            agree_probs.append(f"fallback record survivors {brec.get('n_survivors')} != trace {len(survivors)}")
        if mode == "tournament" and (brec.get("final_selected") or [])[:K] != merged:
            agree_probs.append("fallback record's final selection != the merged tournament order")
    out = {"unit": str(d), "task": d.parent.name, "K": K, "trace": trace, "problems": probs,
           "agree_problems": agree_probs, "decision_recounted": decision_ok,
           "record_present": rec is not None and brec is not None,
           "record_ok": bool(rec and rec.get("ok") and brec and brec.get("ok")),
           "record_problems": ((rec or {}).get("problems") or []) + ((brec or {}).get("problems") or []),
           "final_called_record": (rec or {}).get("final_called"), "final_called_trace": trace["final_called"],
           "fallback_record": (rec or {}).get("task_fallback"), "fallback_trace": task_fb,
           "cause_record": _cause(rec), "cause_trace": None,
           "tool_score_fallback": tool_fb, "n_survivors": len(survivors), "final_n_candidates": len(survivors) if mode else None,
           "record_attempts_missing_from_trace": sum(1 for x in agree_probs if "not in the trace" in x),
           "final_mode": mode, "n_chunks": n_chunks, "n_chunks_failed": len(failed), "n_final_from_fallback": n_fb,
           "share_from_fallback": (n_fb / len(bundle)) if bundle else None,
           "n_context_overflow_attempts": overflow, "n_papers": len(bundle),
           "n_rank_overflow_attempts": rank_400_all, "n_query_step_overflow_attempts": step_400,
           "n_duplicate_survivors": n_dup_survivors, "fill": (len(bundle) / K) if K else None}
    out["agree"] = bool(rec is not None and not probs and not agree_probs)
    return out


def _cause(rec):
    if not rec or not rec.get("task_fallback"):
        return None
    f = rec.get("final") or {}
    if rec.get("tool_score_fallback"):
        return "tool_score_fallback"
    if rec.get("final_mode") == "single":
        return f"single:{f.get('single_fallback_cause')}"
    causes = sorted({str(c.get("fail_cause")) for c in f.get("chunks") or [] if c.get("failed")})
    return "chunk:" + "+".join(str(c) for c in causes)


def summarize(rows, no_bundle=()):
    import statistics
    n = len(rows)
    nnb = len(no_bundle)
    nt = n + nnb
    nf = sum(1 for r in rows if r["fallback_trace"])
    ov_final = sum(r["n_context_overflow_attempts"] or 0 for r in rows)
    ov_rank = sum(r.get("n_rank_overflow_attempts") or 0 for r in rows)
    ov_step = sum(r.get("n_query_step_overflow_attempts") or 0 for r in rows)
    rate = (nf + nnb) / nt if nt else None
    fills = [r["fill"] for r in rows if r.get("fill") is not None]
    fills_t = [r["fill"] for r in rows if r.get("fill") is not None and r["final_mode"] == "tournament"]
    return {"n_units": n, "n_no_bundle": nnb, "no_bundle": list(no_bundle), "n_tasks": nt,
            "n_task_fallback": nf, "fallback_rate": rate,
            "n_context_overflow_attempts": ov_final, "n_rank_overflow_attempts": ov_rank,
            "n_query_step_overflow_attempts": ov_step, "n_disagree": sum(1 for r in rows if not r["agree"]),
            "n_tournament": sum(1 for r in rows if r["final_mode"] == "tournament"),
            "n_single": sum(1 for r in rows if r["final_mode"] == "single"),
            "n_no_final": sum(1 for r in rows if r["final_mode"] is None),
            "n_duplicate_survivors": sum(r.get("n_duplicate_survivors") or 0 for r in rows),
            "fill_median": statistics.median(fills) if fills else None, "fill_min": min(fills) if fills else None,
            "fill_median_tournament": statistics.median(fills_t) if fills_t else None,
            "fill_min_tournament": min(fills_t) if fills_t else None,
            "n_fill_below_0_9": sum(1 for x in fills if x < 0.9),
            "per_task": [{k: r.get(k) for k in ("task", "final_mode", "n_chunks", "n_chunks_failed",
                                                 "n_final_from_fallback", "share_from_fallback",
                                                 "n_context_overflow_attempts", "n_rank_overflow_attempts",
                                                 "n_query_step_overflow_attempts", "n_duplicate_survivors",
                                                 "fallback_trace", "n_papers", "K", "fill", "agree")} for r in rows]
                        + [{"task": t, "final_mode": None, "no_bundle": True, "fallback_trace": True, "agree": None}
                           for t in no_bundle]}

