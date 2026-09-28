#!/usr/bin/env python3
"""Query agent of the acquisition window: subtopic decomposition, duplicate-query guard, per-query
yield ledger and stall rule, prompted with the skill files; writes query_agent_record.json per unit."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILLS_DIR = HERE / "skills"
VERSION = "query_agent/1.0"
STOP_TOOL_REFUSED = "tool_refused_search"
RECORD = "query_agent_record.json"
RECORD_SCHEMA = "query_agent_record/1.0"
SKILL_FILES = ("s_guard.txt", "s_subtopic.txt")
SKILL_MAX_LINES = 25

DUP_JACCARD = 0.8
STALL_ZERO_STEPS = 2
STALL_JACCARD = 0.5
MAX_CONSEC_REFUSALS = 4
N_SUB_MAX = 10
N_SUB_MIN_USABLE = 2
SUBTOPIC_SEED_OFFSET = 5000
QUERY_MAX_TOKENS = 300
SUBTOPIC_MAX_TOKENS = 1500
QTYPE_ROW = re.compile(r"^[ \t]+([a-z][a-z0-9-]*)[ \t]+(\d+)[ \t]+quer(?:y|ies)\b", re.M)
SAFETY_CALLS_PER_SEARCH = 3
LEDGER_QUERY_CHARS = 200

QUERY_STEP = """TOPIC: {question}
FIELD/SCOPE: {topic}
PUBLICATION CUTOFF: {cutoff} (only work published before this year exists for you)
EXCLUDED: {exclusion}

SEARCHES USED: {used}/{cap}   DISTINCT PAPERS FOUND SO FAR: {n_pool}
{coverage}QUERY LEDGER (every search run so far, with the number of NEW papers it added):
{ledger}
{notice}
Issue the NEXT search query{target}, or stop if further searching would not add papers a review of
this topic needs.

Reply with ONLY one JSON object:
  {{"action": "search", "query": "<query text>", "why": "<one clause>"}}
or
  {{"action": "stop", "why": "<one clause>"}}"""

QUERY_SUBTOPIC = """TOPIC: {question}
FIELD/SCOPE: {topic}
PUBLICATION CUTOFF: {cutoff} (only work published before this year exists for you)
EXCLUDED: {exclusion}

PROPOSE THE SUBTOPICS of this review before any search is run. The search budget ({budget} searches) will be
split across them in the order you give.

Reply with ONLY one JSON object, on a SINGLE LINE, no indentation:
  {{"subtopics": [{{"name": "<short name>", "focus": "<one clause: which papers belong here>"}}, ...]}}"""

STOP = set('the and for with from into onto that this these those are was were been being its their of in on to a an '
           'by as at or is be not no via using use based review systematic meta analysis update updated recent '
           'current new role'.split())


class QueryAgentConfigError(RuntimeError):
    pass


def toks(q):
    return {t for t in re.findall(r'[a-z0-9]+', str(q).lower())
            if len(t) > 2 and t not in STOP and not re.fullmatch(r'(19|20)\d\d', t)}


def norm(q):
    return ' '.join(str(q).lower().split())


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def find_duplicate(q, past):
    nq, tq = norm(q), toks(q)
    best = None
    for i, p in enumerate(past, 1):
        if norm(p) == nq:
            return (i, 1.0, "exact")
        tp = toks(p)
        if tq and tp:
            j = jaccard(tq, tp)
            if j >= DUP_JACCARD and best is None:
                best = (i, round(j, 4), "near")
    return best


def stall_conflict(q, refs):
    tq = toks(q)
    for i, p in refs:
        tp = toks(p)
        j = jaccard(tq, tp) if (tq and tp) else (1.0 if norm(q) == norm(p) else 0.0)
        if j >= STALL_JACCARD:
            return (i, round(j, 4))
    return None


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def load_skills(skills_dir=None):
    d = Path(skills_dir) if skills_dir else SKILLS_DIR
    out = []
    for name in SKILL_FILES:
        p = d / name
        if not p.is_file():
            raise QueryAgentConfigError(f"skill file {p} missing")
        raw = p.read_bytes()
        txt = raw.decode("utf-8")
        n = txt.count("\n")
        if not txt.strip() or not txt.endswith("\n") or n > SKILL_MAX_LINES:
            raise QueryAgentConfigError(f"skill file {p}: must be non-empty, newline-terminated, <= {SKILL_MAX_LINES} lines "
                                 f"(has {n})")
        out.append((name, txt, md5_bytes(raw)))
    return out


def qtype_schedule(skill_text):
    out = []
    for name, n in QTYPE_ROW.findall(skill_text):
        out += [name] * int(n)
    return tuple(out)


def config(skills_dir=None):
    sk = load_skills(skills_dir)
    qtypes = list(qtype_schedule(dict((n, t) for n, t, _ in sk).get("s_subtopic.txt", "")))
    return {"version": VERSION, "module": str(Path(__file__).resolve()),
            "module_md5": md5_bytes(Path(__file__).read_bytes()),
            "skills": [{"name": n, "md5": m, "path": str((Path(skills_dir) if skills_dir else SKILLS_DIR) / n)}
                       for n, _, m in sk],
            "_skill_text": {n: t for n, t, _ in sk},
            "constants": {"DUP_JACCARD": DUP_JACCARD, "STALL_ZERO_STEPS": STALL_ZERO_STEPS,
                          "STALL_JACCARD": STALL_JACCARD, "MAX_CONSEC_REFUSALS": MAX_CONSEC_REFUSALS,
                          "LEDGER_QUERY_CHARS": LEDGER_QUERY_CHARS,
                          "N_SUB_MAX": N_SUB_MAX, "N_SUB_MIN_USABLE": N_SUB_MIN_USABLE,
                          "SUBTOPIC_SEED_OFFSET": SUBTOPIC_SEED_OFFSET,
                          "QUERY_MAX_TOKENS": QUERY_MAX_TOKENS, "SUBTOPIC_MAX_TOKENS": SUBTOPIC_MAX_TOKENS,
                          "QTYPES": qtypes, "QTYPES_SOURCE": "skills/s_subtopic.txt (qtype_schedule)"}}


def public_cfg(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def system_prompt(W, cfg, names):
    return W.ACQ_SYS + "\n\n" + "".join(cfg["_skill_text"][n] for n in names)


class _Harness:
    def __init__(self, W, cfg, spec, tool, client, log, seed, cap):
        self.W, self.cfg, self.spec, self.tool, self.client, self.log = W, cfg, spec, tool, client, log
        self.seed = int(seed)
        self.K = int(spec["budget"]["max_ranked_output_K"])
        self.cap = int(cap)
        self.llm_budget = self.cap
        self.executed = []
        self.refused_by_tool = []
        self.searches = []
        self.proposals = []
        self.q_idx = 0
        self.stop_reason = "model_stop"
        self.consec_zero = 0
        self.consec_refused = 0
        self.stall_refs = []
        self.streak = []
        self.info = ""
        self.notice_text = ""
        self.subs = None
        self.sub_rec = None
        self.cur = None
        self.quota = 0
        self.used_in_cur = 0
        self.entered = set()
        self.pass_no = 0
        self.n_calls = {"query": 0, "subtopic": 0}
        self.problems = []
        self.notes = []

    def header(self):
        s = self.spec
        return dict(question=s["question"], topic=s["scope"]["intervention_or_topic"],
                    cutoff=s["publication_cutoff"],
                    exclusion="; ".join(s["scope"]["exclusion"]) or "none")

    def n_searched(self):
        return len(self.searches)

    def ask(self, kind, sys_names, user, max_tokens, seed):
        msgs = [{"role": "system", "content": system_prompt(self.W, self.cfg, sys_names)},
                {"role": "user", "content": user}]
        obj, draw, err = self.W.ask_json(self.client, self.log, "acquisition", msgs, max_tokens=max_tokens,
                                         seed=seed)
        self.n_calls[kind] += 1
        return obj, draw, err

    def tool_used(self):
        u = getattr(getattr(self.tool, "tool", None), "used", None)
        return u.get("search_calls") if isinstance(u, dict) else None

    def search(self, q, source, subtopic=None, p_idx=None):
        n0 = len(self.tool.pool)
        used_before = self.n_searched()
        t0 = self.tool_used()
        rows = self.tool.search(q, k=self.K, route="both", step=used_before)
        if rows is None and self.tool.exhausted["search"] and (t0 is None or self.tool_used() == t0):
            self.refused_by_tool.append({"attempt_s_idx": used_before, "source": source, "query": q,
                                         "subtopic": subtopic, "p_idx": p_idx,
                                         "tool_search_calls": self.tool_used(),
                                         "pool_size": len(self.tool.pool)})
            if len(self.tool.pool) != n0:
                self.problems.append("a tool-refused search changed the pool")
            return None, None
        self.executed.append(q)
        n_new = len(self.tool.pool) - n0
        rec = {"s_idx": used_before, "source": source, "query": q, "subtopic": subtopic, "p_idx": p_idx,
               "n_returned": None if rows is None else len(rows), "n_new": n_new,
               "pool_after": len(self.tool.pool), "budget_refused": bool(rows is None and self.tool.exhausted["search"])}
        self.searches.append(rec)
        if self.subs is not None and subtopic is not None:
            self.subs[subtopic]["n_exec"] += 1
            self.subs[subtopic]["n_new"] += n_new
        return rows, n_new

    def ledger(self):
        if not self.searches:
            return "  (none yet)"
        out = []
        for i, s in enumerate(self.searches, 1):
            tag = f"  (subtopic {s['subtopic'] + 1})" if s["subtopic"] is not None else ""
            out.append(f"  {i}. [+{s['n_new']} new] {self.W.clip(s['query'], LEDGER_QUERY_CHARS, '...')}{tag}")
        return "\n".join(out)

    def coverage(self):
        if self.subs is None:
            return ""
        lines = ["SUBTOPICS (queries run, new papers added):"]
        for i, s in enumerate(self.subs, 1):
            closed = "  [closed]" if s["dead"] else ""
            lines.append(f"  {i}. {s['name']} -- {s['focus']}: {s['n_exec']} queries, +{s['n_new']} new{closed}")
        if self.cur is not None:
            s = self.subs[self.cur]
            qt = self.qtype(self.cur)
            lines.append(f"CURRENT SUBTOPIC: {self.cur + 1}. {s['name']} (query {self.used_in_cur + 1} of "
                         f"{self.quota} for it in this pass" + (f"; suggested query type: {qt})" if qt else ")"))
        return "\n".join(lines) + "\n"

    def qtype(self, i):
        qt = self.cfg["constants"]["QTYPES"]
        return qt[self.subs[i]["n_exec"] % len(qt)] if qt else None

    def step_prompt(self):
        return QUERY_STEP.format(**self.header(), used=self.n_searched(), cap=self.cap, n_pool=len(self.tool.pool),
                               coverage=self.coverage(), ledger=self.ledger(),
                               notice=(self.notice_text + "\n") if self.notice_text else "",
                               target=" for the CURRENT SUBTOPIC" if self.cur is not None else "")

    def decompose(self):
        user = QUERY_SUBTOPIC.format(**self.header(), budget=self.llm_budget)
        seed = self.seed + SUBTOPIC_SEED_OFFSET
        obj, draw, err = self.ask("subtopic", ("s_subtopic.txt",), user, SUBTOPIC_MAX_TOKENS, seed)
        rec = {"called": True, "seed": seed, "ok": False, "n_raw": None, "used": [], "dropped": [],
               "no_subtopics": False, "error": err}
        raw = (obj or {}).get("subtopics") if isinstance(obj, dict) else None
        subs, seen = [], set()
        if isinstance(raw, list):
            rec["n_raw"] = len(raw)
            for x in raw:
                if isinstance(x, dict):
                    name, focus = str(x.get("name") or "").strip(), str(x.get("focus") or "").strip()
                else:
                    name, focus = str(x or "").strip(), ""
                if not name or norm(name) in seen:
                    rec["dropped"].append({"item": str(x)[:200], "why": "empty or repeated name"})
                    continue
                seen.add(norm(name))
                subs.append({"name": name[:120], "focus": focus[:240]})
        limit = min(N_SUB_MAX, max(self.llm_budget, 0))
        if len(subs) > limit:
            rec["dropped"] += [{"item": s["name"], "why": f"beyond the first {limit} (N_SUB_MAX / query budget)"}
                               for s in subs[limit:]]
            subs = subs[:limit]
        if len(subs) < N_SUB_MIN_USABLE:
            rec["no_subtopics"] = True
            self.notes.append(f"subtopic decomposition gave {len(subs)} usable subtopics: the query phase runs "
                              f"without subtopics")
        else:
            rec["ok"] = True
            self.subs = [dict(s, n_exec=0, n_new=0, dead=False) for s in subs]
        rec["used"] = subs if rec["ok"] else []
        self.sub_rec = rec

    def llm_remaining(self):
        return self.llm_budget - sum(1 for s in self.searches if s["source"] == "llm")

    def enter_next(self, why):
        live = [i for i, s in enumerate(self.subs) if not s["dead"]]
        if not live:
            return False
        cand = [i for i in live if i not in self.entered]
        if not cand:
            self.entered = set()
            self.pass_no += 1
            cand = list(live)
        if why == "stall":
            unc = [i for i in cand if self.subs[i]["n_exec"] == 0]
            nxt = (unc or sorted(cand, key=lambda i: (self.subs[i]["n_exec"], i)))[0]
        else:
            nxt = cand[0]
        self.cur = nxt
        self.entered.add(nxt)
        self.quota = max(1, math.ceil(max(self.llm_remaining(), 0) / len(cand)))
        self.used_in_cur = 0
        return True

    def refuse(self, prop, decision, item):
        prop["decision"] = decision
        self.consec_refused += 1
        self.streak.append(item)
        return self.consec_refused >= MAX_CONSEC_REFUSALS

    def build_notice(self):
        parts = [self.info] if self.info else []
        if self.subs is None and self.stall_refs:
            refs = ", ".join(str(i) for i, _ in self.stall_refs)
            parts.append(f"NOTICE: the last {len(self.stall_refs)} searches (ledger {refs}) added 0 new papers. The "
                         f"next query must take a different angle: its content-word overlap with each of them must "
                         f"stay below {STALL_JACCARD:.2f}.")
        if self.streak:
            parts.append("NOTICE: these proposals for the next search were NOT run and NOT charged:\n"
                         + "\n".join(f"  - {x}" for x in self.streak))
        self.info = ""
        return "\n".join(parts)

    def query_phase(self):
        if self.subs is not None:
            self.enter_next("start")
        safety = SAFETY_CALLS_PER_SEARCH * self.cap + 10
        while self.llm_remaining() > 0:
            if self.q_idx >= safety:
                self.stop_reason = "query_agent_call_safety_limit"
                self.notes.append(f"query-prompt calls reached the safety bound {safety}")
                return
            self.notice_text = self.build_notice()
            user = self.step_prompt()
            seed = self.seed + self.q_idx
            shown_notice = self.notice_text
            obj, draw, err = self.ask("query", self.step_skills(), user, QUERY_MAX_TOKENS, seed)
            prop = {"p_idx": len(self.proposals), "q_idx": self.q_idx, "seed": seed, "used_before": self.n_searched(),
                    "subtopic": self.cur, "qtype": (self.qtype(self.cur) if self.cur is not None else None),
                    "notice_shown": shown_notice, "action": None, "query": None, "decision": None}
            self.proposals.append(prop)
            self.q_idx += 1
            if obj is None:
                prop["decision"] = "query_generation_failed"
                self.stop_reason = "query_generation_failed"
                return
            act = str(obj.get("action", "")).lower()
            prop["action"] = act
            if act == "stop":
                unc = [i + 1 for i, s in enumerate(self.subs or []) if s["n_exec"] == 0 and not s["dead"]]
                if self.subs is not None and unc:
                    lim = self.refuse(prop, "refused_stop",
                                      f"stop -- subtopic(s) {', '.join(map(str, unc))} have no search yet")
                    if lim and not self.close_current("refusals"):
                        return
                    continue
                prop["decision"] = "stop_accepted"
                self.stop_reason = "model_stop"
                return
            q = str(obj.get("query") or "").strip()
            prop["query"] = q
            if not q:
                prop["decision"] = "empty_query"
                self.stop_reason = "empty_query"
                return
            dup = find_duplicate(q, self.executed)
            if dup:
                prop.update(dup_of=dup[0], dup_jaccard=dup[1], dup_kind=dup[2])
                lim = self.refuse(prop, "refused_duplicate",
                                  f"\"{self.W.clip(q, LEDGER_QUERY_CHARS, '...')}\" -- repeats ledger query {dup[0]} "
                                  f"\"{self.W.clip(self.executed[dup[0] - 1], LEDGER_QUERY_CHARS, '...')}\" "
                                  f"(content-word overlap {dup[1]:.2f}, limit {DUP_JACCARD:.2f})")
                if lim and not self.after_refusal_limit():
                    return
                continue
            if self.subs is None and self.stall_refs:
                sc = stall_conflict(q, self.stall_refs)
                if sc:
                    prop.update(stall_conflict=sc[0], stall_jaccard=sc[1])
                    lim = self.refuse(prop, "refused_stall",
                                      f"\"{self.W.clip(q, LEDGER_QUERY_CHARS, '...')}\" -- overlaps {sc[1]:.2f} with "
                                      f"zero-yield ledger query {sc[0]} (the "
                                      f"stall rule needs < {STALL_JACCARD:.2f})")
                    if lim and not self.after_refusal_limit():
                        return
                    continue
            prop["decision"] = "executed"
            self.consec_refused = 0
            self.streak = []
            rows, n_new = self.search(q, "llm", subtopic=self.cur, p_idx=prop["p_idx"])
            if n_new is None:
                prop["decision"] = "refused_by_tool"
                prop["s_idx"] = None
                self.stop_reason = STOP_TOOL_REFUSED
                return
            prop["s_idx"] = self.searches[-1]["s_idx"]
            if rows is None and self.tool.exhausted["search"]:
                self.stop_reason = "search_budget_exhausted"
                return
            self.consec_zero = self.consec_zero + 1 if n_new == 0 else 0
            if self.subs is not None:
                self.used_in_cur += 1
                if self.consec_zero >= STALL_ZERO_STEPS:
                    self.consec_zero = 0
                    old = self.cur
                    if not self.enter_next("stall"):
                        self.stop_reason = "subtopics_exhausted"
                        return
                    s = self.subs[self.cur]
                    self.info = (f"NOTICE: the last {STALL_ZERO_STEPS} searches (subtopic {old + 1}) added 0 new "
                                   f"papers; the harness moved on to subtopic {self.cur + 1} ({s['name']}), which has "
                                   f"{s['n_exec']} queries so far.")
                    prop["stall_moved_to"] = self.cur
                elif self.used_in_cur >= self.quota and self.llm_remaining() > 0:
                    if not self.enter_next("quota"):
                        self.stop_reason = "subtopics_exhausted"
                        return
            else:
                if n_new > 0:
                    self.stall_refs = []
                elif self.consec_zero >= STALL_ZERO_STEPS:
                    k = len(self.searches)
                    self.stall_refs = [(i, self.searches[i - 1]["query"]) for i in range(k - STALL_ZERO_STEPS + 1, k + 1)]
                    prop["stall_raised"] = [i for i, _ in self.stall_refs]

    def step_skills(self):
        return SKILL_FILES

    def after_refusal_limit(self):
        if self.subs is None:
            self.stop_reason = "query_agent_refusal_limit"
            return False
        return self.close_current("refusals")

    def close_current(self, why):
        old = self.cur
        self.subs[old]["dead"] = True
        self.consec_refused = 0
        self.streak = []
        if not self.enter_next(why):
            self.stop_reason = "subtopics_exhausted"
            return False
        self.info = (f"NOTICE: {MAX_CONSEC_REFUSALS} proposals in a row were refused; subtopic {old + 1} is closed and "
                       f"the harness moved on to subtopic {self.cur + 1} ({self.subs[self.cur]['name']}).")
        return True

    def run(self):
        if self.llm_budget > 0:
            self.decompose()
            self.query_phase()

    def record(self):
        dec = {}
        for p in self.proposals:
            dec[p["decision"]] = dec.get(p["decision"], 0) + 1
        return {"cap": self.cap, "llm_budget": self.llm_budget, "K": self.K, "unit_seed": self.seed,
                "subtopics": self.sub_rec, "subtopic_state": self.subs, "proposals": self.proposals,
                "searches": self.searches, "executed_queries": list(self.executed),
                "refused_by_tool": list(self.refused_by_tool),
                "stop_reason": self.stop_reason, "decisions": dec, "n_calls": dict(self.n_calls),
                "n_searches": self.n_searched(), "n_new_total": sum(s["n_new"] for s in self.searches),
                "n_zero_yield": sum(1 for s in self.searches if s["n_new"] == 0), "problems": list(self.problems),
                "notes": list(self.notes)}


_TL = threading.local()
_INSTALLED = {}


def install(W=None, skills_dir=None):
    if _INSTALLED.get("cfg"):
        return _INSTALLED["cfg"]
    cfg = config(skills_dir)
    if W is None:
        import windows as W
    import sys
    pbk = sys.modules.get("budget")
    if getattr(W.acquisition, "_cap_budget", False) or (pbk is not None and getattr(pbk, "_INSTALLED", {}).get("rank")):
        raise QueryAgentConfigError("the cap-budget rank recorder already wraps windows.acquisition: install the query agent before it")
    for name in ("ACQ_SYS", "ask_json", "clip", "_sealed", "acquisition"):
        if not hasattr(W, name):
            raise QueryAgentConfigError(f"windows has no {name}")
    orig_acq, orig_sealed = W.acquisition, W._sealed

    def _sealed(artifact, name, log, window, started):
        if getattr(_TL, "defer", False) and name == "evidence_bundle":
            _TL.deferred = True
            return artifact, []
        return orig_sealed(artifact, name, log, window, started)

    def acquisition(spec, tool, client, log, seed=0):
        from common import seal, utcnow
        unit_dir = Path(log.path).parent
        h = _Harness(W, cfg, spec, tool, client, log, seed, spec["budget"]["max_search_calls"])
        rec = {"schema": RECORD_SCHEMA, "cfg": public_cfg(cfg), "task": str(spec.get("task_id")), "ok": False,
               "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        exc = None
        try:
            h.run()
            _TL.defer, _TL.deferred = True, False
            try:
                bundle, _ = orig_acq(spec, tool, client, log, seed=seed)
            finally:
                deferred = _TL.deferred
                _TL.defer, _TL.deferred = False, False
            if not deferred:
                raise QueryAgentConfigError("the inner acquisition did not seal through windows._sealed")
            b = dict(bundle)
            b.pop("content_hash", None)
            v = dict(b.get("validation") or {})
            v["queries"] = list(h.executed)
            v["stop_reason"] = h.stop_reason
            v["query_agent"] = {"version": VERSION, "module_md5": cfg["module_md5"],
                                "skills_md5": {s["name"]: s["md5"] for s in cfg["skills"]}, "n_proposals": len(h.proposals),
                                "decisions": h.record()["decisions"], "n_calls": dict(h.n_calls),
                                "n_subtopics": len(h.subs) if h.subs else 0, "record": RECORD,
                                "refused_by_tool": [x["query"] for x in h.refused_by_tool]}
            b["validation"] = v
            b["failures"] = [h.stop_reason] if h.stop_reason != "model_stop" else []
            status = "budget_exhausted" if (tool.exhausted["search"] or tool.exhausted["open"]) else "done"
            if h.stop_reason == "query_generation_failed" and not b.get("papers"):
                status = "failed"
            b["retrieval_status"] = status
            b = seal(b)
            out = orig_sealed(b, "evidence_bundle", log, "acquisition", utcnow())
            rec.update(ok=not h.problems, bundle_content_hash=b["content_hash"], n_papers=len(b.get("papers") or []))
            return out
        except BaseException as e:
            exc = e
            raise
        finally:
            try:
                rec.update(h.record())
                used_tool = tool.tool.used.get("search_calls") if hasattr(tool, "tool") else None
                rec["tool_search_calls"] = used_tool
                if used_tool is not None and used_tool != h.n_searched():
                    rec["problems"].append(f"tool counted {used_tool} searches, harness {h.n_searched()}")
                    rec["ok"] = False
                if exc is not None:
                    rec["exception"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                rec["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                (unit_dir / RECORD).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            except Exception as e:
                if exc is None:
                    raise QueryAgentConfigError(f"cannot write {unit_dir / RECORD}: {e}")

    acquisition._query_agent = True
    acquisition._query_agent_inner = orig_acq
    _sealed._query_agent = True
    W._QUERY_AGENT_ORIG_ACQ, W._QUERY_AGENT_ORIG_SEALED = orig_acq, orig_sealed
    W.acquisition, W._sealed = acquisition, _sealed
    _INSTALLED["cfg"] = cfg
    return cfg


def stamp_manifests(R, cfg):
    if getattr(R.manifest, "_query_agent_stamped", False):
        return
    orig = R.manifest

    def manifest(*a, **k):
        extra = dict(k.get("extra") or {})
        extra.setdefault("query_agent", cfg["version"])
        extra.setdefault("query_agent_module_md5", cfg["module_md5"])
        extra.setdefault("query_agent_skills_md5", {s["name"]: s["md5"] for s in cfg["skills"]})
        k["extra"] = extra
        return orig(*a, **k)
    for attr in ("_cap_budget_stamped", "_budget_stamped", "_ranker_stamped"):
        if getattr(orig, attr, False):
            setattr(manifest, attr, True)
    manifest._query_agent_stamped = True
    R.manifest = manifest
