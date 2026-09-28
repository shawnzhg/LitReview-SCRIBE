"""The common observation channel: matches output sentences to human claims above the calibrated
threshold and headings to reference sections, and records citations."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import threading
import warnings
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from ccbench import paths
from ccbench.adapters.common import split_sentences, strip_title_number
from ccbench.config import prereg
from ccbench.gt import graphs
from ccbench.gt.units import Units
from ccbench.model import Rollout
from ccbench.readouts import embed


@dataclass
class Induced:
    system: str
    task: str
    claim_match: dict[str, tuple[str, float]] = field(default_factory=dict)
    sent_match: dict[str, tuple[str, float]] = field(default_factory=dict)
    sent_section: dict[str, str] = field(default_factory=dict)
    sent_paragraph: dict[str, str] = field(default_factory=dict)
    sent_cites: dict[str, set[str]] = field(default_factory=dict)
    section_order: dict[str, int] = field(default_factory=dict)
    section_match: dict[str, str] = field(default_factory=dict)
    n_sentences: int = 0
    n_matched_sentences: int = 0
    margin_q: dict[str, float] = field(default_factory=dict)
    pair_cos: dict[str, float] = field(default_factory=dict)
    edge_tau: float | None = None

    def linked(self, a: str, b: str) -> bool | None:
        if a not in self.claim_match or b not in self.claim_match:
            return None
        c = self.pair_cos.get(f"{a}|{b}", self.pair_cos.get(f"{b}|{a}"))
        if c is None or self.edge_tau is None:
            return None
        return c >= self.edge_tau

    sent_top: dict[str, str] = field(default_factory=dict)
    top_order: dict[str, int] = field(default_factory=dict)
    sys_links: set[str] = field(default_factory=set)
    window: str = "writing"

    def coloc(self, a: str, b: str) -> str | None:
        sa, sb = self.claim_match.get(a), self.claim_match.get(b)
        if not sa or not sb:
            return None
        if self.sys_links and f"{sa[0]}|{sb[0]}" in self.sys_links:
            return "strong"
        pa, pb = self.sent_paragraph.get(sa[0]), self.sent_paragraph.get(sb[0])
        if pa is not None and pa == pb:
            return "strong"
        ta, tb = self.sent_top.get(sa[0]), self.sent_top.get(sb[0])
        if ta is None or tb is None:
            return None
        if ta == tb or abs(self.top_order.get(ta, -9) - self.top_order.get(tb, 9)) == 1:
            return "weak"
        return None

    def to_json(self, p):
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {"system": self.system, "task": self.task, "claim_match": self.claim_match, "sent_match": self.sent_match, "sent_section": self.sent_section, "sent_paragraph": self.sent_paragraph, "sent_cites": {k: sorted(v) for k, v in self.sent_cites.items()}, "section_order": self.section_order, "section_match": self.section_match, "n_sentences": self.n_sentences, "n_matched_sentences": self.n_matched_sentences, "margin_q": self.margin_q, "pair_cos": self.pair_cos, "edge_tau": self.edge_tau, "sent_top": self.sent_top, "top_order": self.top_order, "sys_links": sorted(self.sys_links), "window": self.window}
        json.dump(d, open(p, "w"))

    @staticmethod
    def from_json(p) -> "Induced":
        d = json.load(open(p))
        ind = Induced(system=d["system"], task=d["task"])
        ind.claim_match = {k: tuple(v) for k, v in d["claim_match"].items()}
        ind.sent_match = {k: tuple(v) for k, v in d["sent_match"].items()}
        ind.sent_section = d["sent_section"]
        ind.sent_paragraph = d["sent_paragraph"]
        ind.sent_cites = {k: set(v) for k, v in d["sent_cites"].items()}
        ind.section_order = d["section_order"]
        ind.section_match = d["section_match"]
        ind.n_sentences = d["n_sentences"]
        ind.n_matched_sentences = d["n_matched_sentences"]
        ind.margin_q = d.get("margin_q", {})
        ind.pair_cos = d.get("pair_cos", {})
        ind.edge_tau = d.get("edge_tau")
        ind.sent_top = d.get("sent_top", {})
        ind.top_order = d.get("top_order", {})
        ind.sys_links = set(d.get("sys_links", []))
        ind.window = d.get("window", "writing")
        return ind


def _paragraph_ids(ro: Rollout) -> dict[str, str]:
    out: dict[str, str] = {}
    by_sec: dict[str, list] = {}
    for s in ro.report.sentences:
        by_sec.setdefault(s.section, []).append(s)
    text_of = {sec["id"]: sec.get("text", "") for sec in ro.report.sections}
    for sid, sents in by_sec.items():
        paras = [p for p in re.split(r"\n\s*\n", text_of.get(sid, "")) if p.strip()]
        seq = []
        for k, p in enumerate(paras):
            seq += [f"{sid}/p{k}"] * len(split_sentences(re.sub(r"\[[^\]]*\]", "", p)))
        if len(seq) == len(sents):
            for s, pid in zip(sents, seq):
                out[s.sid] = pid
        else:
            for s in sents:
                out[s.sid] = sid
    return out


_NULL_CACHE: dict[str, float] = {}
_NULL_COMPUTE_LOCK = threading.RLock()


class NullStoreUnreadable(RuntimeError):
    pass


def _read_null_store(p, strict: bool = False) -> dict:
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        if strict:
            raise NullStoreUnreadable(f"{p}: {type(e).__name__}: {e}") from e
        return {}
    if not isinstance(d, dict):
        if strict:
            raise NullStoreUnreadable(f"{p}: top-level JSON is {type(d).__name__}, not an object")
        return {}
    return d


@contextlib.contextmanager
def _null_store_lock(p):
    lock = p.parent / (p.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def null_tau(task: str, g: graphs.GStar, n_reviews: int = 2, q: float = 0.95) -> float:
    if task in _NULL_CACHE:
        return _NULL_CACHE[task]
    p = paths.out_dir("E10") / "null_tau.json"
    store = _read_null_store(p)
    if task in store:
        _NULL_CACHE[task] = float(store[task])
        return _NULL_CACHE[task]
    with _NULL_COMPUTE_LOCK:
        if task in _NULL_CACHE:
            return _NULL_CACHE[task]
        val = _compute_null_tau(task, g, n_reviews, q)
        with _null_store_lock(p):
            try:
                disk = _read_null_store(p, strict=True)
            except NullStoreUnreadable as e:
                warnings.warn(f"null_tau store unreadable, NOT rewriting it; {task} kept in memory only: {e}", RuntimeWarning, stacklevel=2)
                disk = None
            if disk is not None:
                if task in disk:
                    val = float(disk[task])
                else:
                    disk[task] = val
                    tmp = p.parent / f"{p.name}.tmp.{os.getpid()}.{threading.get_ident()}"
                    try:
                        with open(tmp, "w") as f:
                            json.dump(disk, f, indent=1)
                        os.replace(tmp, p)
                    finally:
                        if tmp.exists():
                            tmp.unlink()
        _NULL_CACHE[task] = val
        return val


def _compute_null_tau(task: str, g: graphs.GStar, n_reviews: int, q: float) -> float:
    from ccbench.adapters import human
    from ccbench.gt import peers as gt_peers

    rng = np.random.default_rng(abs(hash(task)) % (2**32))
    topic = gt_peers.topic_of(task)
    pool = [pid for pid, rec in gt_peers.per_paper().items() if rec.get("topic") != topic and rec.get("has_graph")]
    picks = list(rng.choice(pool, size=min(n_reviews, len(pool)), replace=False))
    sents: list[str] = []
    for pid in picks:
        try:
            hro = human.adapt(pid)
            sents += [s.text for s in hro.report.sentences if len(s.text) >= 20][:400]
        except Exception:
            continue
    claims = [c.text for c in g.claims if c.text]
    if not sents or not claims:
        return float(prereg()["channel"]["claim_match_cosine"])
    M = embed.cosine_matrix(embed.encode(claims), embed.encode(sents))
    return float(np.quantile(M.max(axis=1), q))


def induce(ro: Rollout, g: graphs.GStar, units: Units, tau_floor: float | None = None, window: str = "writing") -> Induced:
    cfg = prereg()["channel"]
    floor = float(cfg["claim_match_cosine"]) if tau_floor is None else float(tau_floor)
    tau = max(floor, null_tau(ro.task, g))
    ind = Induced(system=ro.system if ro.mode in (None, "campaign", "native_chain", "human") else f"{ro.system}.{ro.mode}", task=ro.task, window=window)
    if ro.is_bot or not ro.report or not ro.report.sentences:
        return ind
    sents = [s for s in ro.report.sentences if len(s.text) >= 20]
    claims = [c for c in g.claims if c.text]
    ind.n_sentences = len(sents)
    ind.sent_section = {s.sid: s.section for s in sents}
    ind.sent_paragraph = _paragraph_ids(ro)
    ind.sent_cites = {s.sid: set(s.cites) for s in sents}
    ind.section_order = {sec["id"]: i for i, sec in enumerate(ro.report.sections)}
    top_of: dict[str, str] = {}
    if ro.panel == "H":
        for sec in ro.report.sections:
            top_of[sec["id"]] = g.top_section(sec["id"]) or sec["id"]
    else:
        levels = [sec.get("level", 1) for sec in ro.report.sections if sec.get("title")]
        lmin = min(levels) if levels else 1
        cur = None
        for sec in ro.report.sections:
            if sec.get("title") and sec.get("level", 1) == lmin:
                cur = sec["id"]
            top_of[sec["id"]] = cur or sec["id"]
    seen: dict[str, int] = {}
    for sec in ro.report.sections:
        t = top_of[sec["id"]]
        if t not in seen:
            seen[t] = len(seen)
    ind.top_order = seen
    ind.sent_top = {s.sid: top_of.get(s.section, s.section) for s in sents}
    if sents and claims:
        ce = embed.encode([c.text for c in claims])
        se = embed.encode([s.text for s in sents])
        M = embed.cosine_matrix(ce, se)
        norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
        by_text: dict[str, int] = {}
        for j, s in enumerate(sents):
            by_text.setdefault(norm(s.text), j)
        bs = M.argmax(axis=1)
        bv = M.max(axis=1)
        for i, c in enumerate(claims):
            j = by_text.get(norm(c.text))
            if j is not None:
                ind.claim_match[c.id] = (sents[j].sid, float(M[i, j]))
            elif bv[i] >= tau:
                ind.claim_match[c.id] = (sents[int(bs[i])].sid, float(bv[i]))
        bc = M.argmax(axis=0)
        bcv = M.max(axis=0)
        for j, s in enumerate(sents):
            if bcv[j] >= tau:
                ind.sent_match[s.sid] = (claims[int(bc[j])].id, float(bcv[j]))
        ind.n_matched_sentences = len(ind.sent_match)
        ind.margin_q = {q: float(np.quantile(bcv, float(q))) for q in ("0.25", "0.5", "0.75")}
        sid_idx = {s.sid: j for j, s in enumerate(sents)}
        pairs: set[tuple[str, str]] = set()
        for lst in units.edges_by_type.values():
            pairs.update(lst)
        pairs.update(units.hard_negatives)
        for m in units.motifs.values():
            for path in m:
                pairs.update(zip(path[:-1], path[1:]))
        for a, b in pairs:
            ma, mb = ind.claim_match.get(a), ind.claim_match.get(b)
            if ma and mb:
                ind.pair_cos[f"{a}|{b}"] = float(se[sid_idx[ma[0]]] @ se[sid_idx[mb[0]]])
        ctext = {c.id: c.text for c in claims}
        own = []
        cidx = {c.id: i for i, c in enumerate(claims)}
        for a, b in set().union(*units.edges_by_type.values()) if units.edges_by_type else set():
            if a in cidx and b in cidx:
                own.append(float(ce[cidx[a]] @ ce[cidx[b]]))
        ind.edge_tau = float(np.quantile(own, 0.10)) if own else None
    sys_secs = [(sec["id"], strip_title_number(sec.get("title") or "")) for sec in ro.report.sections if sec.get("title")]
    if sys_secs and units.top_sections:
        claims_in_sys_sec: dict[str, set[str]] = {}
        for cid, (sid, _) in ind.claim_match.items():
            claims_in_sys_sec.setdefault(ind.sent_section.get(sid), set()).add(cid)
        used = set()
        for t in units.top_sections:
            hum = strip_title_number(t["title"]).lower()
            best, best_sid = None, None
            for sid, title in sys_secs:
                if sid in used:
                    continue
                shared = len(set(t["claims"]) & claims_in_sys_sec.get(sid, set()))
                fz = fuzz.token_set_ratio(hum, title.lower())
                if fz >= cfg["section_match_fuzzy"] and shared:
                    score = (fz, shared)
                    if best is None or score > best:
                        best, best_sid = score, sid
            if best_sid is not None:
                ind.section_match[t["id"]] = best_sid
                used.add(best_sid)
    return ind


def induced_path(system_key: str, task: str):
    return paths.out_dir("E10", "induced", system_key) / f"{task}.json"
