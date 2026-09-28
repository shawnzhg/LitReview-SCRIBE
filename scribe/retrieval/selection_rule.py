#!/usr/bin/env python3
"""Selection rule of the evidence bundle: takes the top of the run's own ranking, groups it into
two-year windows before the cutoff, allots the seats proportionally and fills each window by
in-pool citation count with citers strictly before the cutoff."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

VERSION = "selection_rule/1.0"
RULE = "top-1000 own-order gate, 2-year windows at the cutoff, seats ~ window share (largest remainder, " \
       "ties -> recent), fill by cutoff-restricted in-pool citers (exclude_ids removed) desc, ties own rank"
RECORD = "selection_rule_record.json"
RECORD_SCHEMA = "selection_rule_record/1.0"
RANK_BUNDLE = "ranked_bundle.json"
EDGES_DIR = os.environ.get("POOL_EDGES", "")
EDGES_META_SHA256 = "39e57f055dab7e5a6d3390ccbd86640c0b41790a08438633ac68bbf988832ab2"
INDEX_DIR = os.environ.get("POOL_INDEX", "")
INDEX_META_SHA256 = "263f4ee078ad2e007fb489affb975631a718b8ffb7d4bdd9065fa1905f9aeb33"
HERE = str(Path(__file__).resolve().parent)
NOYEAR = -32768
NOYEAR_WINDOW = 10 ** 6
YEAR_LEN = 45_000_001
GATE, WINDOW_YEARS = 1000, 2
SCAN_CAP = 50000
DECISION_REASON = "selection_rule"


class SelectionRuleError(RuntimeError):
    pass


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module_md5():
    return hashlib.md5(Path(__file__).read_bytes()).hexdigest()


def year_dense(index_dir=INDEX_DIR):
    import numpy as np
    meta = json.loads((Path(index_dir) / "meta.json").read_text())
    Y = np.full(YEAR_LEN, NOYEAR, dtype=np.int16)
    mx = 0
    for s in meta["shards"]:
        d = Path(index_dir) / "shards" / s["name"]
        p = np.asarray(np.load(d / "pmid.npy", mmap_mode="r"), dtype=np.int64)
        y = np.load(d / "year.npy", mmap_mode="r")
        mx = max(mx, int(p.max()) if p.size else 0)
        if mx >= len(Y):
            raise SelectionRuleError(f"index pmid {mx} >= dense year length {len(Y)}")
        Y[p] = y
    return Y


class AuthorityStore:

    def __init__(self, edges_dir=EDGES_DIR, index_dir=INDEX_DIR, pin=True):
        import numpy as np
        t0 = time.time()
        self.edges_dir, self.index_dir = str(edges_dir), str(index_dir)
        self.edges_meta_sha256 = sha256_file(Path(edges_dir) / "meta.json")
        self.index_meta_sha256 = sha256_file(Path(index_dir) / "meta.json")
        if pin and self.edges_meta_sha256 != EDGES_META_SHA256:
            raise SelectionRuleError(f"edge store meta sha256 {self.edges_meta_sha256[:16]} != pin {EDGES_META_SHA256[:16]}")
        if pin and self.index_meta_sha256 != INDEX_META_SHA256:
            raise SelectionRuleError(f"index meta sha256 {self.index_meta_sha256[:16]} != pin {INDEX_META_SHA256[:16]}")
        em = json.loads((Path(edges_dir) / "meta.json").read_text())
        self.bi = np.load(Path(edges_dir) / "bwd_indptr.npy", mmap_mode="r")
        self.bv = np.load(Path(edges_dir) / "bwd_val.npy", mmap_mode="r")
        ne = int((em.get("reverse") or {}).get("n_edges") or -1)
        if ne >= 0 and (len(self.bv) != ne or int(self.bi[-1]) != ne):
            raise SelectionRuleError(f"bwd CSR size {len(self.bv)} / indptr end {int(self.bi[-1])} != meta reverse.n_edges {ne}")
        self.Y = year_dense(index_dir)
        self.load_s = round(time.time() - t0, 2)

    def provenance(self):
        return {"edges_dir": self.edges_dir, "edges_meta_sha256": self.edges_meta_sha256,
                "index_dir": self.index_dir, "index_meta_sha256": self.index_meta_sha256,
                "csr": "bwd (citers), mmap", "n_edges": int(len(self.bv))}

    def counts(self, ids, cut, excl_arr):
        import numpy as np
        n = len(ids)
        st = np.zeros(n, np.int64); deg = np.zeros(n, np.int64)
        yr = np.full(n, NOYEAR, np.int16)
        nb = len(self.bi)
        for i, pid in enumerate(ids):
            s = str(pid)
            if not s.isdigit():
                continue
            p = int(s)
            if p < len(self.Y):
                yr[i] = self.Y[p]
            if p + 1 >= nb:
                continue
            lo, hi = int(self.bi[p]), int(self.bi[p + 1])
            if hi <= lo:
                continue
            deg[i] = hi - lo
            c = np.asarray(self.bv[lo:hi], dtype=np.int64)
            if int(c.max()) >= len(self.Y) or int(c.min()) < 0:
                raise SelectionRuleError(f"citer id out of the year range for {p}")
            yy = self.Y[c]
            ok = yy != NOYEAR
            if excl_arr.size:
                ok &= ~np.isin(c, excl_arr)
            st[i] = int((ok & (yy <= cut - 1)).sum())
        return st, deg, yr


_STORE = {}
_STORE_LOCK = threading.Lock()


def get_store(edges_dir=EDGES_DIR, index_dir=INDEX_DIR):
    with _STORE_LOCK:
        key = (str(edges_dir), str(index_dir))
        if key not in _STORE:
            _STORE[key] = AuthorityStore(edges_dir, index_dir)
        return _STORE[key]


def select_papers(order, K, cutoff, exclude_ids=(), gate=GATE, window_years=WINDOW_YEARS, store=None):
    import numpy as np
    K, cut, gate, wy = int(K), int(cutoff), int(gate), int(window_years)
    if K < 0 or gate < 1 or wy < 1:
        raise SelectionRuleError(f"bad parameters K={K} gate={gate} window_years={wy}")
    store = store or get_store()
    excl = {str(x) for x in (exclude_ids or ())}
    C, seen, n_ex, n_dup = [], set(), 0, 0
    for p in order:
        p = str(p)
        if p in excl:
            n_ex += 1
            continue
        if p in seen:
            n_dup += 1
            continue
        seen.add(p)
        C.append(p)
    G = C[:gate]
    n = len(G)
    excl_arr = np.array(sorted(int(x) for x in excl if x.isdigit()), dtype=np.int64)
    v, deg, yr = store.counts(G, cut, excl_arr)
    idx = np.arange(n)
    win = np.where(yr == NOYEAR, NOYEAR_WINDOW, (cut - 1 - yr.astype(np.int64)) // wy)
    windows = []
    if n <= K:
        sel = [int(i) for i in idx]
        mode = "gate_le_K"
    else:
        mode = "stratified"
        ws, cnt = np.unique(win, return_counts=True)
        q = K * cnt / n
        seats = np.floor(q).astype(int)
        rem = K - seats.sum()
        if rem > 0:
            o = np.lexsort((ws, -(q - seats)))
            seats[o[:rem]] += 1
        sel = []
        for w, c_, qq, s in zip(ws, cnt, q, seats):
            ii = idx[win == w]
            ii = ii[np.lexsort((ii, -v[ii]))]
            sel += [int(i) for i in ii[:s]]
            windows.append({"window": int(w), "years": (None if int(w) == NOYEAR_WINDOW else
                                                       [cut - 1 - int(w) * wy - (wy - 1), cut - 1 - int(w) * wy]),
                            "n_gate": int(c_), "quota": float(qq), "seats": int(s),
                            "min_count_selected": int(v[ii[:s]].min()) if s else None})
    P = [G[i] for i in sel]
    selset = set(sel)
    rec = {"version": VERSION, "rule": RULE, "citer_year_rule": "< cutoff",
           "gate": gate, "window_years": wy, "K": K, "cutoff": cut, "mode": mode,
           "n_order": len(order), "n_candidates": len(C), "n_gate": n, "n_dropped_excluded": n_ex, "n_dropped_repeat": n_dup,
           "exclude_ids": sorted(excl), "windows": windows, "n_deg_gt_50k": int((deg > SCAN_CAP).sum()),
           "n_gate_noyear": int((yr == NOYEAR).sum()),
           "gate_papers": [{"id": G[i], "own_rank": int(i), "year": (None if int(yr[i]) == NOYEAR else int(yr[i])),
                            "window": int(win[i]), "citers_strict": int(v[i]),
                            "in_degree": int(deg[i]), "selected": int(i) in selset} for i in range(n)],
           "P": P, "n_P": len(P),
           "provenance": dict(store.provenance(), module=str(Path(__file__).resolve()), module_md5=module_md5()),
           "tie_breaks": "seats: largest fractional part, then smaller window (more recent); fill: count desc, then own "
                         "rank; P order: windows ascending, each in fill order"}
    return P, rec


def _ranker_audit():
    if HERE not in sys.path:
        sys.path.append(HERE)
    import ranker_audit as CA
    return CA


def own_order(unit_dir, R0, union, exclude, pool_size, rank_record=None, trace_rows=None):
    CA = _ranker_audit()
    d = Path(unit_dir)
    rec = rank_record if rank_record is not None else json.loads((d / "ranking_record.json").read_text())
    u = int(rec.get("unit_seed") or 0)
    rows = trace_rows if trace_rows is not None else (CA.jl(d / "trace.jsonl") or [])
    calls, _ = CA.rank_calls(rows)
    nb = -(-int(pool_size) // CA.CR.BATCH) if pool_size else 0
    known = set(union)
    listing, surv, miss = [], [], 0
    for i in range(nb):
        at = calls.get(u + CA.CR.BATCH_SEED_OFFSET + i)
        if not at:
            miss += 1
            continue
        ids, _ = CA.lines_of(at[0]["prompt"])
        listing += ids
        obj, _, _ = CA.replay(at)
        sel, _ = CA.select(obj, known)
        surv += sel
    fin = rec.get("final") or {}
    g = dict(batches_missing=miss, listing_eq_union=set(listing) == set(union), n_listing=len(listing),
             n_surv=len(surv), surv_eq_record=(fin.get("n_candidates") in (None, len(surv))) if fin.get("called") else None)
    own, seen = [], set()
    for seq in (R0, surv, listing, union):
        for p in seq:
            if p not in seen and p not in exclude:
                seen.add(p)
                own.append(p)
    g["own_topK_eq_R0"] = own[:len(R0)] == list(R0)
    return own, g, listing


_TL = threading.local()
_INSTALLED = {}


def config():
    return {"version": VERSION, "rule": RULE, "gate": GATE, "window_years": WINDOW_YEARS,
            "module": str(Path(__file__).resolve()), "module_md5": module_md5(),
            "edges_dir": EDGES_DIR, "edges_meta_sha256": EDGES_META_SHA256, "index_dir": INDEX_DIR,
            "index_meta_sha256": INDEX_META_SHA256}


class _NoOpenTool:

    def __init__(self, tool):
        object.__setattr__(self, "_selection_tool", tool)
        object.__setattr__(self, "_selection_open_calls", [])

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_selection_tool"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_selection_tool"), name, value)

    def open_document(self, pid, step=None):
        object.__getattribute__(self, "_selection_open_calls").append(str(pid))
        return None


def open_and_build(tool, spec, log, P, decision_reason=DECISION_REASON):
    from common import utcnow
    evidence = []
    for i, pid in enumerate(P):
        if tool.remaining("open") == 0:
            log.failure("acquisition", utcnow(), "BudgetExhausted",
                        f"document-open budget spent after {i} of {len(P)} selected",
                        kind="budget_refusal", recovered=True)
            break
        rec = tool.open_document(pid, step=i)
        if not rec or rec.get("refused"):
            continue
        if rec.get("abstract"):
            evidence.append({"evidence_id": f"e_{pid}", "paper_id": pid,
                             "granularity": "abstract", "locator": f"pmid{pid}",
                             "text_hash": tool.docs.span_hash(rec.get("abstract") or ""),
                             "condition": None})
        for s in (rec.get("sentences") or []):
            evidence.append({"evidence_id": f"e_{pid}_{s['locator'].rsplit('_', 1)[1]}",
                             "paper_id": pid, "granularity": "abstract_sentence",
                             "locator": s["locator"],
                             "text_hash": tool.docs.span_hash(s.get("text") or ""),
                             "condition": None})
    papers = []
    for rank, pid in enumerate(P, 1):
        p = tool.pool.get(pid, {})
        papers.append({
            "paper_id": pid, "doi": p.get("doi"), "title": p.get("title"),
            "year": int(p["year"]) if p.get("year") is not None else None,
            "rank": rank, "score": float(p.get("score") or 0.0),
            "first_seen_step": p.get("first_seen_step"),
            "retrieval_provenance": {"query": p.get("query", ""), "tool": "pool_search",
                                     "rank_from_tool": p.get("rank"),
                                     "route": p.get("route") or "both"},
            "decision": "include", "decision_reason": decision_reason,
            "post_cutoff": bool(p.get("year") and int(p["year"]) >= spec["publication_cutoff"])})
    return papers, evidence


def install_sealed(W=None, cfg=None):
    cfg = cfg or config()
    if _INSTALLED.get("sealed"):
        return _INSTALLED["cfg"]
    if W is None:
        import windows as W
    if getattr(W.acquisition, "_query_agent", False) or getattr(W.acquisition, "_cap_budget", False):
        raise SelectionRuleError("install_sealed must run before the query agent / the cap-budget recorder wrap windows")
    if not hasattr(W, "_RANKER_ORIG_ACQ"):
        raise SelectionRuleError("the selection rule needs the ranker installed first (ranker.install)")
    orig_sealed = W._sealed

    def _sealed(artifact, name, log, window, started):
        if getattr(_TL, "defer", False) and name == "evidence_bundle":
            _TL.deferred = True
            return artifact, []
        return orig_sealed(artifact, name, log, window, started)

    _sealed._selection_rule = True
    W._SELECTION_ORIG_SEALED = orig_sealed
    W._sealed = _sealed
    _INSTALLED.update(sealed=True, cfg=cfg)
    return cfg


def install_acquisition(W=None, cfg=None):
    cfg = cfg or config()
    if _INSTALLED.get("acq"):
        return _INSTALLED["cfg"]
    if W is None:
        import windows as W
    if not _INSTALLED.get("sealed") or not hasattr(W, "_SELECTION_ORIG_SEALED"):
        raise SelectionRuleError("install_sealed must run first")
    store = get_store()
    if store.edges_meta_sha256 != cfg["edges_meta_sha256"] or store.index_meta_sha256 != cfg["index_meta_sha256"]:
        raise SelectionRuleError("store provenance != cfg pins")
    orig_acq = W.acquisition
    orig_sealed = W._SELECTION_ORIG_SEALED

    def acquisition(spec, tool, client, log, seed=0):
        from common import seal, utcnow
        unit_dir = Path(log.path).parent
        rec = {"schema": RECORD_SCHEMA, "cfg": {k: v for k, v in cfg.items()}, "task": str(spec.get("task_id")),
               "ok": False, "problems": [], "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        exc = None
        try:
            view = _NoOpenTool(tool)
            _TL.defer, _TL.deferred = True, False
            try:
                inner = orig_acq(spec, view, client, log, seed=seed)
            finally:
                deferred = _TL.deferred
                _TL.defer, _TL.deferred = False, False
            if not deferred:
                raise SelectionRuleError("the inner acquisition did not seal through windows._sealed (selection-rule deferral)")
            b0 = inner[0]
            (unit_dir / RANK_BUNDLE).write_text(json.dumps(b0, indent=1, ensure_ascii=False))
            R0 = [str(p["paper_id"]) for p in sorted(b0.get("papers") or [], key=lambda p: p.get("rank") or 0)]
            opened0 = list(object.__getattribute__(view, "_selection_open_calls"))
            union = [str(x) for x in b0.get("discovered_union") or []]
            exclude = {str(x) for x in (getattr(tool, "exclude_ids", None) or set())}
            cut = int(spec["publication_cutoff"])
            served = getattr(getattr(tool, "tool", None), "cutoff", None)
            g = {"inner_open_calls_eq_R0": opened0 == R0[:len(opened0)] and len(opened0) <= len(R0),
                 "n_inner_open_calls": len(opened0), "cutoff_eq_served": served is None or int(served) == cut}
            if served is not None and int(served) != cut:
                raise SelectionRuleError(f"spec cutoff {cut} != the tool's served cutoff {served}")
            rrec = json.loads((unit_dir / "ranking_record.json").read_text())
            pool_size = int((b0.get("validation") or {}).get("pool_size") or 0)
            order, og, listing = own_order(unit_dir, R0, union, exclude, pool_size, rank_record=rrec)
            g.update(og)
            pool_order = [str(r["paper_id"]) for r in sorted(tool.pool.values(), key=lambda r: -(r.get("score") or 0))]
            g["listing_eq_pool_order"] = listing == pool_order
            g["R0_in_union"] = set(R0) <= set(union)
            g["review_not_in_order"] = not (set(order) & exclude)
            rec["guards"] = g
            bad = [k for k in ("own_topK_eq_R0", "listing_eq_union", "R0_in_union", "review_not_in_order")
                   if g.get(k) is not True]
            if g.get("surv_eq_record") is False:
                bad.append("surv_eq_record")
            if g.get("batches_missing"):
                bad.append("batches_missing")
            if len(set(pool_order)) != len(set(union)) or set(pool_order) != set(union):
                bad.append("pool_eq_union")
            if bad:
                rec["problems"].append(f"own-order guards failed: {bad}")
                raise SelectionRuleError(f"selection-rule own-order guards failed: {bad} ({ {k: g.get(k) for k in bad} })")
            P, erec = select_papers(order, len(R0), cut, exclude, gate=cfg["gate"], window_years=cfg["window_years"])
            rec["order"] = order
            rec["selection"] = erec
            if len(P) != len(R0) or len(set(P)) != len(P) or not set(P) <= set(union) or set(P) & exclude:
                raise SelectionRuleError(f"selection-rule P invariant failed: |P|={len(P)} |R0|={len(R0)} distinct={len(set(P))}")
            papers, evidence = open_and_build(tool, spec, log, P)
            b = dict(b0)
            b.pop("content_hash", None)
            v = dict(b.get("validation") or {})
            v["n_selected"] = len(P)
            v["selection_rule"] = {"version": VERSION, "rule": "top_1000", "gate": cfg["gate"],
                            "window_years": cfg["window_years"], "K": len(R0), "n_P": len(P),
                            "n_P_not_in_R0": len(set(P) - set(R0)), "rank_bundle": RANK_BUNDLE,
                            "rank_bundle_content_hash": b0.get("content_hash"), "record": RECORD,
                            "edges_meta_sha256": erec["provenance"]["edges_meta_sha256"], "module_md5": cfg["module_md5"]}
            b["validation"] = v
            b["papers"], b["evidence"] = papers, evidence
            b["budget_remaining"] = {"search_calls": tool.remaining("search"), "document_opens": tool.remaining("open")}
            status = "budget_exhausted" if (tool.exhausted["search"] or tool.exhausted["open"]) else "done"
            if v.get("stop_reason") == "query_generation_failed" and not papers:
                status = "failed"
            b["retrieval_status"] = status
            b = seal(b)
            out = orig_sealed(b, "evidence_bundle", log, "acquisition", utcnow())
            rec.update(ok=not rec["problems"], R0=R0, P=P, n_P_not_in_R0=len(set(P) - set(R0)),
                       rank_bundle_content_hash=b0.get("content_hash"), bundle_content_hash=b["content_hash"],
                       n_opened=len({e["paper_id"] for e in evidence}), n_evidence=len(evidence))
            return out
        except BaseException as e:
            exc = e
            raise
        finally:
            try:
                if exc is not None:
                    rec["ok"] = False
                    rec["exception"] = f"{type(exc).__name__}: {str(exc)[:500]}"
                rec["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                (unit_dir / RECORD).write_text(json.dumps(rec, indent=1, ensure_ascii=False))
            except Exception as e:
                if exc is None:
                    raise SelectionRuleError(f"cannot write {unit_dir / RECORD}: {e}")

    acquisition._selection_rule = True
    acquisition._selection_rule_inner = orig_acq
    W._SELECTION_ORIG_ACQ = orig_acq
    W.acquisition = acquisition
    _INSTALLED.update(acq=True, cfg=cfg)
    return cfg


def stamp_manifests(R, cfg):
    if getattr(R.manifest, "_selection_rule_stamped", False):
        return
    orig = R.manifest

    def manifest(*a, **k):
        extra = dict(k.get("extra") or {})
        extra.setdefault("selection_rule", VERSION)
        extra.setdefault("selection_rule_module_md5", cfg["module_md5"])
        k["extra"] = extra
        return orig(*a, **k)
    for attr in ("_cap_budget_stamped", "_budget_stamped", "_ranker_stamped", "_query_agent_stamped"):
        if getattr(orig, attr, False):
            setattr(manifest, attr, True)
    manifest._selection_rule_stamped = True
    R.manifest = manifest
