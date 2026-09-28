#!/usr/bin/env python3
"""Audits selection-rule units: recomputes each unit's ranking order and selected papers from its
trace and records, and checks hashes, provenance, opens and exclusions. Usage: python
selection_audit.py --glob <pattern> --out <json>."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "harness" / "runners"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import selection_rule as SR


def jl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().split("\n") if l.strip()] if p.exists() else []


def shadow(d, tmp):
    d = Path(d)
    s = Path(tmp) / d.parent.name / d.name
    s.mkdir(parents=True)
    for f in d.iterdir():
        if f.name == "evidence_bundle.json":
            continue
        (s / f.name).symlink_to(f.resolve())
    (s / "evidence_bundle.json").symlink_to((d / SR.RANK_BUNDLE).resolve())
    return s


def independent_rule(rec, cut, exclude, store, K):
    import numpy as np
    f = []
    cfg = rec.get("cfg") or {}
    gate_n, wy = int(cfg.get("gate") or SR.GATE), int(cfg.get("window_years") or SR.WINDOW_YEARS)
    G, seen = [], set()
    for p in (str(x) for x in rec.get("order") or []):
        if p in exclude or p in seen:
            continue
        seen.add(p)
        G.append(p)
        if len(G) == gate_n:
            break
    gp = (rec.get("selection") or {}).get("gate_papers") or []
    if [str(g.get("id")) for g in gp] != G:
        f.append("recorded gate != the first GATE distinct non-excluded ids of the recorded order")
    bi, bv, Y = store.bi, store.bv, store.Y
    ex = np.array(sorted(int(x) for x in exclude if str(x).isdigit()), dtype=np.int64)
    rows = []
    for r, p in enumerate(G):
        st = 0
        y = None
        if p.isdigit():
            q = int(p)
            if q < len(Y) and int(Y[q]) != SR.NOYEAR:
                y = int(Y[q])
            if q + 1 < len(bi) and int(bi[q + 1]) > int(bi[q]):
                cc = np.asarray(bv[int(bi[q]):int(bi[q + 1])], dtype=np.int64)
                yy = np.asarray(Y[cc], dtype=np.int64)
                keep = (yy != SR.NOYEAR) & ~np.isin(cc, ex)
                st = int((keep & (yy < cut)).sum())
        rows.append((r, p, y, st))
    bad = sum(1 for (r, p, y, st), g in zip(rows, gp) if (g.get("citers_strict"), g.get("year")) != (st, y))
    if bad or len(gp) != len(rows):
        f.append(f"{bad} recorded gate counts / years != the independent recount (citer year < {cut}, review excluded)")
    if len(G) <= K:
        return G, f
    win = {}
    for r, p, y, st in rows:
        win.setdefault(10 ** 6 if y is None else (cut - 1 - y) // wy, []).append((r, p, st))
    ws = sorted(win)
    q = {w: K * len(win[w]) / len(G) for w in ws}
    seats = {w: int(q[w] // 1) for w in ws}
    for w in sorted(ws, key=lambda w: (-(q[w] - seats[w]), w))[:K - sum(seats.values())]:
        seats[w] += 1
    rw = [(int(x.get("window")), int(x.get("seats"))) for x in (rec.get("selection") or {}).get("windows") or []]
    if rw != [(w, seats[w]) for w in ws]:
        f.append("recorded window seats != the independent largest-remainder allocation")
    P = []
    for w in ws:
        P += [p for r, p, v in sorted(win[w], key=lambda t: (-t[2], t[0]))[:seats[w]]]
    return P, f


def audit_unit(d, store, module_md5=None, do_ranking=False, gold_root=None, counter=None):
    from common import content_hash
    d = Path(d)
    f = []
    rec = json.loads((d / SR.RECORD).read_text())
    E = json.loads((d / "evidence_bundle.json").read_text())
    B0 = json.loads((d / SR.RANK_BUNDLE).read_text())
    spec = json.loads((d / "task_spec.json").read_text())
    rb = json.loads((d / "retrieval_budget.json").read_text()) if (d / "retrieval_budget.json").exists() else {}
    exclude = {str(x) for x in (rb.get("exclude_ids") or [])}
    if not exclude:
        f.append("no exclude_ids found for the unit (retrieval_budget.json)")
    cfg = rec.get("cfg") or {}
    if not rec.get("ok") or rec.get("exception") or rec.get("problems"):
        f.append(f"record not ok: {rec.get('problems')} {rec.get('exception', '')}"[:300])
    ve = (E.get("validation") or {}).get("selection_rule") or {}
    if not ve:
        f.append("evidence_bundle validation.selection_rule missing")
    for name, b in (("evidence_bundle", E), ("rank_bundle", B0)):
        x = dict(b); h = x.pop("content_hash", None)
        if content_hash(x) != h:
            f.append(f"{name} content_hash does not recompute")
    if ve.get("rank_bundle_content_hash") != B0.get("content_hash"):
        f.append("validation.selection_rule.rank_bundle_content_hash != ranked_bundle.json content_hash")
    R0 = [str(p["paper_id"]) for p in sorted(B0.get("papers") or [], key=lambda p: p.get("rank") or 0)]
    union = [str(x) for x in B0.get("discovered_union") or []]
    order, g, _ = SR.own_order(d, R0, union, exclude, int((B0.get("validation") or {}).get("pool_size") or 0))
    if order != rec.get("order"):
        f.append("recomputed own order != the recorded order")
    for k in ("own_topK_eq_R0", "listing_eq_union"):
        if g.get(k) is not True:
            f.append(f"own-order guard {k} = {g.get(k)}")
    if g.get("surv_eq_record") is False or g.get("batches_missing"):
        f.append(f"own-order guards surv_eq_record={g.get('surv_eq_record')} batches_missing={g.get('batches_missing')}")
    P_rec = [str(x) for x in rec.get("P") or []]
    P, er = SR.select_papers(rec.get("order") or [], len(R0), int(spec["publication_cutoff"]), exclude,
                           gate=int(cfg.get("gate") or SR.GATE), window_years=int(cfg.get("window_years") or SR.WINDOW_YEARS),
                           store=store)
    P_E = [str(p["paper_id"]) for p in sorted(E.get("papers") or [], key=lambda p: p.get("rank") or 0)]
    if P != P_rec:
        f.append("P recomputed from the recorded order + edge store != the recorded P")
    if P_E != P_rec:
        f.append("evidence_bundle papers != the recorded P (order included)")
    if (rec.get("selection") or {}).get("windows") != er.get("windows"):
        f.append("recorded window seats != recomputed")
    if (rec.get("selection") or {}).get("citer_year_rule") != "< cutoff":
        f.append(f"citer rule {(rec.get('selection') or {}).get('citer_year_rule')!r} != '< cutoff'")
    P_ind, f8 = independent_rule(rec, int(spec["publication_cutoff"]), exclude, store, len(R0))
    f += f8
    if P_ind != P_rec or P_ind != P_E:
        f.append("P re-derived independently != the recorded / sealed P")
    K = int(spec["budget"]["max_ranked_output_K"])
    if len(P_E) != len(R0) or len(P_E) > K:
        f.append(f"|P| {len(P_E)} != |R0| {len(R0)} or > K {K}")
    if len(set(P_E)) != len(P_E):
        f.append("P has repeats")
    if not set(P_E) <= set(union) or [str(x) for x in E.get("discovered_union") or []] != union:
        f.append("P not a subset of discovered_union / union changed")
    if set(P_E) & exclude:
        f.append(f"an excluded id (the review) is in P: {sorted(set(P_E) & exclude)}")
    if gold_root:
        gp = Path(gold_root) / f"{d.parent.name}.json"
        if gp.exists():
            rp = str(json.loads(gp.read_text()).get("review_pmid"))
            if rp in set(P_E):
                f.append(f"the gold review {rp} is in P")
    pc = jl(d / "pool_calls.jsonl")
    kept = set()
    for c in pc:
        if c.get("kind") == "search":
            kept.update(str(x["pmid"]) for x in c.get("kept") or [])
    if pc and not set(P_E) <= kept:
        f.append(f"{len(set(P_E) - kept)} P papers were never kept by a logged search")
    if (rec.get("selection") or {}).get("provenance", {}).get("edges_meta_sha256") != SR.EDGES_META_SHA256 or \
            store.edges_meta_sha256 != SR.EDGES_META_SHA256:
        f.append("edge-store meta sha256 != the pin")
    if module_md5 and cfg.get("module_md5") != module_md5:
        f.append(f"module md5 {cfg.get('module_md5')} != {module_md5}")
    if pc:
        opened = [str(c.get("pmid")) for c in pc if c.get("kind") == "open"]
        if opened != P_E[:len(opened)] or (len(opened) < len(P_E) and E.get("budget_remaining", {}).get("document_opens")):
            f.append(f"pool_calls opens ({len(opened)}) are not P in order")
    qa_path = d / "query_agent_record.json"
    if qa_path.exists():
        qa = json.loads(qa_path.read_text())
        if qa.get("bundle_content_hash") != B0.get("content_hash"):
            f.append("query_agent_record.bundle_content_hash != the rank bundle's (the selection rule did not receive the query agent's bundle)")
        if not qa.get("ok") or qa.get("problems"):
            f.append(f"qa record not ok: {qa.get('problems')}"[:200])
    rrec = json.loads((d / "ranking_record.json").read_text())
    fin = rrec.get("final") or {}
    if fin.get("final_mode") == "tournament" and (fin.get("merged") or [])[:K] != R0:
        f.append("ranking tournament merged[:K] != R0")
    if fin.get("final_mode") == "single" and not fin.get("single_fallback") and (fin.get("selected") or [])[:K] != R0:
        f.append("ranking single selected[:K] != R0")
    if not rrec.get("ok"):
        f.append(f"ranking record not ok: {rrec.get('problems')}"[:200])
    ranking_row = None
    if do_ranking:
        import ranker_audit as CA
        with tempfile.TemporaryDirectory(prefix="selection_shadow_", dir=os.environ.get("TMPDIR")) as tmp:
            s = shadow(d, tmp)
            try:
                r = CA.unit_audit(s, int(rrec.get("unit_seed") or 0), counter)
                ranking_row = {k: r.get(k) for k in ("problems", "agree_problems", "agree", "final_mode", "n_chunks",
                                                 "n_chunks_failed", "share_from_fallback", "record_ok")}
                rest = list(r.get("problems") or []) + list(r.get("agree_problems") or [])
                ranking_row["problems_counted"] = rest
                if rest:
                    f.append(f"ranking audit on the rank bundle: {rest}"[:300])
            except Exception as e:
                f.append(f"ranking audit crashed: {type(e).__name__}: {e}"[:300])
    return {"unit": str(d), "task": d.parent.name, "K": K, "n_R0": len(R0), "n_P": len(P_E),
            "n_P_not_in_R0": len(set(P_E) - set(R0)), "n_order": len(order),
            "own_guards": {k: v for k, v in g.items()}, "ranking": ranking_row, "findings": f}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--module-md5", default=None)
    ap.add_argument("--ranking-audit", action="store_true")
    ap.add_argument("--require", action="store_true")
    ap.add_argument("--min-units", type=int, default=1)
    ap.add_argument("--gold-root", default=None)
    a = ap.parse_args(argv)
    units = sorted(glob.glob(a.glob)) if a.glob else []
    units = [u for u in units if (Path(u) / "evidence_bundle.json").exists()]
    store = SR.get_store()
    counter = None
    if a.ranking_audit:
        import ranker as CR
        counter = CR.PromptCounter()
    rows, findings, non_selection = [], [], []
    for u in units:
        if not (Path(u) / SR.RECORD).exists():
            non_selection.append(u)
            if a.require:
                findings.append(f"{u}: no {SR.RECORD} (not a selection-rule unit)")
            continue
        try:
            r = audit_unit(u, store, a.module_md5, a.ranking_audit, a.gold_root, counter)
        except Exception as e:
            r = {"unit": u, "findings": [f"audit crashed: {type(e).__name__}: {e}"[:300]]}
        rows.append(r)
        findings += [f"{Path(u).parent.name}/{Path(u).name}: {x}" for x in r["findings"]]
    n = len(rows)
    if n < a.min_units:
        findings.append(f"{n} selection-rule units < --min-units {a.min_units}")
    out = {"schema": "selection_audit/1.0", "n_units": n, "n_units_without_record": len(non_selection),
           "n_findings": len(findings), "findings": findings[:200], "edges_meta_sha256": store.edges_meta_sha256,
           "module_md5_expected": a.module_md5, "ranking_shadow_audit": a.ranking_audit, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"### selection-rule audit: {n} units ({len(non_selection)} without the rule); findings {len(findings)}; "
          f"P changed vs R0 median {sorted(r.get('n_P_not_in_R0', 0) for r in rows)[n // 2] if n else None} -> {a.out}",
          flush=True)
    for x in findings[:10]:
        print("###   selection-rule finding:", x[:300], flush=True)
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
