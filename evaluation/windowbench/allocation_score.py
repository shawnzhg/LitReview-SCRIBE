"""Scores the extracted evidence allocations against the human review graph, with the same readouts
for the peer reviews. Usage: python -m windowbench.allocation_score [--out <dir>]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

READOUTS = ("alloc_coloc_recall", "alloc_coloc_precision", "alloc_coloc_f1", "alloc_pair_agreement", "alloc_ari", "alloc_evidence_placed", "alloc_spread")


def _cc():
    C.bootstrap_env()
    from ccbench.adapters.common import strip_title_number
    from ccbench.config import prereg
    from ccbench.gt import graphs, units
    from ccbench.ingest import gold
    from rapidfuzz import fuzz
    from ccbench.readouts.outline import GENERIC
    return dict(graphs=graphs, units=units, gold=gold, fuzz=fuzz, strip=strip_title_number,
                GENERIC=GENERIC, FUZZ=int(prereg()["channel"]["section_match_fuzzy"]))


def human_reference(cc, task: str) -> tuple[dict[str, set], list[dict]]:
    g, u = cc["graphs"].load(task), cc["units"].build(task)
    tops = [{"id": s["id"], "title": s["title"]} for s in u.top_sections]
    ids = {s["id"] for s in tops}
    by: dict[str, set] = {s["id"]: set() for s in tops}
    for c in g.claims:
        top = g.top_section(c.section_id)
        if top in ids:
            by[top].update(str(p) for p in c.pmids)
    return by, tops


def top_of(nodes: list[dict], cc) -> tuple[dict, dict]:
    ns = [n for n in nodes if (n.get("title") or "").strip()]
    if not ns:
        return {}, {}
    lvl = lambda n: int(n["level"]) if n.get("level") is not None else 1
    while True:
        lmin = min(lvl(n) for n in ns)
        at = [n for n in ns if lvl(n) == lmin]
        below = [n for n in ns if lvl(n) == lmin + 1]
        if len(at) == 1 and len(below) >= 2 and cc["strip"](at[0].get("title") or "").lower().strip() not in cc["GENERIC"]:
            ns = [n for n in ns if n is not at[0]]
            continue
        break
    lmin = min(lvl(n) for n in ns)
    by_id = {str(n["id"]): n for n in ns}
    tops = {str(n["id"]): (n.get("title") or "") for n in ns if lvl(n) == lmin}
    anc = {}
    for n in ns:
        cur, seen = n, 0
        while str(cur["id"]) not in tops and cur.get("parent") is not None and seen < 20:
            nxt = by_id.get(str(cur["parent"]))
            if nxt is None:
                break
            cur, seen = nxt, seen + 1
        anc[str(n["id"])] = str(cur["id"]) if str(cur["id"]) in tops else None
    return anc, tops


def score_allocation(cc, alloc: dict[str, list], papers: set, task: str, outline: list | None = None) -> dict:
    R, tops_h = human_reference(cc, task)
    P = set(papers)
    norm = lambda s: cc["strip"](s or "").lower().strip()
    if outline:
        anc, tops_s = top_of(outline, cc)
        title_to_id = {norm(v.get("title") or ""): str(v["id"]) for v in outline if (v.get("title") or "").strip()}
        agg: dict[str, set] = {t: set() for t in tops_s}
        for k, v in alloc.items():
            nid = str(k) if str(k) in anc else title_to_id.get(norm(k))
            top = anc.get(nid) if nid else None
            if top:
                agg[top].update(v)
        sys_secs = [(t, norm(tops_s[t])) for t in tops_s]
        alloc = {t: sorted(agg[t]) for t in tops_s}
    else:
        sys_secs = [(k, norm(k)) for k in alloc]
    human_sec = {}
    for h in tops_h:
        for pm in (R[h["id"]] & P):
            human_sec.setdefault(pm, set()).add(h["id"])
    sys_sec = {}
    for k, v in alloc.items():
        for pm in set(v) & P:
            sys_sec.setdefault(pm, set()).add(k)
    shared = sorted(set(human_sec) & set(sys_sec))
    tp = fp = fn = tn = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            a_, b_ = shared[i], shared[j]
            h_same = bool(human_sec[a_] & human_sec[b_])
            s_same = bool(sys_sec[a_] & sys_sec[b_])
            if h_same and s_same: tp += 1
            elif s_same and not h_same: fp += 1
            elif h_same and not s_same: fn += 1
            else: tn += 1
    n_pairs = tp + fp + fn + tn
    coloc_rec = tp / (tp + fn) if (tp + fn) else np.nan
    coloc_pre = tp / (tp + fp) if (tp + fp) else np.nan
    coloc_f1 = (2 * coloc_rec * coloc_pre / (coloc_rec + coloc_pre)) if (coloc_rec == coloc_rec and coloc_pre == coloc_pre and coloc_rec + coloc_pre > 0) else (0.0 if n_pairs else np.nan)
    agree = (tp + tn) / n_pairs if n_pairs else np.nan
    exp = ((tp + fp) * (tp + fn) / n_pairs) if n_pairs else np.nan
    maxi = ((tp + fp) + (tp + fn)) / 2 if n_pairs else np.nan
    ari = ((tp - exp) / (maxi - exp)) if (n_pairs and maxi != exp) else np.nan
    reach = set()
    for h in tops_h:
        reach |= (R[h["id"]] & P)
    placed_any = {pm for pm in reach if pm in sys_sec}
    placed = [x for v in alloc.values() for x in v]
    spread = (len(placed) / len({*placed})) if placed else np.nan
    return {"alloc_coloc_recall": coloc_rec, "alloc_coloc_precision": coloc_pre, "alloc_coloc_f1": coloc_f1,
            "alloc_pair_agreement": agree, "alloc_ari": ari,
            "alloc_evidence_placed": (len(placed_any) / len(reach)) if reach else np.nan,
            "alloc_spread": spread,
            "n_shared_papers": len(shared), "n_pairs": n_pairs, "n_reachable": len(reach),
            "n_sys_sections": len(alloc), "n_human_sections": len(tops_h), "n_papers": len(P)}


def human_peer_rows(cc, tasks: list[str]) -> pd.DataFrame:
    from ccbench.gt import peers as PE
    rows = []
    for t in tasks:
        try:
            peer_ids = PE.peers(t)
        except Exception:
            peer_ids = []
        for p in list(peer_ids) + [t]:
            try:
                Rp, topsp = human_reference(cc, p)
            except Exception:
                continue
            alloc = {d["title"]: sorted(Rp[d["id"]]) for d in topsp}
            papers = {x for v in alloc.values() for x in v}
            r = score_allocation(cc, alloc, papers, t, None)
            rows.append({"task": t, "source_review": p, "is_self": p == t, **r})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.OUT_DEFAULT / "allocation"))
    a = ap.parse_args(argv)
    cc = _cc()
    out = Path(a.out)
    rows = []
    for fp in sorted(out.glob("*.jsonl")):
        arm = fp.stem
        for line in open(fp):
            d = json.loads(line)
            if d.get("error"):
                continue
            r = score_allocation(cc, d["allocation"], set(d["papers"]), d["task"], d.get("outline") or None)
            rows.append({"system": arm, "task": d["task"], **r})
    S = pd.DataFrame(rows)
    S.to_csv(out / "allocation_scores.csv", index=False)
    tasks = sorted(S.task.unique())
    H = human_peer_rows(cc, tasks)
    H.to_csv(out / "allocation_human.csv", index=False)
    print(f"wrote {out}/allocation_scores.csv, allocation_human.csv")


if __name__ == "__main__":
    main()
