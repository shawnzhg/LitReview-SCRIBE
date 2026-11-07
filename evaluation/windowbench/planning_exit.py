"""Scores each system's own outline at the planning exit against the target review's top-level
sections, with the scorer's organisation and outline readouts plus lexical and embedding title F1;
--controls also writes the admission-screen controls: each outline against another task's review,
the target review's own outline and each peer review's outline. Usage: python -m
windowbench.planning_exit [--outlines <allocation_extract out dir>] [--out <file.jsonl>] [--controls]."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import config as C

DROP_TITLES = ("references", "bibliography", "acknowledgements", "acknowledgments")


def _level(n: dict) -> int:
    return int(n["level"]) if n.get("level") is not None else 1


def outline_nodes(nodes: list[dict], cco) -> list[dict]:
    nodes = [n for n in nodes if (n.get("title") or "").strip()]
    while nodes:
        lmin = min(_level(n) for n in nodes)
        at = [n for n in nodes if _level(n) == lmin]
        below = [n for n in nodes if _level(n) == lmin + 1]
        if len(at) == 1 and len(below) >= 2 and cco._norm(at[0].get("title") or "") not in cco.GENERIC:
            nodes = [n for n in nodes if n is not at[0]]
            continue
        break
    return nodes


def top_level(nodes: list[dict], cco) -> list[dict]:
    from ccbench.adapters.common import strip_title_number
    nodes = outline_nodes(nodes, cco)
    if not nodes:
        return []
    lmin = min(_level(n) for n in nodes)
    return [n for n in nodes if _level(n) == lmin
            and strip_title_number(n.get("title") or "").lower().strip(" :.-") not in DROP_TITLES]


def title_f1_lex(sys_titles, hum_titles, cc):
    if not sys_titles or not hum_titles:
        return None
    n, _ = cc["cco"].match_titles(sys_titles, hum_titles, cc["TITLE_FUZZ"])
    p, r = n / len(sys_titles), n / len(hum_titles)
    return 2 * p * r / (p + r) if p + r else 0.0


def title_f1_emb(sys_titles, hum_titles, cc):
    if not sys_titles or not hum_titles:
        return None
    E = cc["embed"]
    S = E.cosine_matrix(E.encode(hum_titles), E.encode(sys_titles))
    pairs = sorted(((float(S[i, j]), i, j) for i in range(len(hum_titles)) for j in range(len(sys_titles))
                    if S[i, j] >= cc["SEC_COS"]), reverse=True)
    uh, us = set(), set()
    for _, i, j in pairs:
        if i in uh or j in us:
            continue
        uh.add(i)
        us.add(j)
    n = len(uh)
    p, r = n / len(sys_titles), n / len(hum_titles)
    return 2 * p * r / (p + r) if p + r else 0.0


def org_size_fit(n_sys: int, n_hum: int):
    return (1 - min(1.0, abs(math.log(n_sys / n_hum)))) if n_sys and n_hum else None


def make_cc() -> dict:
    C.bootstrap_env()
    from ccbench.config import prereg
    from ccbench.readouts import embed
    from ccbench.readouts import outline as ccoutline
    cfg = prereg()
    return {"cco": ccoutline, "embed": embed, "SEC_COS": float(cfg["channel"]["section_match_cosine"]),
            "SEC_FUZZ": float(cfg["channel"]["section_match_fuzzy"]),
            "TITLE_FUZZ": int(cfg["readouts"]["title_match_fuzzy"])}


def _view(task: str, key: str, top: list[dict], nodes: list[dict]):
    from ccbench.model import Context, OutlineNode, Report, Rollout
    from ccbench.readouts.window_views import planning_view
    ro = Rollout(system=key, task=task, panel="A", context=Context(task=task, topic="", cutoff=9999), mode="campaign")
    ro.outline = [OutlineNode(id=str(n.get("id") or i), title=n.get("title") or "", level=_level(n),
                              parent=n.get("parent")) for i, n in enumerate(nodes)]
    ro.report = Report()
    secs = [{"id": str(n.get("id") or f"t{i}"), "title": n.get("title") or "", "parent": None, "claim_ids": []}
            for i, n in enumerate(top)]
    view = planning_view(ro, secs, {}, {})
    view.outline = ro.outline
    return view


def _section_match(view, u, cc):
    from rapidfuzz import fuzz
    from ccbench.adapters.common import strip_title_number
    from ccbench.readouts.channel import Induced
    E = cc["embed"]
    ind = Induced(system=view.system, task=view.task, window="planning")
    ind.section_order = {sec["id"]: i for i, sec in enumerate(view.report.sections)}
    sys_secs = [(sec["id"], strip_title_number(sec.get("title") or "")) for sec in view.report.sections if sec.get("title")]
    if not sys_secs or not u.top_sections:
        return ind, 0
    hum = [strip_title_number(t["title"]) for t in u.top_sections]
    S = E.cosine_matrix(E.encode(hum), E.encode([t for _, t in sys_secs]))
    used = set()
    for i, t in enumerate(u.top_sections):
        best, best_j = None, None
        for j, (sid, title) in enumerate(sys_secs):
            if sid in used:
                continue
            if fuzz.token_set_ratio(hum[i].lower(), title.lower()) >= cc["SEC_FUZZ"] or S[i, j] >= cc["SEC_COS"]:
                if best is None or float(S[i, j]) > best:
                    best, best_j = float(S[i, j]), j
        if best_j is not None:
            ind.section_match[t["id"]] = sys_secs[best_j][0]
            used.add(sys_secs[best_j][0])
    return ind, len(sys_secs)


def score_one(task: str, key: str, outline: list[dict], cc, against: str | None = None) -> dict:
    from ccbench.gt import graphs, units
    from ccbench.readouts import subgraph
    cco = cc["cco"]
    top = top_level(outline, cco)
    if not top:
        return {"task": task, "key": key, "no_outline": True, "n_outline_nodes": len(outline), "readouts": {}}
    g, u = graphs.load(against or task), units.build(against or task)
    view = _view(task, key, top, outline_nodes(outline, cco))
    ind, n_sys = _section_match(view, u, cc)
    rows = {s.name: {"value": s.value, "n_units": s.n_units} for s in subgraph.organisation(view, ind, g, u)}
    hum_top = [cco._norm(t["title"]) for t in u.top_sections]
    sys_top = [cco._norm(sec["title"]) for sec in view.report.sections if sec.get("title")]
    return {"task": task, "key": key, "no_outline": False, "n_sections": n_sys,
            "n_human_sections": len(u.top_sections), "n_outline_nodes": len(outline),
            "title_f1_lex": title_f1_lex(sys_top, hum_top, cc), "title_f1_emb": title_f1_emb(sys_top, hum_top, cc),
            "readouts": rows}


def human_outline(review: str) -> list[dict]:
    from ccbench.gt import units
    return [{"id": s["id"], "title": s["title"], "level": 1, "parent": None} for s in units.build(review).top_sections]


def human_controls(tasks: list[str], cc, ctl) -> None:
    from ccbench.gt import peers
    for t in tasks:
        for review in [t] + peers.peers(t):
            try:
                outline = human_outline(review)
            except FileNotFoundError:
                continue
            r = score_one(t, "human_own" if review == t else f"human:{review}", outline, cc)
            r.update(stage="final", admitted=True, control="human_own" if review == t else "human_peer", source_review=review)
            ctl.write(json.dumps(r) + "\n")
        print(f"{t} human controls scored", flush=True)


def build(outlines: Path, dest: Path, controls: Path | None = None) -> int:
    cc = make_cc()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tasks, ctl = [], None
    if controls:
        from ccbench.ingest import gold
        tasks = gold.campaign50_tasks()
        controls.parent.mkdir(parents=True, exist_ok=True)
        ctl = open(controls, "w")
    wrong = {t: tasks[(i + 1) % len(tasks)] for i, t in enumerate(tasks)}
    n = 0
    with open(dest, "w") as out:
        for fp in sorted(outlines.glob("*.jsonl")):
            key = fp.stem
            for line in open(fp):
                rec = json.loads(line)
                if "error" in rec:
                    out.write(json.dumps({"task": rec["task"], "key": key, "stage": "final", "admitted": False,
                                          "control": "none", "no_outline": True, "readouts": {}}) + "\n")
                    continue
                r = score_one(rec["task"], key, rec.get("outline") or [], cc)
                r.update(stage="final", admitted=True, control="none")
                out.write(json.dumps(r) + "\n")
                n += 1
                if ctl and not r["no_outline"] and rec["task"] in wrong:
                    w = score_one(rec["task"], key, rec.get("outline") or [], cc, against=wrong[rec["task"]])
                    w.update(stage="final", admitted=True, control="wrong_review", source_review=wrong[rec["task"]])
                    ctl.write(json.dumps(w) + "\n")
            print(f"{key} scored", flush=True)
    if ctl:
        human_controls(tasks, cc, ctl)
        ctl.close()
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outlines", default=str(C.OUT_DEFAULT / "allocation"), help="directory of <system>.jsonl written by allocation_extract")
    ap.add_argument("--out", default=None, help="output JSONL (default $WINDOWBENCH_OUTLINE_WINDOW/scores_arms.jsonl)")
    ap.add_argument("--controls", action="store_true", help="also write $WINDOWBENCH_OUTLINE_WINDOW/scores_controls.jsonl")
    a = ap.parse_args(argv)
    dest = Path(a.out) if a.out else C.SOURCES["outline_arms"]
    build(Path(a.outlines), dest, C.SOURCES["outline_controls"] if a.controls else None)


if __name__ == "__main__":
    main()
