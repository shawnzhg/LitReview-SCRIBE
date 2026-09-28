"""Builds the planning-axis readouts, the title-match F1 of top-level section titles against the
target review, and their peer band from the peer reviews. Usage: python -m windowbench.outline_axis
[--systems <keys>] [--out <dir>]."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from . import config as C
from . import roster as R
from .planning_exit import title_f1_emb, title_f1_lex, top_level

OUT = C.OUT_DEFAULT / "outline_axis"
OWN_ROWS = C.SOURCES["outline_arms"]
ROLLOUT_DIR = C.CCB_OUT / "rollouts"
READOUTS = ("outline_title_f1_lex", "outline_title_f1_emb")


def _cc():
    C.bootstrap_env()
    from ccbench.config import prereg
    from ccbench.gt import peers, units
    from ccbench.ingest import gold
    from ccbench.model import Rollout
    from ccbench.readouts import embed
    from ccbench.readouts import outline as ccoutline
    cfg = prereg()
    return dict(units=units, peers=peers, gold=gold, Rollout=Rollout, embed=embed, cco=ccoutline,
                SEC_COS=float(cfg["channel"]["section_match_cosine"]), TITLE_FUZZ=int(cfg["readouts"]["title_match_fuzzy"]))


def rollout_path(system: str, task: str) -> Path | None:
    p = ROLLOUT_DIR / system / f"{task}.json"
    return p if p.exists() else None


def peer_band(tasks: list[str], hum_top: dict, cc) -> list[dict]:
    norm = cc["cco"]._norm
    out = []
    for t in tasks:
        for p in [t] + cc["peers"].peers(t):
            try:
                top = [x for x in (norm(s["title"]) for s in cc["units"].build(p).top_sections) if x]
            except FileNotFoundError:
                continue
            out.append({"task": t, "source_review": p,
                        "outline_title_f1_lex": title_f1_lex(top, hum_top[t], cc),
                        "outline_title_f1_emb": title_f1_emb(top, hum_top[t], cc)})
    return out


def build(systems: list[str] | None = None, verbose: bool = True, out_dir: Path | None = None,
          band: bool = True) -> dict:
    OUT = out_dir or globals()["OUT"]
    cc = _cc()
    norm = cc["cco"]._norm
    tasks = cc["gold"].campaign50_tasks()
    hum_top = {t: [norm(s["title"]) for s in cc["units"].build(t).top_sections] for t in tasks}
    rows = []
    systems = systems or [k for k, v in R.SYSTEMS.items() if v["entries"].get("system") is not None]
    t0 = time.time()
    for s in systems:
        n_ok = n_none = 0
        for t in tasks:
            p = rollout_path(s, t)
            if p is None:
                continue
            ro = cc["Rollout"].from_json(p)
            if ro.is_bot or not ro.report:
                rows.append({"system": s, "task": t, "obs": "system", "no_outline": True, "bot": True,
                             "outline_title_f1_lex": None, "outline_title_f1_emb": None})
                n_none += 1
                continue
            nodes = [{"title": o.title, "level": o.level} for o in ro.outline]
            top = [norm(n["title"]) for n in top_level(nodes, cc["cco"])]
            top = [x for x in top if x]
            if not top:
                rows.append({"system": s, "task": t, "obs": "system", "no_outline": True, "bot": False,
                             "outline_title_f1_lex": None, "outline_title_f1_emb": None, "n_sys_top": 0, "n_hum_top": len(hum_top[t])})
                n_none += 1
                continue
            rows.append({"system": s, "task": t, "obs": "system", "no_outline": False, "bot": False,
                         "outline_title_f1_lex": title_f1_lex(top, hum_top[t], cc),
                         "outline_title_f1_emb": title_f1_emb(top, hum_top[t], cc),
                         "n_sys_top": len(top), "n_hum_top": len(hum_top[t]), "n_outline_nodes": len(nodes)})
            n_ok += 1
        if verbose:
            print(f"  {s:28s} report headings: {n_ok} scored, {n_none} without headings/report  ({time.time()-t0:.0f}s)", file=sys.stderr)
    if OWN_ROWS.exists():
        for line in open(OWN_ROWS):
            d = json.loads(line)
            if d.get("stage") != "final" or not d.get("admitted", True) or d.get("control", "none") != "none" or d.get("hum_task") != d.get("task"):
                continue
            key = d["key"]
            key = key[:-5] if key.endswith(".self") else key
            if key not in R.SYSTEMS:
                continue
            rows.append({"system": key, "task": d["task"], "obs": "planning", "no_outline": bool(d.get("no_outline")), "bot": False,
                         "outline_title_f1_lex": d.get("title_f1_lex"), "outline_title_f1_emb": d.get("title_f1_emb"),
                         "n_sys_top": d.get("n_sections"), "n_hum_top": d.get("n_human_sections")})
    human = peer_band(tasks, hum_top, cc) if band else []
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if band:
        with open(OUT / "human.jsonl", "w") as f:
            for r in human:
                f.write(json.dumps(r) + "\n")
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = df.groupby(["obs", "system"]).agg(n=("task", "nunique"), none=("no_outline", "sum"),
                                              f1_lex=("outline_title_f1_lex", "mean"), f1_emb=("outline_title_f1_emb", "mean")).round(3)
    (OUT / "summary.txt").write_text(summ.to_string())
    if verbose:
        print(summ.to_string())
    return {"rows": rows, "human": human}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=None, help="comma list; default = every roster system with a system window")
    ap.add_argument("--out", default=None, help="output directory (default out/outline_axis)")
    a = ap.parse_args()
    build([x for x in a.systems.split(",") if x] if a.systems else None, out_dir=Path(a.out) if a.out else None)
