"""Prints per-system medians of bundle size, synthesis claim count and full-length report words from
the scorer's outputs. Usage: python mechanism.py --ccbench-out <dir> --row
LABEL=SYSTEM[,SYSTEM...] [--row ...]."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd


def parse_row(spec: str) -> tuple:
    label, sep, members = spec.partition("=")
    names = [m.strip() for m in members.split(",") if m.strip()]
    if not sep or not label.strip() or not names:
        raise SystemExit(f"--row {spec!r}: expected LABEL=SYSTEM[,SYSTEM...]")
    return label.strip(), names


def claims_of(rollouts: Path, system: str, tasks) -> pd.Series:
    out = {}
    for t in tasks:
        p = rollouts / system / f"{t}.json"
        if not p.is_file():
            continue
        g = (json.loads(p.read_text()).get("graph") or {})
        if "nodes" in g:
            out[t] = len(g.get("nodes") or [])
    return pd.Series(out, dtype=float)


def median(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.median()) if len(s) else float("nan")


def member_frame(conf: pd.DataFrame, rollouts: Path, system: str) -> pd.DataFrame:
    c = conf[(conf.system == system) & (conf.status == "ok")].set_index("task")
    f = pd.DataFrame({"papers": c.n_papers.astype(float), "words": c.words.astype(float)})
    f["claims"] = claims_of(rollouts, system, list(f.index)).reindex(f.index)
    return f


def row_stats(conf: pd.DataFrame, rollouts: Path, label: str, members: list) -> dict:
    frames = {m: member_frame(conf, rollouts, m) for m in members}
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    med = {k: median(pd.concat([frames[m].loc[common, k] for m in members], axis=1).mean(axis=1))
           for k in ("papers", "claims", "words")}
    return {"row": label, "members": members, "n_tasks": len(common), "median_papers": med["papers"],
            "median_claims": med["claims"], "median_words": med["words"],
            "by_member": {m: {k: median(frames[m][k].loc[common]) for k in ("papers", "claims", "words")}
                          for m in members}}


def compute(ccbench_out: Path, rows: list) -> list:
    conf = pd.read_csv(Path(ccbench_out) / "E1" / "conformance.csv",
                       usecols=["system", "task", "status", "n_papers", "words"])
    rollouts = Path(ccbench_out) / "rollouts"
    return [row_stats(conf, rollouts, label, members) for label, members in rows]


def fmt(v: float) -> str:
    return "-" if v != v else f"{v:,.1f}"


def report(res: list) -> str:
    lines = [f"{'row':<34}{'tasks':>6}{'papers':>10}{'claims':>10}{'words':>12}"]
    for r in res:
        lines.append(f"{r['row']:<34}{r['n_tasks']:>6}{fmt(r['median_papers']):>10}{fmt(r['median_claims']):>10}"
                     f"{fmt(r['median_words']):>12}")
        if len(r["members"]) > 1:
            for m, v in r["by_member"].items():
                lines.append(f"  {m:<32}{'':>6}{fmt(v['papers']):>10}{fmt(v['claims']):>10}{fmt(v['words']):>12}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ccbench-out", default=os.environ.get("CCBENCH_OUT"))
    ap.add_argument("--row", action="append", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if not a.ccbench_out:
        raise SystemExit("--ccbench-out (or CCBENCH_OUT) is required")
    res = compute(Path(a.ccbench_out), [parse_row(s) for s in a.row])
    print(report(res))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
