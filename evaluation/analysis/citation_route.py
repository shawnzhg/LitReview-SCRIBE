"""Counts the tasks on which a pipeline requested the evaluated review's reference list through the
citation route, with its composite margins with and without them. Usage: python citation_route.py
--pool-logs <runs dir> --gold <dir> --tasks <tasks> --board <board dir> [--system sgi]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
_loaded = sys.modules.get("ccbench")
if _loaded is not None and Path(getattr(_loaded, "__file__", "") or "").resolve().parent.parent != EVAL:
    raise SystemExit(f"ccbench already imported from {_loaded.__file__}, not from {EVAL}")
if str(EVAL) in sys.path:
    sys.path.remove(str(EVAL))
sys.path.insert(0, str(EVAL))
POOL = EVAL.parent / "biolitbench" / "pool"
if str(POOL) not in sys.path:
    sys.path.append(str(POOL))

import pandas as pd

from ccbench import merge_windowbench as MW
from pool_common import resolve_id

ROUTE = "s2_references"


def pmid_of(raw) -> str | None:
    return resolve_id(str(raw or ""))


def read_tasks(path: Path) -> list:
    txt = Path(path).read_text()
    if Path(path).suffix == ".json":
        d = json.loads(txt)
        return sorted(d["tasks"] if isinstance(d, dict) else d)
    return sorted(t for t in txt.split() if t and not t.startswith("#"))


def route_requests(log: Path, review: str, route: str = ROUTE) -> dict:
    n = n_nonempty = 0
    with Path(log).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("route") != route or pmid_of((r.get("params") or {}).get("paper_id")) != review:
                continue
            n += 1
            n_nonempty += int((r.get("n_returned") or 0) > 0)
    return {"calls": n, "calls_nonempty": n_nonempty}


def route_table(pool_logs: Path, gold: Path, tasks: list, route: str = ROUTE) -> pd.DataFrame:
    rows = []
    for t in tasks:
        review = str(json.loads((Path(gold) / f"{t}.json").read_text()).get("review_pmid") or "")
        log = Path(pool_logs) / t / "search_calls.jsonl"
        rec = route_requests(log, review, route) if (review and log.is_file()) else {"calls": None,
                                                                                     "calls_nonempty": None}
        rows.append({"task": t, "review_pmid": review, "log": log.is_file(), **rec})
    return pd.DataFrame(rows)


def margins(board: Path, system: str, tier: str, obs: str, without: list) -> pd.DataFrame:
    z, _ = MW.load(Path(board))
    lb = pd.read_csv(Path(board) / "merged_leaderboard.csv")
    names = lb[(lb.obs == obs) & (lb.tier == tier)].system.tolist()
    if system not in names:
        raise SystemExit(f"{system} is not on the {tier} board at {obs}")
    U, _ = MW.composite(z, obs, names)
    rows = []
    for o in names:
        if o == system:
            continue
        d = (U.loc[system] - U.loc[o]).dropna()
        dr = d[d.index.isin(set(without))]
        rows.append({"obs": obs, "opponent": o, "gap_all": float(d.mean()), "n_all": int(len(d)),
                     "gap_without_route": float(dr.mean()) if len(dr) else float("nan"),
                     "n_without_route": int(len(dr))})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-logs", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--board", default=None)
    ap.add_argument("--system", default="sgi")
    ap.add_argument("--obs", default="system_trunc,system")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    tasks = read_tasks(Path(a.tasks))
    rt = route_table(Path(a.pool_logs), Path(a.gold), tasks)
    missing = rt[~rt.log | rt.calls.isna()]
    req = rt[rt.calls.fillna(0) > 0]
    nonempty = rt[rt.calls_nonempty.fillna(0) > 0]
    without = sorted(set(rt.task) - set(req.task))
    print(f"{a.system}: {ROUTE} on the evaluated review requested on {len(req)} of {len(rt)} tasks "
          f"(answered with papers on {len(nonempty)}); not requested on {len(without)}; logs missing {len(missing)}")
    out = {"route_tasks": req.task.tolist(), "answered_tasks": nonempty.task.tolist(), "without_route": without,
           "n_tasks": int(len(rt)), "missing": missing.task.tolist()}
    if a.board:
        M = pd.concat([margins(Path(a.board), a.system, "self_retrieving", o, without) for o in a.obs.split(",") if o])
        print(f"{'obs':<14}{'opponent':<24}{'gap all':>10}{'n':>4}{'gap w/o route':>15}{'n':>4}")
        for r in M.itertuples():
            print(f"{r.obs:<14}{r.opponent:<24}{r.gap_all:>+10.3f}{r.n_all:>4}{r.gap_without_route:>+15.3f}"
                  f"{r.n_without_route:>4}")
        out["margins"] = M.to_dict(orient="records")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
