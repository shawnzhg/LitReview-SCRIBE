#!/usr/bin/env python3
"""Builds the planning-training candidate list from the training split, excluding the evaluation
tasks, their peers and the training reviews they cite, and writes train_candidates.txt and
eval_excluded.txt. Usage: python build_train_tasks.py [--train_gold <dir>] [--out_dir <dir>] [--check]."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_N = 264


def _env(name: str) -> str:
    return os.environ.get(name) or f"/nonexistent/{name}"


def sha_order(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def ccbench():
    ev = str(ROOT / "evaluation")
    if ev not in sys.path:
        sys.path.insert(0, ev)
    from ccbench import paths
    from ccbench.gt import peers
    return peers, paths


def eval_peers(eval_tasks, peers_mod) -> set:
    return set().union(*(set(peers_mod.peers(t)) for t in sorted(eval_tasks))) - set(eval_tasks)


def eval_reference_pmids(eval_tasks, paths_mod) -> set:
    out = set()
    for t in sorted(eval_tasks):
        refs = json.loads(paths_mod.gt_refs_path(t).read_text())
        out |= {str(r["pmid"]) for r in refs.values() if r.get("pmid")}
    return out


def review_pmid(train_gold: Path, task: str) -> str | None:
    p = train_gold / f"{task}.json"
    rp = json.loads(p.read_text()).get("review_pmid") if p.is_file() else None
    return None if rp is None else str(rp)


def build(train_split: list, eval_tasks: set, peer_set: set, ref_pmids: set, train_gold: Path,
          n: int) -> dict:
    rp = {t: review_pmid(train_gold, t) for t in train_split}
    no_gold = [t for t, v in rp.items() if v is None]
    if no_gold:
        raise SystemExit(f"{len(no_gold)} training tasks have no review_pmid under {train_gold}: "
                         f"{no_gold[:5]}")
    cited = {t for t, v in rp.items() if v in ref_pmids}
    excluded = sorted(((peer_set | cited) & set(train_split)) - eval_tasks)
    drop = eval_tasks | set(excluded)
    pool = sorted((t for t in train_split if t not in drop), key=sha_order)
    if len(pool) < n:
        raise SystemExit(f"pool has {len(pool)} tasks, fewer than N={n}")
    cand = pool[:n]
    bad = set(cand) & (eval_tasks | peer_set | cited)
    if bad:
        raise SystemExit(f"[build_train_tasks] overlap check failed: {sorted(bad)[:5]}")
    return {"candidates": cand, "excluded": excluded,
            "counts": {"train_split": len(train_split), "eval_tasks": len(eval_tasks),
                       "eval_peers": len(peer_set - eval_tasks),
                       "train_reviews_cited_by_eval_tasks": len(cited),
                       "excluded": len(excluded), "pool": len(pool), "candidates": len(cand)}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--train_manifest", default=_env("SCRIBE_TRAIN_MANIFEST"))
    ap.add_argument("--eval_tasks", default=_env("SCRIBE_EVAL_TASKS"))
    ap.add_argument("--train_gold", default=str(Path(_env("SCRIBE_RUNS_ROOT")) / "gold" / "train"),
                    help="gold/train of the training-split root (review PMIDs)")
    ap.add_argument("--out_dir", default=str(ROOT / "configs" / "train_tasks"))
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="candidates that get a base run")
    ap.add_argument("--check", action="store_true", help="recompute and compare with the written lists")
    a = ap.parse_args(argv)
    for k in ("train_manifest", "eval_tasks", "train_gold"):
        if not Path(getattr(a, k)).exists():
            raise SystemExit(f"[build_train_tasks] input {k} not found: {getattr(a, k)}")
    train_split = [json.loads(l)["task_id"] for l in open(a.train_manifest) if l.strip()]
    eval_tasks = set(json.load(open(a.eval_tasks))["tasks"])
    P, paths = ccbench()
    r = build(train_split, eval_tasks, eval_peers(eval_tasks, P), eval_reference_pmids(eval_tasks, paths),
              Path(a.train_gold), a.n)
    cand_txt = "".join(t + "\n" for t in r["candidates"])
    excl_txt = "".join(t + "\n" for t in r["excluded"])
    out = Path(a.out_dir)
    list_p, excl_p = out / "train_candidates.txt", out / "eval_excluded.txt"
    if a.check:
        same = (list_p.is_file() and excl_p.is_file()
                and list_p.read_text() == cand_txt and excl_p.read_text() == excl_txt)
        print(f"[build_train_tasks] --check: {'OK' if same else 'MISMATCH'} "
              f"(candidates sha256 {hashlib.sha256(cand_txt.encode()).hexdigest()[:16]})")
        return 0 if same else 1
    out.mkdir(parents=True, exist_ok=True)
    list_p.write_text(cand_txt)
    excl_p.write_text(excl_txt)
    print(f"[build_train_tasks] {r['counts']}; wrote {list_p} and {excl_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
