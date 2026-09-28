"""Scores every rollout and every human peer through the observation channel and writes the unit
readout table. Usage: python -m ccbench.score_units [--peers] [--systems ...]."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from ccbench import paths
from ccbench.build import load_rollouts
from ccbench.gt import graphs, peers, units
from ccbench.ingest import gold
from ccbench.model import Rollout
from ccbench.readouts import channel, subgraph


def _key(ro: Rollout) -> str:
    return ro.system if ro.mode in (None, "campaign", "native_chain", "human") else f"{ro.system}.{ro.mode}"


def score_one(ro: Rollout, cache: bool = True) -> list[dict]:
    g = graphs.load(ro.task)
    u = units.build(ro.task)
    key = _key(ro)
    p = channel.induced_path(key, ro.task)
    if cache and p.exists():
        ind = channel.Induced.from_json(p)
    else:
        ind = channel.induce(ro, g, u)
        ind.to_json(p)
    rows = []
    for s in subgraph.all_unit_readouts(ro, ind, g, u):
        rows.append({"system": key, "task": ro.task, "panel": ro.panel, "mode": ro.mode, "readout": s.name, "stage": s.stage, "family": s.family, "stratum": s.stratum, "criterion": s.criterion, "direction": s.direction, "value": s.value, "is_bot": s.value is None, "n_units": s.n_units, "note": s.note, "source_review": ro.meta.get("source_review")})
    return rows


def merge_unit_scores(old_df: pd.DataFrame, new_df: pd.DataFrame, by_system: bool, tasks: list[str] | None) -> pd.DataFrame:
    if by_system:
        drop = old_df["system"].isin(set(new_df["system"].unique()))
        if tasks:
            drop &= old_df["task"].isin(set(tasks))
    else:
        drop = old_df["task"].isin(set(tasks or []))
    return pd.concat([old_df[~drop], new_df], ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--peers", action="store_true")
    ap.add_argument("--systems", default=None)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args(argv)
    tasks = a.tasks.split(",") if a.tasks else None
    rollouts = load_rollouts(a.systems.split(",") if a.systems else None, tasks)
    rows = []
    t0 = time.time()
    for i, ro in enumerate(rollouts):
        rows += score_one(ro, cache=not a.no_cache)
        if i % 25 == 0:
            print(f"[{i+1}/{len(rollouts)}] {_key(ro)} {ro.task} ({time.time()-t0:.0f}s)", flush=True)
    if a.peers:
        from ccbench.adapters import human

        for t in (tasks or gold.campaign50_tasks()):
            for p in [t] + peers.peers(t):
                rows += score_one(human.adapt(p, target_task=t), cache=not a.no_cache)
            print(f"peers scored {t}", flush=True)
    df = pd.DataFrame(rows)
    out = paths.out_dir("E10")
    up = out / "unit_scores.parquet"
    if up.exists() and (a.systems or tasks):
        df = merge_unit_scores(pd.read_parquet(up), df, by_system=bool(a.systems), tasks=tasks)
    df.to_parquet(up, index=False)
    print(f"wrote {out/'unit_scores.parquet'} ({len(df)} rows, {df.readout.nunique()} readouts) in {time.time()-t0:.0f}s")
    print(df[df.panel != "H"].groupby("readout").value.mean().round(3).to_string())


if __name__ == "__main__":
    main()
