"""Scores the readouts at each window view, including the length-matched report of systems and peers,
and writes the window score table. Usage: python -m ccbench.score_windows [--peers] [--truncate]
[--systems ...]."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from ccbench import paths
from ccbench.build import load_rollouts
from ccbench.gt import graphs, peers, units
from ccbench.ingest import gold
from ccbench.readouts import window_views as WV


def _key(ro) -> str:
    return ro.system if ro.mode in (None, "campaign", "native_chain", "human") else f"{ro.system}.{ro.mode}"


def relabel_final(unit_scores: pd.DataFrame) -> pd.DataFrame:
    out = []
    ret = unit_scores[unit_scores.readout.str.startswith("ret_")].copy()
    ret["window"] = "retrieval"
    out.append(ret)
    for w in ("writing", "system"):
        d = unit_scores.copy()
        d["window"] = w
        out.append(d)
    return pd.concat(out, ignore_index=True)


def merge_selected_systems(old: pd.DataFrame, df: pd.DataFrame, sel: list[str], tasks: list[str] | None) -> pd.DataFrame:
    drop = old.system.isin(set(df.system.unique()) - set(old.system.unique()) | set(sel))
    if tasks:
        drop &= old.task.isin(set(tasks))
    return pd.concat([old[~drop], df[df.system.isin(sel)]], ignore_index=True).drop_duplicates(["system", "task", "window", "readout"], keep="last")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--peers", action="store_true")
    ap.add_argument("--windows", default="synthesis,planning")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--systems", default=None, help="only these agent keys; their rows replace theirs in the existing table")
    ap.add_argument("--truncate", action="store_true", help="also score the length-matched view (window system_trunc) of every system and peer")
    a = ap.parse_args(argv)
    windows = a.windows.split(",")
    if a.truncate and "system_trunc" not in windows:
        windows.append("system_trunc")
    explicit_tasks = bool(a.tasks)
    tasks = a.tasks.split(",") if a.tasks else gold.campaign50_tasks()
    out = paths.out_dir("E13")
    rows = []
    t0 = time.time()
    sel = a.systems.split(",") if a.systems else (None if a.truncate else paths.agent_keys())
    for ro in load_rollouts(sel, tasks):
        if ro.panel not in ("A", "B"):
            continue
        g = graphs.load(ro.task)
        u = units.build(ro.task)
        for w in windows:
            if w == "system_trunc":
                if ro.is_bot:
                    continue
                view, links = WV.truncated_view(ro), set()
            elif ro.panel == "A":
                continue
            else:
                try:
                    view, links = WV.agent_view(ro, w)
                except FileNotFoundError:
                    view = None
            if view is None:
                rows.append({"system": _key(ro), "task": ro.task, "panel": "B", "mode": ro.mode, "window": w, "readout": "completion", "stage": "all", "family": "task", "stratum": "all", "criterion": "completion", "direction": "quality", "value": 0.0, "is_bot": False, "n_units": 1, "note": "no_exit_artifact", "source_review": None})
                continue
            rows += WV.score_view(view, links, w, g, u, _key(ro), cache=not a.no_cache)
        print(f"{_key(ro):22s} {ro.task} ({time.time()-t0:.0f}s)", flush=True)
    if a.peers:
        from ccbench.adapters import human

        for t in tasks:
            g = graphs.load(t)
            u = units.build(t)
            for p in [t] + peers.peers(t):
                hro = human.adapt(p, target_task=t)
                for w in windows:
                    view, links = WV.human_view(hro, w)
                    if view is None:
                        continue
                    rows += WV.score_view(view, links, w, g, u, _key(hro), cache=not a.no_cache)
            print(f"peers scored {t} ({time.time()-t0:.0f}s)", flush=True)
    new = pd.DataFrame(rows)
    dest = out / "window_scores.parquet"
    e10 = pd.read_parquet(paths.OUT / "E10" / "unit_scores.parquet")
    if explicit_tasks:
        e10 = e10[e10.task.isin(tasks)]
    if a.systems and dest.exists():
        df = merge_selected_systems(pd.read_parquet(dest), pd.concat([relabel_final(e10), new], ignore_index=True), sel, tasks if explicit_tasks else None)
    elif dest.exists():
        old = pd.read_parquet(dest)
        old = old[~(old.window.isin(windows) & old.task.isin(tasks))]
        df = pd.concat([old, relabel_final(e10), new], ignore_index=True).drop_duplicates(["system", "task", "window", "readout"], keep="last")
    else:
        df = pd.concat([relabel_final(e10), new], ignore_index=True)
    df.to_parquet(dest, index=False)
    print(f"wrote {dest} ({len(df)} rows; new window rows {len(new)}) in {time.time()-t0:.0f}s")
    if len(new):
        print(new[new.panel != "H"].groupby(["window", "readout"]).value.mean().round(3).to_string())


if __name__ == "__main__":
    main()
