#!/usr/bin/env python3
"""Assembles the data root of one profile from a release snapshot: links the stored files, copies the frozen scoring tables
to out/ and a working copy for the scorer to scores/, and writes site.env."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_format as rf

SCORING_PREFIX = "out/"
WORKING_OUT = "scores"
LINKED_DIRS = ("manifests", "splits")


def source_path(snapshot: str, row: dict) -> str:
    rel = rf.safe_relpath(row["path"])
    if row["origin"] == "store":
        return os.path.join(snapshot, "store", rel)
    if row["origin"] == "scoring":
        if not row["path"].startswith(SCORING_PREFIX):
            raise ValueError(f"scoring row outside out/: {row['path']}")
        return os.path.join(snapshot, "scoring", "test", rf.safe_relpath(row["path"][len(SCORING_PREFIX):]))
    raise ValueError(f"origin {row['origin']!r} is not prepared by links or copies")


def link(src: str, dst: str, conflicts: list) -> int:
    if os.path.islink(dst):
        if os.readlink(dst) == src:
            return 0
        os.unlink(dst)
    elif os.path.exists(dst):
        conflicts.append(dst)
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)
    return 1


def copy(src: str, dst: str, conflicts: list, overwrite: bool) -> int:
    if os.path.lexists(dst):
        if os.path.isfile(dst) and not os.path.islink(dst) and filecmp.cmp(src, dst, shallow=False):
            return 0
        if not overwrite:
            conflicts.append(dst)
            return 0
        os.unlink(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    return 1


def site_env(root: str, profile: str, parent: str | None) -> dict:
    env = {"CCBENCH_ROOT": root,
           "CCBENCH_EVAL_SUBDIR": "",
           "CCBENCH_OUT": os.path.join(root, WORKING_OUT),
           "CCBENCH_SHARED_OUT": os.path.join(root, WORKING_OUT),
           "SCRIBE_RUNS_ROOT": os.path.join(root, "runs"),
           "SCRIBE_DATASET_ROOT": os.path.join(root, "data_2000", "dataset_2000"),
           "SCRIBE_BENCH_RESULTS": os.path.join(root, "biolit-bench", "results_2000"),
           "SCRIBE_REF_YEARS": os.path.join(root, "data_2000", "dataset_2000", "manifests", "ref_years.json"),
           "SCRIBE_EVAL_TASKS": os.path.join(root, "data", "pool_corpus", "campaign50.json"),
           "SCRIBE_REHYDRATE_REPORT": os.path.join(root, "rehydrate_report.json")}
    if profile == "test":
        env["SCRIBE_ALLOWLISTS"] = os.path.join(root, "eval", "test", "allowlists")
        env["WINDOWBENCH_MEMBERSHIP"] = os.path.join(root, WORKING_OUT, "readout_membership.json")
    else:
        env["SCRIBE_TRAIN_MANIFEST"] = os.path.join(root, "splits", "train.jsonl")
    if parent:
        env["CCBENCH_PARENT"] = parent
    return env


def write_site_env(root: str, env: dict) -> str:
    dst = os.path.join(root, "site.env")
    lines = [f"# load with: set -a; . {shlex.quote(dst)}; set +a"]
    lines += [f"{k}={shlex.quote(v) if v else ''}" for k, v in env.items()]
    tmp = f"{dst}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, dst)
    return dst


def prepare(snapshot: str, profile: str, root: str, parent: str | None = None, overwrite_scoring: bool = False) -> dict:
    snapshot, root = os.path.abspath(snapshot), os.path.abspath(root)
    real_snap, real_root = os.path.realpath(snapshot), os.path.realpath(root)
    if real_root == real_snap or real_root.startswith(real_snap + os.sep) or real_snap.startswith(real_root + os.sep):
        raise SystemExit(f"--root {root} and --snapshot {snapshot} must not contain each other")
    rows = [r for r in rf.read_originals(snapshot) if profile in rf.parse_profiles(r.get("profiles"))]
    os.makedirs(os.path.join(root, "out"), exist_ok=True)
    os.makedirs(os.path.join(root, WORKING_OUT), exist_ok=True)
    missing, conflicts = [], []
    counts = {"linked": 0, "copied": 0, "seeded": 0, "store_rows": 0, "scoring_rows": 0, "rehydrate_rows": 0}
    for r in rows:
        if r["origin"] == "rehydrate":
            counts["rehydrate_rows"] += 1
            continue
        if r["origin"] == "scoring" and profile != "test":
            continue
        src = source_path(snapshot, r)
        if not os.path.isfile(src):
            missing.append(src)
            continue
        dst = os.path.join(root, rf.safe_relpath(r["path"]))
        if r["origin"] == "store":
            counts["store_rows"] += 1
            counts["linked"] += link(src, dst, conflicts)
        else:
            counts["scoring_rows"] += 1
            counts["copied"] += copy(src, dst, conflicts, overwrite_scoring)
            work = os.path.join(root, WORKING_OUT, rf.safe_relpath(r["path"][len(SCORING_PREFIX):]))
            if not os.path.lexists(work):
                counts["seeded"] += copy(src, work, conflicts, False)
    for name in LINKED_DIRS:
        src = os.path.join(snapshot, name)
        if not os.path.isdir(src):
            missing.append(src)
            continue
        counts["linked"] += link(src, os.path.join(root, name), conflicts)
    env = site_env(root, profile, os.path.abspath(parent) if parent else None)
    summary = {"profile": profile, "root": root, "snapshot": snapshot, **counts,
               "site_env": write_site_env(root, env), "missing_in_snapshot": sorted(missing),
               "conflicts": sorted(conflicts), "ok": not missing and not conflicts}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--profile", choices=rf.PROFILES, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--parent")
    ap.add_argument("--overwrite-scoring", action="store_true")
    a = ap.parse_args(argv)
    summary = prepare(a.snapshot, a.profile, a.root, a.parent, a.overwrite_scoring)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("missing_in_snapshot", "conflicts")}))
    for p in summary["missing_in_snapshot"]:
        print(f"[prepare_root] missing in snapshot: {p}", file=sys.stderr)
    for p in summary["conflicts"]:
        print(f"[prepare_root] not replaced (exists and differs): {p}", file=sys.stderr)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
