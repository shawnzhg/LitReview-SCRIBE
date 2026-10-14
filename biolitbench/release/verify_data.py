#!/usr/bin/env python3
"""Checks a data root against the release manifests and the rehydration report, writes verify_report.json and lists every failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_format as rf

VERIFY_SCHEMA = "biolit-verify-report/1"
TEST_SET_SHA16 = "cae13eb0c9a0767d"
CHECKS = ("snapshot", "closure", "file_gate", "units", "report", "wrappers", "graph_set", "splits", "cutoff",
          "null_tau", "parent")
EXTRA_REQUIRED = {"test": ("manifests/original_sha256.tsv", "rehydrate_report.json"),
                  "train": ("manifests/original_sha256.tsv", "rehydrate_report.json", "splits/train.jsonl")}
BUNDLE_KIND = {"test": "bundle_test", "train": "bundle_train"}
GRAPHS = "data_2000/dataset_2000/graphs"
NULL_TAU = "out/E10/null_tau.json"
HUB_FILES = (".gitattributes",)
HUB_DIRS = (".git", ".cache", ".huggingface")
WRAPPER_HEAD = re.compile(rb'^\s*\{\s*"format"\s*:\s*"biolit-dehydrated/')
PRINT_LIMIT = 200


class Checks:
    def __init__(self):
        self.res = {c: {"ok": True, "n_checked": 0, "failures": []} for c in CHECKS}

    def fail(self, check: str, msg: str) -> None:
        self.res[check]["ok"] = False
        self.res[check]["failures"].append(msg)

    def count(self, check: str, n: int = 1) -> None:
        self.res[check]["n_checked"] += n

    def skip(self, check: str, why: str) -> None:
        self.res[check]["skipped"] = why

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.res.values())


def file_digest(path: str):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            head = f.read(512)
            h.update(head)
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest(), head
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None, b""


def digests(paths: list, workers: int) -> dict:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return dict(zip(paths, ex.map(file_digest, paths)))


def read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_ids(path: str) -> set:
    with open(path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def json_stems(path: str) -> set:
    return {n[:-5] for n in os.listdir(path) if n.endswith(".json")} if os.path.isdir(path) else set()


def sample(ids) -> str:
    ids = sorted(ids)
    return ", ".join(ids[:5]) + (f" (+{len(ids) - 5} more)" if len(ids) > 5 else "")


def set_diff(name: str, got: set, want: set) -> str | None:
    if got == want:
        return None
    parts = []
    if want - got:
        parts.append(f"missing {len(want - got)}: {sample(want - got)}")
    if got - want:
        parts.append(f"unexpected {len(got - want)}: {sample(got - want)}")
    return f"{name}: " + "; ".join(parts)


def snapshot_dir(root: str, given: str | None) -> str | None:
    if given:
        return os.path.abspath(given)
    link = os.path.join(root, "manifests")
    return os.path.dirname(os.path.realpath(link)) if os.path.islink(link) else None


def check_snapshot(C: Checks, snap: str, workers: int) -> None:
    man = os.path.join(snap, "MANIFEST.sha256")
    if not os.path.isfile(man):
        C.fail("snapshot", "MANIFEST.sha256 missing")
        return
    listed = {}
    with open(man, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            sha, sep, rel = line.partition("  ")
            if not sep or not re.fullmatch(r"[0-9a-f]{64}", sha) or not rel:
                C.fail("snapshot", f"MANIFEST.sha256 line {n} is malformed")
                continue
            listed[rel] = sha
    got = digests([os.path.join(snap, rf.safe_relpath(p)) for p in listed], workers)
    for rel, sha in listed.items():
        C.count("snapshot")
        have = got[os.path.join(snap, rf.safe_relpath(rel))][0]
        if have is None:
            C.fail("snapshot", f"{rel}: listed in MANIFEST.sha256 but missing")
        elif have != sha:
            C.fail("snapshot", f"{rel}: sha256 differs from MANIFEST.sha256")
    for dirpath, dirnames, filenames in os.walk(snap):
        rel_dir = os.path.relpath(dirpath, snap)
        if rel_dir == ".":
            dirnames[:] = [d for d in dirnames if d not in HUB_DIRS]
        for name in filenames:
            rel = name if rel_dir == "." else os.path.join(rel_dir, name)
            if rel == "MANIFEST.sha256" or (rel_dir == "." and name in HUB_FILES):
                continue
            if rel not in listed:
                C.fail("snapshot", f"{rel}: in the snapshot but not listed in MANIFEST.sha256")


def load_report(C: Checks, root: str, profile: str, tsv_sha: str | None, required: set):
    path = os.path.join(root, "rehydrate_report.json")
    C.count("report")
    if not os.path.isfile(path):
        C.fail("report", "rehydrate_report.json missing (run biolitbench/release/rehydrate.py)")
        return None
    try:
        rep = read_json(path)
    except ValueError:
        C.fail("report", "rehydrate_report.json is not valid JSON")
        return None
    if not isinstance(rep, dict) or rep.get("schema") != rf.REPORT_SCHEMA:
        C.fail("report", f"rehydrate_report.json schema is not {rf.REPORT_SCHEMA}")
        return None
    if os.path.realpath(str(rep.get("root") or "")) != os.path.realpath(root):
        C.fail("report", f"rehydrate_report.json root {rep.get('root')!r} is not this root")
    if rep.get("manifest_sha256") != tsv_sha:
        C.fail("report", "rehydrate_report.json manifest_sha256 differs from manifests/original_sha256.tsv")
    if rep.get("profile") != profile:
        C.fail("report", f"rehydrate_report.json profile {rep.get('profile')!r} is not {profile!r}")
    not_fetched = (rep.get("abstracts") or {}).get("not_fetched")
    if not_fetched is None:
        C.fail("report", "rehydrate_report.json has no abstracts.not_fetched count (rerun biolitbench/release/rehydrate.py)")
    elif not_fetched:
        C.fail("report", f"{not_fetched} abstracts were not fetched (rerun biolitbench/release/fetch_abstracts.py)")
    for f in rep.get("files") or []:
        if f.get("path") in required and f.get("status") in ("failed", "not_rehydratable"):
            C.fail("report", f"{f.get('path')}: rehydrate status {f.get('status')} ({f.get('reason')})")
    return rep


def unit_values(root: str, snap: str | None, row: dict, obj) -> dict:
    kind, task, path = row["kind"], row["task_id"], row["path"]
    units = {}
    if kind == "baseline_refs":
        if snap is None:
            raise ValueError("needs --snapshot: the baseline unit digests are in the dehydrated wrapper")
        w = read_json(os.path.join(snap, "dehydrated", rf.safe_relpath(path)))
        refs, phs = obj.get("refs") or [], (w.get("data") or {}).get("refs") or []
        if len(refs) != len(phs):
            raise ValueError("reference count differs from the wrapper")
        for r, ph in zip(refs, phs):
            if str(r.get("pmid")) != str(ph.get("pmid")):
                raise ValueError("reference order differs from the wrapper")
            ab = ph.get("abstract")
            if isinstance(ab, dict) and "sha256" in ab:
                units[(f"pmid{r.get('pmid')}", "baseline_abstract")] = (r.get("abstract"), "sha256:" + ab["sha256"])
        return units
    bundle = read_json(os.path.join(root, rf.contract_path(rf.UNIT_BUNDLE[kind], task)))
    if kind == "oracle_sentences":
        sealed = {}
        for e in bundle.get("evidence") or []:
            sealed.setdefault(e.get("locator"), (e.get("granularity"), e.get("text_hash")))
        for loc, v in obj.items():
            if loc not in sealed:
                raise ValueError(f"{loc}: no sealed digest in the oracle bundle")
            units[(loc, sealed[loc][0])] = (v, sealed[loc][1])
        return units
    for e in bundle.get("evidence") or []:
        entry = obj.get(str(e.get("paper_id"))) or {}
        if e.get("granularity") == "abstract":
            v = entry.get("abstract")
        elif e.get("granularity") == "abstract_sentence":
            v = (entry.get("sentences") or {}).get(e.get("locator"))
        else:
            continue
        units[(e.get("locator"), e.get("granularity"))] = (v, e.get("text_hash"))
    return units


def reverify(root: str, snap: str | None, row: dict, listed: dict) -> list:
    obj = read_json(os.path.join(root, rf.safe_relpath(row["path"])))
    if rf.is_wrapper(obj):
        return ["still a dehydrated wrapper"]
    try:
        units = unit_values(root, snap, row, obj)
    except (ValueError, KeyError, TypeError, OSError) as e:
        return [f"cannot re-verify units: {e}"]
    errs = []
    for key, (v, h) in sorted(units.items()):
        got = None if v is None else rf.text_hash(v)
        if key in listed:
            if got != listed[key]:
                errs.append(f"{key[0]} ({key[1]}): text does not hash to the report's written_text_hash")
        elif got is None or got != h:
            errs.append(f"{key[0]} ({key[1]}): differs from its sealed digest and the report does not list it")
    for key in sorted(set(listed) - set(units)):
        errs.append(f"{key[0]} ({key[1]}): listed in the report but not a unit of this file")
    return errs


def verify(root: str, profile: str, snapshot: str | None = None, parent: str | None = None, workers: int = 8) -> dict:
    C = Checks()
    root = os.path.abspath(root)
    snap = snapshot_dir(root, snapshot)
    if snapshot:
        check_snapshot(C, os.path.abspath(snapshot), workers)
    else:
        C.skip("snapshot", "no --snapshot")
    tsv = os.path.join(root, "manifests", "original_sha256.tsv")
    if not os.path.isfile(tsv):
        C.fail("closure", "manifests/original_sha256.tsv missing (run biolitbench/release/prepare_root.py)")
        return finish(C, root, profile, snap)
    tsv_sha = rf.file_sha256(tsv)
    rows = rf.read_tsv(tsv, rf.ORIGINAL_COLUMNS)
    sources = os.path.join(root, "manifests", "article_sources.tsv")
    nr = set()
    if os.path.isfile(sources):
        nr = {r["task_id"] for r in rf.read_tsv(sources, rf.SOURCE_COLUMNS) if not rf.flag(r.get("rehydratable", "true"))}
    else:
        C.fail("closure", "manifests/article_sources.tsv missing")
    req = [r for r in rows if profile in rf.parse_profiles(r.get("profiles"))
           and not (r["kind"] in rf.REVIEW_KINDS and r["task_id"] in nr)]
    required = [r["path"] for r in req] + list(EXTRA_REQUIRED[profile])
    present = set()
    for p in required:
        C.count("closure")
        if os.path.exists(os.path.join(root, rf.safe_relpath(p))):
            present.add(p)
        else:
            C.fail("closure", f"{p}: missing")

    gate_rows = [r for r in req if r["path"] in present and (r["origin"] in ("store", "scoring") or r["kind"] in rf.REVIEW_KINDS)]
    unit_rows = [r for r in req if r["path"] in present and r["origin"] == "rehydrate" and r["kind"] in rf.ABSTRACT_KINDS]
    got = digests([os.path.join(root, rf.safe_relpath(r["path"])) for r in gate_rows + unit_rows], workers)

    def sha(r):
        return got[os.path.join(root, rf.safe_relpath(r["path"]))][0]

    mismatched = set()
    for r in gate_rows:
        C.count("file_gate")
        if sha(r) != r["sha256"]:
            mismatched.add(r["path"])
            C.fail("file_gate", f"{r['path']}: sha256 differs from manifests/original_sha256.tsv")

    rep = load_report(C, root, profile, tsv_sha, set(required))
    drift_files = {f.get("path") for f in (rep or {}).get("files") or [] if f.get("status") == "drift"}
    listed = {}
    for d in (rep or {}).get("drift") or []:
        listed.setdefault(d.get("path"), {})[(d.get("locator"), d.get("granularity"))] = d.get("written_text_hash")
    for r in unit_rows:
        C.count("units")
        if sha(r) == r["sha256"]:
            continue
        mismatched.add(r["path"])
        if r["path"] not in drift_files:
            C.fail("units", f"{r['path']}: sha256 differs and rehydrate_report.json does not list the file as drift")
            continue
        for e in reverify(root, snap, r, listed.get(r["path"], {})):
            C.fail("units", f"{r['path']}: {e}")

    for r in gate_rows + unit_rows:
        if not r["path"].endswith(".json"):
            continue
        C.count("wrappers")
        wrapped = bool(WRAPPER_HEAD.match(got[os.path.join(root, rf.safe_relpath(r["path"]))][1]))
        if not wrapped and r["path"] in mismatched:
            try:
                wrapped = rf.is_wrapper(read_json(os.path.join(root, rf.safe_relpath(r["path"]))))
            except ValueError:
                wrapped = False
        if wrapped:
            C.fail("wrappers", f"{r['path']}: dehydrated wrapper at a contract path (run biolitbench/release/rehydrate.py)")

    splits = os.path.join(root, "splits")
    try:
        test_ids = read_ids(os.path.join(splits, "test.txt"))
        train_ids = read_ids(os.path.join(splits, "train.txt"))
        peers = {r["peer_id"] for r in rf.read_tsv(os.path.join(splits, "test_peers.tsv"), ("task_id", "peer_id"))}
    except OSError as e:
        C.fail("splits", f"split lists unreadable: {e}")
        return finish(C, root, profile, snap)

    want_graphs = (test_ids | peers) if profile == "test" else (train_ids - nr)
    manifest_graphs = {r["task_id"] for r in req if r["kind"] == "graph"}
    gdir = os.path.join(root, GRAPHS)
    have_graphs = {t for t in os.listdir(gdir) if os.path.exists(os.path.join(gdir, t, "graph_enhanced.json"))} \
        if os.path.isdir(gdir) else set()
    C.count("graph_set", len(have_graphs))
    for msg in (set_diff("manifest graph rows vs splits", manifest_graphs, want_graphs),
                set_diff("graphs in the root", have_graphs, want_graphs)):
        if msg:
            C.fail("graph_set", msg)

    dev_dir, train_dir = os.path.join(root, "runs/taskspecs/dev"), os.path.join(root, "runs/taskspecs/train")
    if profile == "test":
        C.count("splits", 4)
        if hashlib.sha256("\n".join(sorted(test_ids)).encode()).hexdigest()[:16] != TEST_SET_SHA16:
            C.fail("splits", f"splits/test.txt is not the released test set (sha256_16 != {TEST_SET_SHA16})")
        for msg in (set_diff("runs/taskspecs/dev vs splits/test.txt", json_stems(dev_dir), test_ids),
                    set_diff("runs/taskspecs/train vs test peers", json_stems(train_dir), peers)):
            if msg:
                C.fail("splits", msg)
        try:
            camp = set(read_json(os.path.join(root, "data/pool_corpus/campaign50.json"))["tasks"])
            msg = set_diff("campaign50.json tasks vs splits/test.txt", camp, test_ids)
            if msg:
                C.fail("splits", msg)
        except (OSError, ValueError, KeyError, TypeError) as e:
            C.fail("splits", f"data/pool_corpus/campaign50.json unreadable: {type(e).__name__}")
    else:
        C.count("splits", 2)
        if os.path.lexists(dev_dir):
            C.fail("splits", "runs/taskspecs/dev exists in a train root")
        msg = set_diff("runs/taskspecs/train vs splits/train.txt", json_stems(train_dir), train_ids)
        if msg:
            C.fail("splits", msg)

    spec_split = rf.SPLIT_DIR[profile]
    for r in req:
        if r["kind"] != BUNDLE_KIND[profile] or r["path"] not in present:
            continue
        C.count("cutoff")
        try:
            spec = read_json(os.path.join(root, rf.contract_path("taskspec", r["task_id"], spec_split)))
            bundle = read_json(os.path.join(root, rf.safe_relpath(r["path"])))
            cutoff = int(spec["publication_cutoff"])
            late = [str(p.get("paper_id")) for p in bundle.get("papers") or []
                    if p.get("year") is not None and int(p["year"]) > cutoff]
        except (OSError, ValueError, KeyError, TypeError) as e:
            C.fail("cutoff", f"{r['path']}: cannot read the bundle or its TaskSpec ({type(e).__name__})")
            continue
        if late:
            C.fail("cutoff", f"{r['path']}: {len(late)} papers dated after the cutoff year {cutoff}: {sample(late)}")

    if profile == "test":
        C.count("null_tau")
        try:
            msg = set_diff(NULL_TAU + " keys vs splits/test.txt", set(read_json(os.path.join(root, NULL_TAU))), test_ids)
            if msg:
                C.fail("null_tau", msg)
        except (OSError, ValueError, TypeError) as e:
            C.fail("null_tau", f"{NULL_TAU} unreadable: {type(e).__name__}")
    else:
        C.skip("null_tau", "train profile")

    parent = parent if parent is not None else os.environ.get("CCBENCH_PARENT")
    if parent and os.path.realpath(parent) != os.path.realpath(root):
        for p in required:
            C.count("parent")
            if os.path.lexists(os.path.join(parent, rf.safe_relpath(p))):
                C.fail("parent", f"{p}: also present under CCBENCH_PARENT={parent}; the scorer may read it from there")
    else:
        C.skip("parent", "CCBENCH_PARENT unset or equal to the root")
    return finish(C, root, profile, snap)


def finish(C: Checks, root: str, profile: str, snap: str | None) -> dict:
    report = {"schema": VERIFY_SCHEMA, "dataset": rf.DATASET, "version": rf.VERSION, "profile": profile,
              "root": root, "snapshot": snap,
              "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "ok": C.ok, "checks": C.res}
    if os.path.isdir(root):
        dst = os.path.join(root, "verify_report.json")
        tmp = f"{dst}.tmp-{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=1, ensure_ascii=False))
        os.replace(tmp, dst)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--profile", choices=rf.PROFILES, required=True)
    ap.add_argument("--snapshot")
    ap.add_argument("--parent")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    report = verify(a.root, a.profile, a.snapshot, a.parent, a.workers)
    for name, r in report["checks"].items():
        for msg in r["failures"][:PRINT_LIMIT]:
            print(f"[verify_data] FAIL {name}: {msg}", file=sys.stderr)
        if len(r["failures"]) > PRINT_LIMIT:
            print(f"[verify_data] FAIL {name}: {len(r['failures']) - PRINT_LIMIT} more in verify_report.json", file=sys.stderr)
    print(json.dumps({"ok": report["ok"], "profile": report["profile"], "root": report["root"],
                      "checks": {k: ("skipped" if "skipped" in v else "ok" if v["ok"] else f"FAIL ({len(v['failures'])})")
                                 for k, v in report["checks"].items()}}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
