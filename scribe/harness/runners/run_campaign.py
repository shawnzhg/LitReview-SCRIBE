#!/usr/bin/env python3
"""Runs many tasks in parallel for acquisition or generation, skipping runs whose exit artifact
exists and appending to a campaign index; the drivers call its main()."""

from __future__ import annotations
import argparse
import json
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RUNS, utcnow, host_class, hostname
from llm import VLLMClient
import runner as R

IDX = RUNS / "_index"
_ILOCK = threading.Lock()
SEEDS = (0, 1, 2)


def dev_tasks(only=None):
    rows = [json.loads(l) for l in (RUNS / "taskspecs" / "dev_index.jsonl").open()]
    ids = [r["task_id"] for r in rows]
    if only:
        want = set(only.split(","))
        ids = [t for t in ids if t in want]
    return ids


TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def tasks_from_file(path):
    ids, seen, bad = [], set(), []
    for line in Path(path).read_text().splitlines():
        t = line.split("#", 1)[0].strip()
        if not t or t in seen:
            continue
        if not TASK_ID_RE.match(t) or t in (".", ".."):
            bad.append(t)
            continue
        seen.add(t)
        ids.append(t)
    if bad:
        raise SystemExit(f"--tasks-file {path}: {len(bad)} line(s) are not a valid task id "
                         f"(a task id is a path segment matching {TASK_ID_RE.pattern}): "
                         f"{bad[:5]}")
    if not ids:
        raise SystemExit(f"--tasks-file {path} contains no task ids")
    return ids


def select_tasks(a):
    if getattr(a, "tasks_file", None):
        if a.tasks:
            raise SystemExit("--tasks and --tasks-file are mutually exclusive")
        return tasks_from_file(a.tasks_file)
    return dev_tasks(a.tasks)


def done(level, system, mode, task, seed, phase):
    d = R.run_dir(level, system, mode, task, seed)
    return (d / ("evidence_bundle.json" if phase == "acquisition" else "report_artifact.json")).exists()


ENTRY_MODE_OF = {"native_chain": "native_chain", "bundle_entry": "bundle_entry"}


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["acquisition", "generation"], required=True)
    ap.add_argument("--systems", default="SCRIBE")
    ap.add_argument("--mode", default="native_chain", choices=["native_chain", "bundle_entry"])
    ap.add_argument("--seeds", default="0", help="comma-separated seeds from 0, 1, 2")
    ap.add_argument("--tasks", default=None, help="comma-separated task_ids (dev_index only)")
    ap.add_argument("--tasks-file", default=None,
                    help="path to a file of task_ids of any split, one per line (# comments allowed); "
                         "mutually exclusive with --tasks.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--campaign", default=None)
    ap.add_argument("--redo", action="store_true")
    return ap


def one(phase, task, system, mode, seed, level, client, entry_mode, client_write=None):
    t0 = time.time()
    row = {"task_id": task, "system": system, "mode": mode, "seed": seed, "phase": phase,
           "level": level, "started_at": utcnow(), "host": hostname(),
           "host_class": host_class()}
    try:
        if phase == "acquisition":
            bundle, m = R.run_acquisition(task, system, seed, client, level=level, mode=mode)
            row.update(status=m["status"], exit_hash=m["exit_hash"],
                       resources=m["observed_resources"],
                       n_papers=len(bundle["papers"]) if bundle else 0,
                       n_evidence=len(bundle["evidence"]) if bundle else 0)
        else:
            row.update(entry_mode=entry_mode)
            entry = R.load_canonical(task, "evidence_bundle") if entry_mode == "bundle_entry" else None
            out, mans = R.run_generation(task, system, seed, client, level=level, mode=mode,
                                         entry_bundle=entry, entry_mode=entry_mode,
                                         client_write=client_write)
            r = out.get("report_artifact")
            eb = entry or {}
            if not eb:
                p_eb = R.run_dir(level, system, mode, task, seed) / "entry_evidence_bundle.json"
                eb = json.loads(p_eb.read_text()) if p_eb.exists() else {}
            row.update(n_papers_in=len(eb.get("papers") or []),
                       n_evidence_in=len(eb.get("evidence") or []))
            row.update(status=(mans[-1]["status"] if mans else "failed"),
                       windows={m["window"]: m["status"] for m in mans},
                       resources=(mans[-1]["observed_resources"] if mans else {}),
                       n_claims=len(out.get("synthesis_graph", {}).get("claims", [])),
                       n_sections=len(out.get("outline_plan", {}).get("sections", [])),
                       n_words=(r or {}).get("terminal_audit", {}).get("n_words"),
                       exit_hash=(r or {}).get("content_hash", ""))
            d = R.run_dir(level, system, mode, task, seed)
            ip = d / "integrity.json"
            if ip.exists():
                row["integrity_findings"] = json.loads(ip.read_text())["n_findings"]
            fr = next((m.get("failure_reason") for m in mans if m.get("failure_reason")), None)
            if fr:
                row["error"] = fr
            if entry_mode == "bundle_entry":
                dp = d / "delivery.json"
                dj = json.loads(dp.read_text()) if dp.exists() else {}
                ds = dj.get("summary") or {}
                row.update(allowlist_sha256=ds.get("allowlist_sha256"),
                           n_without_abstract=ds.get("n_without_abstract"),
                           n_year_unknown=ds.get("n_year_unknown"),
                           delivery_ok=bool(dj.get("ok")) if dj else False)
                if dj and not dj.get("ok"):
                    row["error"] = "harness_failure:bundle_delivery"
                elif not dj and not fr:
                    row["error"] = ("MissingEntryArtifact: canonical bundle not found at "
                                    f"{R.canonical_root(task) / 'evidence_bundle' / (task + '.json')}")
    except Exception as e:
        row.update(status="failed", error=f"{type(e).__name__}: {e}",
                   traceback=traceback.format_exc()[:3000])
    row["wall_s"] = round(time.time() - t0, 1)
    row["ended_at"] = utcnow()
    return row


def main():
    a = build_parser().parse_args()

    entry_mode = ENTRY_MODE_OF[a.mode]
    if a.phase == "acquisition" and entry_mode == "bundle_entry":
        sys.exit("--mode bundle_entry has no acquisition phase: the entry bundle is the "
                 "canonical allowlist bundle (biolitbench/tasks/build_canonical_bundles.py)")
    level = a.level
    systems = a.systems.split(",")
    bad = [sy for sy in systems if sy not in R.SYSTEMS]
    if bad:
        sys.exit(f"--systems {a.systems}: {bad} not in {R.SYSTEMS}")
    seeds = [int(x) for x in a.seeds.split(",")]
    if not seeds or any(s not in SEEDS for s in seeds) or len(set(seeds)) != len(seeds):
        sys.exit(f"--seeds {a.seeds}: distinct seeds from {SEEDS}")
    tasks = select_tasks(a)
    campaign = a.campaign or f"{a.phase}.{a.mode}.{'-'.join(systems)}.s{'-'.join(map(str, seeds))}"
    IDX.mkdir(parents=True, exist_ok=True)
    out_path = IDX / f"{campaign}.jsonl"

    import os as _os
    _rv = _os.environ.get("SCRIBE_RENDEZVOUS")
    if not _rv:
        sys.exit("SCRIBE_RENDEZVOUS is not set (the launchers write the serving rendezvous)")
    client = VLLMClient(rendezvous=json.loads(_rv))
    if not client.health():
        sys.exit(f"vLLM endpoint {client.url} is not healthy")
    _rw = _os.environ.get("SCRIBE_RENDEZVOUS_WRITING")
    client_write = VLLMClient(rendezvous=json.loads(_rw)) if _rw else None
    if client_write is not None and not client_write.health():
        sys.exit(f"writing-window vLLM endpoint {client_write.url} is not healthy")

    jobs = [(t, sy, sd) for sy in systems for sd in seeds for t in tasks
            if a.redo or not done(level, sy, a.mode, t, sd, a.phase)]
    skipped = len(tasks) * len(systems) * len(seeds) - len(jobs)
    print(f"[campaign] {campaign}\n[campaign] host={hostname()} class={host_class()} "
          f"endpoint={client.url} writing_endpoint={client_write.url if client_write else 'same'}\n"
          f"[campaign] {len(jobs)} runs to do, {skipped} already complete, "
          f"workers={a.workers}", flush=True)

    t0, n_ok, n_fail = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, a.phase, t, sy, a.mode, sd, level, client, entry_mode, client_write):
                (t, sy, sd) for (t, sy, sd) in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            row = f.result()
            n_ok += row["status"] == "ok"
            n_fail += row["status"] != "ok"
            with _ILOCK, out_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            if len(jobs) <= 20 or i % 5 == 0 or row["status"] != "ok":
                el = time.time() - t0
                print(f"  [{i}/{len(jobs)}] {row['system']} {row['task_id'][:22]} "
                      f"{row['status']} {row['wall_s']}s | ok={n_ok} fail={n_fail} "
                      f"| {el/60:.1f}min elapsed, ~{el/i*(len(jobs)-i)/60:.0f}min left", flush=True)

    print(f"[campaign] DONE {n_ok} ok / {n_fail} failed in {(time.time()-t0)/60:.1f} min")
    print(f"[campaign] index: {out_path}")


if __name__ == "__main__":
    main()
