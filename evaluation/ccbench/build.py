"""Builds the common rollout of every system and task through the adapters and writes the
conformance table. Usage: python -m ccbench.build [--systems ...] [--tasks ...] [--campaign ...]."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
import traceback

from ccbench import paths
from ccbench.adapters import AGENT_ADAPTER, BASELINE_ADAPTERS
from ccbench.ingest import gold
from ccbench.model import Rollout, bot_rollout


def agent_runs() -> list[tuple[str, str]]:
    out = []
    for k in paths.agent_keys():
        system, _, mode = k.partition(".")
        out.append((system, mode or "native_chain"))
    return out


def system_key(system: str, mode: str | None) -> str:
    return system if mode in (None, "campaign", "native_chain") else f"{system}.{mode}"


def rollout_path(system: str, mode: str | None, task: str):
    return paths.out_dir("rollouts", system_key(system, mode)) / f"{task}.json"


def load_rollouts(system_keys: list[str] | None = None, tasks: list[str] | None = None) -> list[Rollout]:
    root = paths.OUT / "rollouts"
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        if system_keys and d.name not in system_keys:
            continue
        for p in sorted(d.glob("pmcid_*.json")):
            if tasks and p.stem not in tasks:
                continue
            out.append(Rollout.from_json(p))
    return out


def _row(ro: Rollout, secs: float) -> dict:
    r = ro.report
    return {
        "system": system_key(ro.system, ro.mode),
        "task": ro.task,
        "panel": ro.panel,
        "status": ro.status,
        "bot_reason": ro.bot_reason or "",
        "n_events": len(ro.events),
        "n_llm": ro.resources.get("n_llm", 0),
        "n_search": ro.resources.get("n_search", 0),
        "n_open": ro.resources.get("n_open", 0),
        "usd": round(float(ro.resources.get("usd") or 0.0), 4),
        "wall_s": ro.resources.get("wall_s") or round(float(ro.resources.get("wall_ms") or 0) / 1000.0, 1),
        "n_papers": len(ro.papers),
        "n_outline": len(ro.outline),
        "n_sentences": len(r.sentences) if r else 0,
        "words": r.words if r else 0,
        "n_bib": len(r.bibliography) if r else 0,
        "citation_marks": r.total_citation_marks if r else 0,
        "unresolved": r.unresolved_citations if r else 0,
        "resolution_rate": round(1 - r.unresolved_citations / r.total_citation_marks, 4) if r and r.total_citation_marks else "",
        "has_graph": bool(ro.graph),
        "adapt_s": round(secs, 2),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=None, help="comma list of arms or agent keys (<system>.<mode>)")
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--campaign", default="campaign50", help="campaign50 (same pool) or campaign50_ref (fixed input, keyed <arm>.ref)")
    a = ap.parse_args(argv)
    tasks = a.tasks.split(",") if a.tasks else gold.campaign50_tasks()
    want = set(a.systems.split(",")) if a.systems else None
    suffix = ".ref" if a.campaign.endswith("_ref") else ""
    arms = paths.FIXED_INPUT_ARMS if suffix else tuple(BASELINE_ADAPTERS)
    jobs: list[tuple[str, str]] = [(arm, "campaign") for arm in arms] + ([] if suffix else agent_runs())
    rows = []
    for system, mode in jobs:
        key = system_key(system, mode) + (suffix if mode == "campaign" else "")
        if want and key not in want and system not in want:
            continue
        for task in tasks:
            dest = rollout_path(system + (suffix if mode == "campaign" else ""), mode, task)
            t0 = time.time()
            if dest.exists() and not a.redo:
                ro = Rollout.from_json(dest)
            else:
                try:
                    if mode == "campaign":
                        td = paths.task_dir(system, task, a.campaign)
                        if td.exists():
                            ro = BASELINE_ADAPTERS[system](task, td)
                            ro.system = system + suffix
                            if suffix:
                                ro.meta["campaign"] = a.campaign
                                ro.meta["provenance_tier"] = "reference_derived"
                        else:
                            ro = bot_rollout(system + suffix, task, "A", gold.context_for(task), "task_dir_missing", mode="campaign")
                    else:
                        ro = AGENT_ADAPTER(system, task, mode)
                except Exception as e:
                    traceback.print_exc()
                    ro = bot_rollout(system + (suffix if mode == "campaign" else ""), task, "A" if mode == "campaign" else "B", gold.context_for(task), f"adapter_error:{type(e).__name__}:{str(e)[:80]}", mode=mode)
                ro.to_json(dest)
            rows.append(_row(ro, time.time() - t0))
            print(f"{key:20s} {task}  {ro.status:4s} P={len(ro.papers):4d} words={rows[-1]['words']:6d} bib={rows[-1]['n_bib']:4d} {'' if ro.status=='ok' else ro.bot_reason}", flush=True)

    e1 = paths.out_dir("E1")
    dest = e1 / "conformance.csv"
    if dest.exists():
        mine = {r["system"] for r in rows}
        with open(dest, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["system"] not in mine] + rows
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
        w.writeheader()
        w.writerows(rows)
    summ: dict[str, dict] = {}
    for r in rows:
        s = summ.setdefault(r["system"], {"system": r["system"], "panel": r["panel"], "tasks": 0, "ok": 0, "bot": 0, "papers_med": [], "words_med": [], "bib_med": [], "resolution": [], "usd": 0.0})
        s["tasks"] += 1
        s["ok" if r["status"] == "ok" else "bot"] += 1
        if r["status"] == "ok":
            s["papers_med"].append(int(r["n_papers"]))
            s["words_med"].append(int(r["words"]))
            s["bib_med"].append(int(r["n_bib"]))
            if r["resolution_rate"] != "":
                s["resolution"].append(float(r["resolution_rate"]))
        s["usd"] += float(r["usd"])
    med = lambda xs: statistics.median(xs) if xs else ""
    with open(e1 / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "panel", "tasks", "ok", "bot", "papers_median", "words_median", "bib_median", "citation_resolution_mean", "usd_total"])
        for s in summ.values():
            w.writerow([s["system"], s["panel"], s["tasks"], s["ok"], s["bot"], med(s["papers_med"]), med(s["words_med"]), med(s["bib_med"]), round(statistics.mean(s["resolution"]), 4) if s["resolution"] else "", round(s["usd"], 2)])
    print(f"\nwrote {e1/'conformance.csv'} ({len(rows)} rows) and summary.csv")


if __name__ == "__main__":
    main()
