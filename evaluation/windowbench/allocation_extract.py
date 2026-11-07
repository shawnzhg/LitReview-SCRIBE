"""Extracts each arm's evidence allocation (the papers handed to each section), its outline and its
held papers per task into JSONL files. Usage: python -m windowbench.allocation_extract extract
[--arms <list>] [--ours <key>=<run tree>,...] [--out <dir>]."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import time
from pathlib import Path

from . import config as C

SELF_ARMS = ("autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr", "lira")
REF_ARMS = ("autosurvey.ref", "surveyg.ref", "llmxmr.ref", "sgi.ref")
ARMS = SELF_ARMS + REF_ARMS
CAMPAIGN = {a: C.RUNS / "campaign50" / a for a in SELF_ARMS}


def _task_dir(arm: str, task: str) -> Path:
    return C.RUN_ROOT_FIXED_INPUT / arm[:-4] / task if arm.endswith(".ref") else CAMPAIGN[arm] / task


def parse_ours(spec: str | None) -> dict[str, Path]:
    out = {}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        key, sep, tree = item.partition("=")
        if not sep or not key.strip() or not tree.strip():
            raise SystemExit(f"--ours takes <key>=<run tree>, got {item!r}")
        out[key.strip()] = Path(tree.strip())
    return out

_PMID = re.compile(r"^(?:pmid[_:]?)?(\d{6,9})$", re.I)


def _as_pmid(x) -> str | None:
    m = _PMID.match(str(x).strip())
    return m.group(1) if m else None


def normalise_allocation(arm: str, ex: dict) -> dict[str, set]:
    base = arm[:-4] if arm.endswith(".ref") else arm
    if base == "autosurvey":
        src = ex.get("section_evidence_pmids") or {}
    else:
        src = ex.get("section_evidence") or {}
    lookup = {}
    if base == "llmxmr":
        lookup = {p["bibkey"]: str(p["pmid"]) for p in (ex.get("papers") or []) if p.get("pmid")}
    out: dict[str, set] = {}
    for sec, vals in src.items():
        got = set()
        for v in (vals or []):
            p = _as_pmid(v) or lookup.get(str(v))
            if p:
                got.add(p)
        out[str(sec)] = got
    return out


def rollout_papers(arm: str, task: str) -> set[str] | None:
    fp = C.CCB_OUT / "rollouts" / arm / f"{task}.json"
    if not fp.exists():
        return None
    try:
        return {str(x) for x in (json.load(open(fp)).get("papers") or [])}
    except Exception:
        return None


def extract_ours(key: str, root: Path, tasks: list[str], out_dir: Path, verbose: bool = True) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{key}.jsonl"
    done = {json.loads(l)["task"] for l in open(fp)} if fp.exists() else set()
    n = 0
    with open(fp, "a") as f:
        for t in tasks:
            if t in done:
                continue
            d = root / t / "seed0"
            if not d.is_dir():
                f.write(json.dumps({"task": t, "arm": key, "error": "no_task_dir"}) + "\n")
                continue
            t0 = time.time()
            try:
                plan = json.load(open(d / "outline_plan.json"))
                graph = json.load(open(d / "synthesis_graph.json"))
                bundle = json.load(open(d / "entry_evidence_bundle.json"))
                ev2pm = {e["evidence_id"]: _as_pmid(e.get("paper_id")) for e in (bundle.get("evidence") or [])}
                cl2ev = {c["claim_id"]: (c.get("evidence_ids") or []) for c in (graph.get("claims") or [])}
                alloc = {}
                outline = [{"id": str(sec.get("section_id")), "title": sec.get("title") or "",
                            "level": 2 if sec.get("parent_id") else 1,
                            "parent": str(sec["parent_id"]) if sec.get("parent_id") else None}
                           for sec in (plan.get("sections") or [])]
                for sec in (plan.get("sections") or []):
                    got = set()
                    for cid in (sec.get("claim_ids") or []):
                        for eid in cl2ev.get(cid, []):
                            pm = ev2pm.get(eid)
                            if pm:
                                got.add(pm)
                    for eid in (sec.get("evidence_ids") or []):
                        pm = ev2pm.get(eid)
                        if pm:
                            got.add(pm)
                    alloc[str(sec.get("title") or sec.get("section_id"))] = got
                papers = rollout_papers(key, t)
                if papers is None:
                    papers = {p for p in (_as_pmid(x.get("paper_id")) for x in (bundle.get("papers") or [])) if p}
                    papers |= {p for v in alloc.values() for p in v}
                rec = {"task": t, "arm": key, "n_sections": len(alloc),
                       "allocation": {k: sorted(v) for k, v in alloc.items()}, "outline": outline,
                       "papers": sorted(papers), "wall_s": round(time.time() - t0, 1)}
            except Exception as e:
                rec = {"task": t, "arm": key, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if verbose:
                print(f"  {key} {t}: {rec.get('n_sections', '-')} sections, {len(rec.get('papers', []))} papers"
                      f"{' ERROR ' + rec['error'] if 'error' in rec else ''}", flush=True)
    return n


def extract_arm(arm: str, tasks: list[str], out_dir: Path, verbose: bool = True) -> int:
    mod = importlib.import_module(f".window_extractors.{arm[:-4] if arm.endswith('.ref') else arm}", __package__)
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"{arm}.jsonl"
    done = set()
    if fp.exists():
        done = {json.loads(l)["task"] for l in open(fp)}
    n = 0
    with open(fp, "a") as f:
        for t in tasks:
            if t in done:
                continue
            d = _task_dir(arm, t)
            if not d.is_dir():
                f.write(json.dumps({"task": t, "arm": arm, "error": "no_task_dir"}) + "\n")
                continue
            t0 = time.time()
            try:
                ex = mod.extract(d)
                alloc = normalise_allocation(arm, ex)
                papers = rollout_papers(arm, t)
                if papers is None:
                    papers = set()
                    for k in ("papers", "papers_union", "retrieval"):
                        v = ex.get(k)
                        if isinstance(v, list):
                            for x in v:
                                p = _as_pmid(x) if not isinstance(x, dict) else _as_pmid(x.get("pmid") or x.get("paper_id") or "")
                                if p:
                                    papers.add(p)
                    papers |= {p for s in alloc.values() for p in s}
                rec = {"task": t, "arm": arm, "n_sections": len(alloc),
                       "allocation": {k: sorted(v) for k, v in alloc.items()},
                       "outline": ex.get("outline_final") or ex.get("outline_initial") or [],
                       "papers": sorted(papers), "wall_s": round(time.time() - t0, 1)}
            except Exception as e:
                rec = {"task": t, "arm": arm, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if verbose:
                print(f"  {arm} {t}: {rec.get('n_sections', '-')} sections, "
                      f"{len(rec.get('papers', []))} papers, {rec.get('wall_s', '-')}s{' ERROR ' + rec['error'] if 'error' in rec else ''}", flush=True)
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--arms", default=",".join(ARMS))
    e.add_argument("--ours", default=None, help="comma list of <key>=<run tree>, the tree holding <task>/seed0 of our run")
    e.add_argument("--tasks", default="all", help="'all' or an integer = first N evaluation tasks")
    e.add_argument("--out", default=str(C.OUT_DEFAULT / "allocation"))
    a = ap.parse_args(argv)
    C.bootstrap_env()
    from ccbench.ingest import gold
    tasks = gold.campaign50_tasks()
    if a.tasks != "all":
        tasks = tasks[: int(a.tasks)]
    out = Path(a.out)
    for arm in [x for x in a.arms.split(",") if x]:
        print(f"== {arm} ({len(tasks)} tasks)")
        extract_arm(arm, tasks, out)
    for key, root in parse_ours(a.ours).items():
        print(f"== {key} ({len(tasks)} tasks) from {root}")
        extract_ours(key, root, tasks, out)


if __name__ == "__main__":
    main()
