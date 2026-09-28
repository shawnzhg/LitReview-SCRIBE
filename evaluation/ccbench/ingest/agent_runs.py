"""Loaders for staged agent run directories and their typed artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from ccbench.ingest.logs import iso_to_epoch, iter_jsonl

ARTIFACTS = ("task_spec", "evidence_bundle", "synthesis_graph", "outline_plan", "report_artifact")
ENTRY_ARTIFACTS = ("entry_evidence_bundle", "entry_synthesis_graph", "entry_outline_plan")


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def load_run(run_dir: Path) -> dict:
    out: dict = {"dir": str(run_dir)}
    for name in ARTIFACTS + ENTRY_ARTIFACTS:
        p = run_dir / f"{name}.json"
        out[name] = load_json(p) if p.exists() else None
    out["manifests"] = list(iter_jsonl(run_dir / "manifests.jsonl"))
    out["resources"] = load_json(run_dir / "resources.json") if (run_dir / "resources.json").exists() else {}
    out["integrity"] = load_json(run_dir / "integrity.json") if (run_dir / "integrity.json").exists() else {}
    out["binding"] = load_json(run_dir / "binding.json") if (run_dir / "binding.json").exists() else {}
    return out


def iter_trace(run_dir: Path):
    for r in iter_jsonl(run_dir / "trace.jsonl"):
        r["_t0"] = iso_to_epoch(r["started_at"]) if r.get("started_at") else None
        r["_t1"] = iso_to_epoch(r["ended_at"]) if r.get("ended_at") else None
        yield r


def window_status(run: dict) -> dict[str, dict]:
    out = {}
    for m in run["manifests"]:
        out[m["window"]] = {
            "entry_hash": m.get("entry_hash"),
            "exit_hash": m.get("exit_hash"),
            "status": m.get("status"),
        }
    return out
