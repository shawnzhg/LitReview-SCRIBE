"""Registered constants and input paths of windowbench, and the registry of constants and input
hashes written next to every table."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
CCB = EVAL / "ccbench"
CCB_OUT = Path(os.environ.get("CCBENCH_OUT") or Path(os.environ.get("CCBENCH_ROOT") or CCB) / "out")
CCB_DATA = Path(os.environ.get("CCBENCH_ROOT") or CCB_OUT.parent)
OUT_DEFAULT = Path(os.environ.get("WINDOWBENCH_OUT") or "/nonexistent/WINDOWBENCH_OUT")


def env_path(var: str) -> Path:
    return Path(os.environ.get(var) or f"/nonexistent/{var}")


RUNS = env_path("SCRIBE_RUNS_ROOT")
OUTLINE_WINDOW = env_path("WINDOWBENCH_OUTLINE_WINDOW")
RUN_ROOT_FIXED_INPUT = RUNS / "campaign50_ref"
RUN_ROOTS_SAME_POOL = (RUNS / "campaign50_native", RUNS / "campaign50")


def _ccbench_paths():
    import importlib.util
    spec = importlib.util.spec_from_file_location("windowbench_ccbench_paths", CCB / "paths.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CCB_PATHS = _ccbench_paths()


def mode_level(mode: str) -> str:
    return _CCB_PATHS.mode_levels().get(mode) or _CCB_PATHS.DEFAULT_MODE_LEVELS[mode]


def bootstrap_env() -> None:
    os.environ.setdefault("CCBENCH_ROOT", str(CCB_DATA))
    os.environ.setdefault("CCBENCH_OUT", str(CCB_OUT))
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if str(EVAL) in sys.path:
        sys.path.remove(str(EVAL))
    sys.path.insert(0, str(EVAL))
    m = sys.modules.get("ccbench")
    if m is not None and Path(m.__file__).resolve().parent != CCB:
        raise ImportError(f"ccbench was imported from {m.__file__}, not from {CCB}")


SOURCES = {
    "window_scores": CCB_OUT / "E13/window_scores.parquet",
    "calibration_e10": CCB_OUT / "E10/calibration.csv",
    "conformance": CCB_OUT / "E1/conformance.csv",
    "membership": env_path("WINDOWBENCH_MEMBERSHIP"),
    "outline_arms": OUTLINE_WINDOW / "scores_arms.jsonl",
    "outline_controls": OUTLINE_WINDOW / "scores_controls.jsonl",
    "outline_axis_rows": OUT_DEFAULT / "outline_axis/rows.jsonl",
    "outline_axis_human": OUT_DEFAULT / "outline_axis/human.jsonl",
    "radii": CCB_OUT / "E13/radii.parquet",
    "distances": CCB_OUT / "E13/distances.parquet",
    "campaign50": env_path("SCRIBE_EVAL_TASKS"),
    "allowlists": env_path("SCRIBE_ALLOWLISTS"),
    "bundle_entry_root": RUNS / mode_level("bundle_entry"),
    "canonical_bundles": RUNS / "canonical" / "campaign50_ref" / "evidence_bundle",
    "prereg": CCB / "prereg.yaml",
}

REGISTERED: dict = {
    "alpha": 0.05,
    "cluster_level": "subfield",
    "min_tasks": 10,
    "aa_flips": 2000,
    "tolerance_admit": {"gt": 0.0, "lt": 0.5},
    "windows": ["retrieval", "synthesis", "planning", "writing", "draft", "draft_trunc", "system", "system_trunc"],
    "axes": ["retrieval", "synthesis", "reasoning", "planning", "writing", "form"],
    "axis_groups": {"retrieval": ["retrieval"], "synthesis": ["claim_coverage"], "reasoning": ["reasoning"],
                    "planning": ["organisation"], "writing": ["citation_fidelity"], "form": ["form"]},
    "axis_exclude_flags": ["non_discriminating", "channel_defect", "guardrail", "no_topical_signal", "saturated"],
    "planning_axis_readouts": ["outline_title_f1_lex", "outline_title_f1_emb"],
    "axis_primary_observation": "system_trunc",
    "draft_exit_axes": ["synthesis", "reasoning", "writing", "form"],
    "channel_defects": {
        "syn_forbidden_ok": "inverted: a forbidden claim matches an admissible one, so injecting it raises the score",
        "org_placement_within": "ceiling at a single top-level section",
        "org_placement_adjacent": "tracks paragraph counts",
        "org_placement_distant": "tracks paragraph counts",
    },
    "admission_window": "system",
    "planning_topical_margin": 0.15,
    "planning_saturation_gap": 0.01,
    "guardrails": ["completion", "wr_cite_fidelity", "wr_post_cutoff_ok", "syn_paper_reuse"],
}


def sha256_file(p: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    if p.is_dir():
        for q in sorted(x for x in p.rglob("*") if x.is_file()):
            h.update(str(q.relative_to(p)).encode())
            h.update(sha256_file(q, 64).encode())
        return h.hexdigest()[:n]
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def registry(extra: dict | None = None) -> dict:
    inputs = {}
    for k, p in SOURCES.items():
        inputs[k] = {"path": str(p), "sha256_16": sha256_file(p) if p.exists() else None, "exists": p.exists()}
    reg = {"constants": REGISTERED, "inputs": inputs}
    if extra:
        reg.update(extra)
    reg["constants_sha256_16"] = hashlib.sha256(json.dumps(REGISTERED, sort_keys=True).encode()).hexdigest()[:16]
    return reg
