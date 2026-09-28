"""Resolves data and output locations from the CCBENCH_ROOT, CCBENCH_PARENT, CCBENCH_EVAL_SUBDIR and
CCBENCH_OUT environment variables."""

from __future__ import annotations

import functools
import os
import re
from pathlib import Path

ICLR = Path(os.environ.get("CCBENCH_ROOT", "/nonexistent/CCBENCH_ROOT"))
PARENT = Path(os.environ.get("CCBENCH_PARENT", "/nonexistent/CCBENCH_PARENT"))
EVAL_SUBDIR = os.environ.get("CCBENCH_EVAL_SUBDIR", "").strip("/")
EVAL_PREFIX = f"{EVAL_SUBDIR}/" if EVAL_SUBDIR else ""
OUT = Path(os.environ.get("CCBENCH_OUT", str(ICLR / "out")))

BASELINE_ARMS = ("autosurvey", "surveyforge", "surveyg", "sgi", "lira", "llmxmr", "drtulu")
FIXED_INPUT_ARMS = ("autosurvey", "surveyg", "sgi", "llmxmr")
PANEL_A = BASELINE_ARMS
SEED_RE = re.compile(r"^(?P<mode>.+?)\.seed(?P<seed>\d+)$")


def agent_keys() -> list[str]:
    return [s.strip() for s in os.environ.get("CCBENCH_PANEL_B_EXTRA", "").split(",") if s.strip()]


def resolve(rel: str | os.PathLike) -> Path:
    rel = Path(rel)
    for root in (ICLR, PARENT):
        p = root / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"{rel} not found under {ICLR} or {PARENT}")


def resolve_opt(rel: str | os.PathLike) -> Path | None:
    try:
        return resolve(rel)
    except FileNotFoundError:
        return None


def campaign50_root(campaign: str = "campaign50") -> Path:
    return resolve(f"{EVAL_PREFIX}runs/{campaign}")


def arm_dir(arm: str, campaign: str = "campaign50") -> Path:
    if arm == "drtulu" and campaign == "campaign50":
        p = resolve_opt(f"{EVAL_PREFIX}runs/campaign50_native/drtulu")
        if p is not None:
            return p
    return campaign50_root(campaign) / arm


def task_dir(arm: str, task: str, campaign: str = "campaign50") -> Path:
    return arm_dir(arm, campaign) / task


def mode_levels() -> dict[str, str]:
    out: dict[str, str] = {}
    for item in os.environ.get("CCBENCH_MODE_LEVELS", "").split(","):
        if "=" not in item:
            continue
        mode, level = (x.strip() for x in item.split("=", 1))
        if mode and level:
            out[mode] = level if level.startswith("level") else f"level{level}"
    return out


DEFAULT_MODE_LEVELS = {"native_chain": "level1", "bundle_entry": "level7"}


def split_mode(mode: str) -> tuple[str, int]:
    m = SEED_RE.match(mode)
    return (m.group("mode"), int(m.group("seed"))) if m else (mode, 0)


def agent_run_dir(system: str, task: str, mode: str) -> Path:
    base, seed = split_mode(mode)
    level = mode_levels().get(base) or DEFAULT_MODE_LEVELS[base]
    return resolve(f"{EVAL_PREFIX}runs/{level}/{system}/{base}/{task}/seed{seed}")


SPLITS = ("dev", "train")


@functools.lru_cache(maxsize=8192)
def split_of(task: str) -> str:
    probed = []
    for split in SPLITS:
        rel = f"{EVAL_PREFIX}runs/taskspecs/{split}/{task}.json"
        probed.append(rel)
        if resolve_opt(rel) is not None:
            return split
    raise FileNotFoundError(f"no taskspec for task {task!r}; probed " + ", ".join(probed))


def canonical_bundle_rel(split: str = "dev") -> str:
    if split == "dev":
        return f"{EVAL_PREFIX}runs/canonical/campaign50_ref/evidence_bundle"
    return f"{EVAL_PREFIX}runs/canonical/{split}/bundles/evidence_bundle"


def taskspec_path(task: str) -> Path:
    return resolve(f"{EVAL_PREFIX}runs/taskspecs/{split_of(task)}/{task}.json")


def gold_path(task: str) -> Path:
    return resolve(f"{EVAL_PREFIX}runs/gold/{split_of(task)}/{task}.json")


def gold_refs_path(task: str) -> Path:
    return resolve(f"{EVAL_PREFIX}runs/baseline_inputs/gold/{task}/refs.json")


def campaign50_json() -> Path:
    return resolve("data/pool_corpus/campaign50.json")


def dataset2000_root() -> Path:
    return resolve("data_2000/dataset_2000")


def gt_graph_path(task: str) -> Path:
    return dataset2000_root() / "graphs" / task / "graph_enhanced.json"


def gt_refs_path(task: str) -> Path:
    return dataset2000_root() / "refs_enriched" / f"{task}.json"


def gt_manifest(name: str) -> Path:
    return dataset2000_root() / "manifests" / name


def nomic_model_dir() -> Path:
    return PARENT / "data" / "models" / "nomic-embed-text-v1"


def out_dir(*parts: str) -> Path:
    p = OUT.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
