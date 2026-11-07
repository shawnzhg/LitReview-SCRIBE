"""Shared setup of the training tests: repo paths, site data locations, sys.path and a synthetic
planning-window cells table."""

import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
TRAINING = REPO / "scribe" / "training"
EVAL = REPO / "evaluation"
PIN = REPO / "configs" / "models" / "Qwen3.8-27B-nothink.json"


def _env_path(name: str) -> Path:
    return Path(os.environ.get(name) or f"/nonexistent/{name}")


MODEL = _env_path("SCRIBE_MODEL_DIR")
HF_HOME = _env_path("SCRIBE_HF_HOME")
EVAL_TASKS = _env_path("SCRIBE_EVAL_TASKS")
TRAIN_MANIFEST = _env_path("SCRIBE_TRAIN_MANIFEST")
RUNS_ROOT = _env_path("SCRIBE_RUNS_ROOT")


def need(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(f"test data not available (set the site environment): {missing}")


def hf_snapshot() -> Path:
    pin = json.loads(PIN.read_text())
    org, name = pin["hf_id"].split("/")
    return HF_HOME / "hub" / f"models--{org}--{name}" / "snapshots" / pin["revision"]


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("KVSKILL_CN_NO_AUTOPATCH", "1")
for p in (str(TRAINING), str(TRAINING / "launch"), str(REPO / "third_party" / "kvskill"), str(EVAL)):
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)

PE_READOUTS = ("org_size_fit", "outline_title_f1_emb", "outline_title_f1_lex")
INITIAL = "initial.fixed"
CELLS = {
    "outline_title_f1_lex": {"a": (0.8, 0.10, "tolerance"), "b": (0.6, 0.30, "tolerance"),
                             "c": (0.7, 0.05, "unfair")},
    "outline_title_f1_emb": {"a": (0.3, 0.05, "tolerance"), "b": (0.2, 0.05, "exact"),
                             "c": (0.1, 0.90, "unfair")},
    "org_size_fit": {"a": (0.52, 0.01, "tolerance"), "b": (0.51, 0.02, "tolerance"),
                     "c": (0.90, 0.50, "unfair")},
}
OURS = 0.5
EXPECTED = {"outline_title_f1_lex": ("gain", 2.5, "a", 0.2),
            "outline_title_f1_emb": ("guard", 0.5, "a", 0.05),
            "org_size_fit": ("gain", 3.0, "c", 0.03)}


def synthetic_cells(initial: str = INITIAL) -> str:
    rows = ["window,readout,ours_family,opponent,regime,n,ours,opp,diff,q,status"]
    for r, opps in CELLS.items():
        for o, (opp, q, regime) in opps.items():
            rows.append(f"planning,{r},{initial},{o},{regime},50,{OURS},{opp},{OURS - opp},{q},undecided")
        rows.append(f"system,{r},{initial},a,tolerance,50,0.1,0.9,-0.8,0.01,decided")
    return "\n".join(rows) + "\n"
