"""Writes the readout admission screen that WINDOWBENCH_MEMBERSHIP points at: each readout's human
tolerance, peer and system spread and its non-discriminating, humans-incomparable and guardrail flags
on the published pipelines at the system window, and the planning-exit readouts that carry no topical
signal (an outline scored against another task's review loses less than the registered margin) or are
saturated (the peers' 10th percentile reaches the target review's own outline). Usage: python -m
windowbench.admission [--out <json>]."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import roster as R

OUT = C.SOURCES["membership"]


def development_panel() -> list[str]:
    return sorted(k for k, v in R.SYSTEMS.items() if v["role"] == "baseline")


def discriminability_table(scores: pd.DataFrame, cal: pd.DataFrame, tasks: list[str], panel: list[str]) -> pd.DataFrame:
    C.bootstrap_env()
    from ccbench.config import prereg
    from ccbench.rankability.normalise import discriminability
    cfg = prereg()["calibration"]
    at = scores[(scores.window == C.REGISTERED["admission_window"]) & scores.task.isin(tasks)]
    return discriminability(cal, at[at.panel == "H"], at[(at.panel != "H") & at.system.isin(panel)],
                            eps_fair_q=cfg["epsilon_fair_quantile"], eps_ge=cfg["non_discriminating_if_eps_fair_ge"],
                            sd_lt=cfg["non_discriminating_if_peer_sd_lt"])


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} is missing: run python -m windowbench.planning_exit --controls")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def value(row: dict, readout: str):
    return ((row.get("readouts") or {}).get(readout) or {}).get("value")


def planning_checks(arms: list[dict], controls: list[dict], panel: set[str], candidates: list[str]) -> list[dict]:
    own = {(d["key"], d["task"]): d for d in arms if d.get("stage") == "final" and d.get("admitted")
           and d.get("control", "none") == "none" and not d.get("no_outline") and d["key"] in panel}
    wrong = {(d["key"], d["task"]): d for d in controls if d["control"] == "wrong_review" and d["key"] in panel}
    human = [d for d in controls if d["control"] == "human_own"]
    peers = [d for d in controls if d["control"] == "human_peer"]
    if not wrong or not human or not peers:
        raise SystemExit("the control rows lack the wrong-review, human-own or human-peer rows: run python -m windowbench.planning_exit --controls")
    margin, gap = C.REGISTERED["planning_topical_margin"], C.REGISTERED["planning_saturation_gap"]
    out = []
    for r in candidates:
        drops = [value(own[k], r) - value(wrong[k], r) for k in sorted(own.keys() & wrong.keys())
                 if value(own[k], r) is not None and value(wrong[k], r) is not None]
        ceiling = [value(d, r) for d in human if value(d, r) is not None]
        band = [value(d, r) for d in peers if value(d, r) is not None]
        if not drops or not ceiling or not band:
            continue
        ceil, q10, drop = float(np.mean(ceiling)), float(np.quantile(band, 0.10)), float(np.mean(drops))
        verdict = "saturated" if q10 >= ceil - gap else ("no_topical_signal" if drop < margin else "admitted")
        out.append({"readout": r, "human_own": ceil, "peer_q10": q10, "drop_against_wrong_review": drop,
                    "n_pairs": len(drops), "verdict": verdict})
    return out


def build(panel: list[str], scores_path: Path, arms_path: Path, controls_path: Path, out: Path) -> dict:
    scores = pd.read_parquet(scores_path)
    cal = pd.read_csv(C.SOURCES["calibration_e10"])
    tasks = sorted(json.load(open(C.SOURCES["campaign50"]))["tasks"])
    disc = discriminability_table(scores, cal, tasks, panel)
    nondisc = sorted(disc[disc.non_discriminating.astype(bool)].readout)
    direction = scores.drop_duplicates("readout").set_index("readout").direction.to_dict()
    arms, controls = read_rows(arms_path), read_rows(controls_path)
    excluded = set(nondisc) | set(C.REGISTERED["channel_defects"]) | set(C.REGISTERED["guardrails"])
    candidates = sorted({r for d in arms for r in (d.get("readouts") or {})
                         if direction.get(r) == "quality" and r not in excluded})
    checks = planning_checks(arms, controls, set(panel), candidates)
    mem = {"window": C.REGISTERED["admission_window"], "panel": panel, "non_discriminating": nondisc,
           "humans_incomparable": sorted(disc[disc.humans_incomparable.astype(bool)].readout),
           "n_readouts": int(len(disc)), "discriminability_table": json.loads(disc.to_json(orient="records")),
           "planning_controls": checks,
           "planning_no_topical_signal": sorted(c["readout"] for c in checks if c["verdict"] == "no_topical_signal"),
           "planning_saturated": sorted(c["readout"] for c in checks if c["verdict"] == "saturated")}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(mem, f, indent=1)
    print(f"wrote {out}: {len(nondisc)} non-discriminating of {mem['n_readouts']}; no topical signal "
          f"{mem['planning_no_topical_signal']}; saturated {mem['planning_saturated']}", flush=True)
    return mem


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT), help="output JSON (default $WINDOWBENCH_MEMBERSHIP)")
    a = ap.parse_args(argv)
    build(development_panel(), C.SOURCES["window_scores"], C.SOURCES["outline_arms"], C.SOURCES["outline_controls"], Path(a.out))


if __name__ == "__main__":
    main()
