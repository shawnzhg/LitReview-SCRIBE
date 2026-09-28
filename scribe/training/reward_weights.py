#!/usr/bin/env python3
"""Builds the planning-exit reward targets: per readout, the initial agent's gap to the strongest
published pipeline sets the weight and the half-width sets the scale. Usage: python
reward_weights.py --cells <cells.csv> --out <json>."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import pandas as pd

WINDOW = "planning_exit"
CELLS_WINDOW = "planning"
READOUTS = ("org_size_fit", "outline_title_f1_emb", "outline_title_f1_lex")
ADMISSIBLE = ("exact", "tolerance")
S_FLOOR = 0.03
W_CAP = 3.0
W_GUARD = 0.5
GUARD_AHEAD = 2.0


def sha16(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def body_sha16(body: dict) -> str:
    b = {k: v for k, v in body.items() if k != "sha256_16"}
    return hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest()[:16]


def _finite(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def weight(gap: float, s: float) -> tuple[str, float]:
    if -gap >= GUARD_AHEAD * s:
        return "guard", W_GUARD
    return "gain", min(W_CAP, 1.0 + max(0.0, gap / s))


def readout_entry(cells: pd.DataFrame, readout: str) -> dict:
    c = cells[cells["readout"] == readout]
    scored = c[c["diff"].map(_finite) & c["opp"].map(_finite) & c["ours"].map(_finite)]
    if not len(scored):
        raise SystemExit(f"{readout}: no published pipeline is scored against the initial agent")
    best = scored.loc[scored["opp"].idxmax()]
    gap = float(best["opp"]) - float(best["ours"])
    adm = c[c["regime"].astype(str).isin(ADMISSIBLE) & c["q"].map(_finite)]
    if not len(adm):
        raise SystemExit(f"{readout}: no admissible cell carries a half-width")
    q = float(statistics.median(float(x) for x in adm["q"]))
    s = max(S_FLOOR, q)
    mode, w = weight(gap, s)
    return {"initial": round(float(best["ours"]), 6), "best": round(float(best["opp"]), 6),
            "best_system": str(best["opponent"]), "n_tasks": int(best["n"]), "gap": round(gap, 6),
            "q": round(q, 6), "q_opponents": sorted(str(o) for o in adm["opponent"]),
            "s": round(s, 6), "mode": mode, "w": round(w, 6)}


def build(cells_csv: str) -> dict:
    cells = pd.read_csv(cells_csv)
    cells = cells[(cells["window"] == CELLS_WINDOW) & cells["readout"].isin(READOUTS)]
    if not len(cells):
        raise SystemExit(f"{cells_csv} has no {CELLS_WINDOW}-window cell of {list(READOUTS)}")
    fams = sorted(set(cells["ours_family"].astype(str)))
    if len(fams) != 1:
        raise SystemExit(f"{cells_csv} compares {len(fams)} families {fams}; it must hold the "
                         f"initial agent only")
    initial = fams[0]
    spec = {r: readout_entry(cells, r) for r in READOUTS}
    body = {"schema": "scribe_reward_targets/1", "initial_agent": initial,
            "baselines": sorted(set(cells["opponent"].astype(str))),
            "cells": {"path": str(cells_csv), "sha256_16": sha16(cells_csv)},
            "rule": (f"gap = the strongest pipeline's mean minus the initial agent's mean on their "
                     f"common leaderboard tasks at the {CELLS_WINDOW} window; s = max({S_FLOOR}, median q "
                     f"over the admissible cells); w = {W_GUARD} with g clipped at 0 if -gap >= "
                     f"{GUARD_AHEAD} s, else min({W_CAP}, 1 + max(0, gap / s))"),
            "window_sets": {WINDOW: sorted(READOUTS)},
            "window_readouts": {WINDOW: spec}}
    body["sha256_16"] = body_sha16(body)
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", required=True,
                    help="cells.csv of windowbench.run --ours <SCRIBE (untrained)> --windows planning")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    body = build(a.cells)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1, sort_keys=True))
    for r, v in sorted(body["window_readouts"][WINDOW].items()):
        print(f"  {r:22s} initial {v['initial']:.4f} best {v['best']:.4f} ({v['best_system']}) "
              f"gap {v['gap']:+.4f} s {v['s']:.4f} mode {v['mode']} w {v['w']:.4f}")
    print(f"[reward_weights] {out} sha256_16 {body['sha256_16']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
