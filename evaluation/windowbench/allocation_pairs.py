"""Normalises the evidence-allocation scores against the peer band and certifies every admissible
pair on the adjusted Rand index. Usage: python -m windowbench.allocation_pairs --out <dir>."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import decide as DEC
from . import fairness as F
from . import roster as R

ALLOC_READS = ("alloc_ari",)
DESC_READS = ("alloc_coloc_f1", "alloc_pair_agreement", "alloc_coloc_recall", "alloc_coloc_precision",
              "alloc_evidence_placed", "alloc_spread")
ENTRY_WINDOW = "writing"
OBS = "allocation"


def _dataset_of(system: str) -> str | None:
    for ds, keys in R.DATASETS.items():
        if system in keys:
            return ds
    return None


def _collapse(row) -> float:
    ns, sp = row.get("n_sys_sections"), row.get("alloc_spread")
    if ns is None or not np.isfinite(ns) or ns <= 0:
        return 1.0
    return float(sp / ns) if np.isfinite(sp) else np.nan


def calibrate(S: pd.DataFrame, H: pd.DataFrame, reads: tuple[str, ...]) -> pd.DataFrame:
    C.bootstrap_env()
    from ccbench.rankability.normalise import calibration, normalise
    Hl = H.melt(id_vars=["task", "source_review"], value_vars=list(reads), var_name="readout", value_name="value")
    cal = calibration(Hl)
    Sl = S.assign(collapse=[_collapse(r) for _, r in S.iterrows()]).melt(
        id_vars=["system", "task", "collapse"], value_vars=list(reads), var_name="readout", value_name="value")
    Sl["direction"] = "quality"
    Z = normalise(Sl, cal).rename(columns={"q_lo": "q10", "q_hi": "q90"})
    return Z[["system", "task", "readout", "value", "q10", "q90", "own", "z", "z_note", "collapse"]]


def pairs(Z: pd.DataFrame, systems: list[str], D, reads: tuple[str, ...], alpha: float | None = None) -> pd.DataFrame:
    from .axes import bot_tasks
    alpha = alpha if alpha is not None else C.REGISTERED["alpha"]
    regs = {}
    for a, b in itertools.combinations(systems, 2):
        regs[(a, b)] = F.regime(a, b, ENTRY_WINDOW, None, D.radii, D.eps_fair)
    cond = {s: R.SYSTEMS[s]["provenance"] for s in systems}
    table = {c: DEC.table_pairs(sum(1 for s in systems if cond[s] == c)) for c in set(cond.values())}
    n_reads = max(1, len(reads))
    wide = {r: Z[Z.readout == r].pivot_table(index="system", columns="task", values="z") for r in reads}
    rows = []
    for (a, b), reg in regs.items():
        ds_a, ds_b = _dataset_of(a), _dataset_of(b)
        n_pairs = table[cond[a]]
        for r in reads:
            W = wide[r]
            base = {"obs": OBS, "readout": r, "a": a, "b": b, "regime": reg["regime"], "reason": reg["reason"],
                    "dataset_a": ds_a, "dataset_b": ds_b, "n_pairs_in_family": n_pairs,
                    "n_readouts_in_family": n_reads, "d_eff": alpha / (n_pairs * n_reads),
                    "role": "secondary_descriptive"}
            if a not in W.index or b not in W.index:
                rows.append({**base, "status": "not_scored", "certified": False, "winner": None})
                continue
            sa, sb = W.loc[a], W.loc[b]
            base["n_not_observable_a"] = int(sa.isna().sum())
            base["n_not_observable_b"] = int(sb.isna().sum())
            ba, bb = bot_tasks(D, a, ENTRY_WINDOW), bot_tasks(D, b, ENTRY_WINDOW)
            eps = reg["eps_outside"] if reg["regime"] in ("exact", "tolerance") else np.nan
            res = DEC.contrast(DEC.drop_failed(sa, ba), DEC.drop_failed(sb, bb),
                               D.clusters, alpha / (n_pairs * n_reads), eps=eps)
            if reg["regime"] not in ("exact", "tolerance"):
                res["certified"], res["status"], res["winner"] = False, "descriptive", None
            rows.append({**base, **res})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(C.OUT_DEFAULT / "allocation"))
    ap.add_argument("--alpha", type=float, default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    S = pd.read_csv(out / "allocation_scores.csv")
    H = pd.read_csv(out / "allocation_human.csv")
    reads = ALLOC_READS + DESC_READS
    Z = calibrate(S, H, reads)
    Z.to_csv(out / "allocation_calibrated.csv", index=False)
    from .load import Data
    D = Data()
    systems = [s for s in S.system.unique() if s in R.SYSTEMS]
    P = pairs(Z, systems, D, ALLOC_READS, alpha=a.alpha)
    P.to_csv(out / "allocation_pairs.csv", index=False)
    print(f"{int(P.certified.fillna(False).astype(bool).sum()) if len(P) else 0} certified of {len(P)} cells")
    print(f"wrote {out}/allocation_calibrated.csv, allocation_pairs.csv")


if __name__ == "__main__":
    main()
