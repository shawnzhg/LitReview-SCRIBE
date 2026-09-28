"""Calibrates every readout against the human peer band that defines the fraction-of-human scale.
Usage: python -m ccbench.experiments.E10_dimensions."""

from __future__ import annotations

import argparse

import pandas as pd

from ccbench import paths
from ccbench.config import prereg
from ccbench.rankability.normalise import calibration


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    cfg = prereg()["calibration"]
    out = paths.out_dir("E10")
    df = pd.read_parquet(out / "unit_scores.parquet")
    H = df[df.panel == "H"]
    lo, hi = cfg["band_quantiles"]
    cal = calibration(H, lo, hi)
    cal.to_csv(out / "calibration.csv", index=False)
    print("calibration rows", len(cal))


if __name__ == "__main__":
    main()
