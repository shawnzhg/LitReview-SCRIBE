"""Loads the pre-registered analysis constants from prereg.yaml."""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def prereg() -> dict:
    with open(HERE / "prereg.yaml") as f:
        return yaml.safe_load(f)
