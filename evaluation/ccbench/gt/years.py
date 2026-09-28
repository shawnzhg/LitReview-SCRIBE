"""Offline lookup of publication year, venue and subfield for PMIDs."""

from __future__ import annotations

import functools
import json

from ccbench import paths


@functools.lru_cache(maxsize=1)
def ref_years() -> dict[str, int]:
    with open(paths.gt_manifest("ref_years.json")) as f:
        d = json.load(f)
    return {str(k): int(v) for k, v in d.items() if v is not None}


@functools.lru_cache(maxsize=1)
def subfields() -> dict[str, str]:
    with open(paths.gt_manifest("pmid_subfield.json")) as f:
        return {str(k): v for k, v in json.load(f).items()}


def year_of(pmid: str) -> int | None:
    return ref_years().get(str(pmid))


def is_post_cutoff(pmid: str, cutoff: int) -> bool | None:
    y = year_of(pmid)
    return None if y is None else y >= cutoff


def before_cutoff(pmid: str, cutoff: int) -> bool:
    y = year_of(pmid)
    return y is not None and y < cutoff
