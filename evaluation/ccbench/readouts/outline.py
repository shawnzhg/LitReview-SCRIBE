"""Heading normalisation and fuzzy one-to-one matching of output headings against human section
titles."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from ccbench.adapters.common import strip_title_number

GENERIC = {"introduction", "conclusion", "conclusions", "abstract", "references", "summary", "background", "overview", "discussion", "future directions", "outlook", "acknowledgements"}


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", strip_title_number(t).lower()).strip(" :.-")


def match_titles(sys_titles: list[str], hum_titles: list[str], threshold: int) -> tuple[int, list[tuple[str, str, float]]]:
    pairs = []
    for h in hum_titles:
        for s in sys_titles:
            sc = fuzz.token_set_ratio(h, s)
            if sc >= threshold:
                pairs.append((sc, h, s))
    pairs.sort(reverse=True)
    used_h, used_s, out = set(), set(), []
    for sc, h, s in pairs:
        if h in used_h or s in used_s:
            continue
        used_h.add(h)
        used_s.add(s)
        out.append((h, s, sc))
    return len(used_h), out
