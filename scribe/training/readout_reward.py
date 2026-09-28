"""Planning-exit reward relative to the initial carrier: the weighted mean of tanh-scaled z differences
per readout, read from a hash-checked targets file."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def load_targets(path: str | Path) -> dict:
    p = Path(path)
    t = json.loads(p.read_text())
    body = {k: v for k, v in t.items() if k != "sha256_16"}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    if t.get("sha256_16") != h:
        raise ValueError(f"readout targets {p}: sha256_16 {t.get('sha256_16')} != recomputed {h}")
    for name, rs in (t.get("window_sets") or {}).items():
        if not rs:
            raise ValueError(f"readout targets {p}: window set {name!r} is empty")
    t["_path"] = str(p)
    t["_sha256_16"] = h
    return t


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(float(x))


def compute(cand: dict, base: dict, targets: dict, window: str) -> dict | None:
    sets = targets.get("window_sets") or {}
    if window not in sets:
        raise KeyError(f"window {window!r} not in targets' window_sets {sorted(sets)}")
    spec = targets["window_readouts"][window]
    parts, num, den = {}, 0.0, 0.0
    for r in sets[window]:
        zc, zb = cand.get(r), base.get(r)
        if not (_finite(zc) and _finite(zb)):
            parts[r] = {"skipped": "missing_on_one_side"}
            continue
        s, w, mode = float(spec[r]["s"]), float(spec[r]["w"]), spec[r]["mode"]
        d = float(zc) - float(zb)
        g = math.tanh(d / s)
        if mode == "guard":
            g = min(0.0, g)
        parts[r] = {"z": float(zc), "z0": float(zb), "d": d, "s": s, "w": w, "mode": mode, "g": g}
        num += w * g
        den += w
    if den <= 0:
        return None
    R = num / den
    return {"reward01": 0.5 * (1.0 + R), "R": R, "n_scored": sum(1 for p in parts.values() if "g" in p),
            "window": window, "targets_sha256_16": targets.get("_sha256_16"), "parts": parts}
