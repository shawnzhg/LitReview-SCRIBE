"""Adds one arm's planning-axis rows to the outline-axis rows file without rebuilding the other
arms. Usage: python -m windowbench.outline_axis_add --tag <family>."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from . import config as C
from . import outline_axis as OA
from . import roster as R

CANON = C.SOURCES["outline_axis_rows"]
READOUTS = OA.READOUTS


def _sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p)] if p.exists() else []


def coverage(rows: list[dict]) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for d in rows:
        c = out.setdefault((d["obs"], d["system"]), {"n": 0, "lex": 0, "emb": 0})
        c["n"] += 1
        for r, k in (("outline_title_f1_lex", "lex"), ("outline_title_f1_emb", "emb")):
            if d.get(r) is not None:
                c[k] += 1
    return out


def keys_of(tag: str) -> list[str]:
    if tag in R.FAMILIES:
        return list(R.FAMILIES[tag])
    if tag in R.SYSTEMS:
        return [tag]
    raise SystemExit(f"unknown arm {tag!r}; declare it in the roster-extra file ($WINDOWBENCH_ROSTER_EXTRA) first")


def merge(canon: Path, new_rows: list[dict], systems: list[str]) -> tuple[list[dict], dict]:
    want = set(systems)
    old = read_rows(canon)
    kept = [d for d in old if d["system"] not in want]
    fresh = [d for d in new_rows if d["system"] in want]
    merged = kept + fresh
    before, after = {d["system"] for d in old}, {d["system"] for d in merged}
    lost = sorted(before - after)
    if lost:
        raise SystemExit(f"refusing to write: the merge would drop system keys {lost} that {canon} already had")
    return merged, {"rows_before": len(old), "rows_kept": len(kept), "rows_new": len(fresh),
                    "rows_after": len(merged), "systems_before": len(before), "systems_after": len(after),
                    "systems_added": sorted(after - before)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None, help="family name; its ccbench keys are built")
    ap.add_argument("--systems", default=None, help="comma-separated ccbench system keys instead of --tag")
    ap.add_argument("--canon", default=str(CANON), help=f"the file load.py reads (default {CANON})")
    ap.add_argument("--scratch", default=None, help="scratch build directory (default <canon dir>/_add_<tag>)")
    a = ap.parse_args(argv)

    canon = Path(a.canon)
    systems = ([s for s in a.systems.split(",") if s] if a.systems else keys_of(a.tag) if a.tag else None)

    if not systems:
        ap.error("--tag or --systems is required")
    t0 = time.time()
    scratch = Path(a.scratch) if a.scratch else canon.parent / f"_add_{a.tag or 'keys'}"
    print(f"building the planning axis for {len(systems)} key(s): {', '.join(systems)}")
    OA.build(systems, verbose=True, out_dir=scratch, band=False)
    new_rows = read_rows(scratch / "rows.jsonl")
    got = {d["system"] for d in new_rows} & set(systems)
    if not got:
        raise SystemExit(f"the builder produced no rows for {systems}: are the ccbench rollouts scored "
                         f"(out/rollouts/<key>/*.json) and is the arm declared in the roster?")
    merged, stats = merge(canon, new_rows, systems)
    with open(canon, "w") as f:
        for d in merged:
            f.write(json.dumps(d) + "\n")
    cov = coverage(merged)
    for s in systems:
        for obs in ("system", "planning"):
            c = cov.get((obs, s))
            print(f"  {obs:10s} {s:28s} " + (f"n={c['n']:4d} lex={c['lex']:4d} emb={c['emb']:4d}" if c else "no rows"))
    meta = {"builder": "windowbench/outline_axis_add.py", "built": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tag": a.tag, "systems": systems, "readouts": list(READOUTS),
            "counts": stats, "sha256_16_of_output_file": _sha16(canon)}
    (canon.parent / "rows.jsonl.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"{canon}: {stats['rows_before']} -> {stats['rows_after']} rows, "
          f"{stats['systems_before']} -> {stats['systems_after']} system keys "
          f"(added {stats['systems_added']}), sha256_16={meta['sha256_16_of_output_file']} in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
