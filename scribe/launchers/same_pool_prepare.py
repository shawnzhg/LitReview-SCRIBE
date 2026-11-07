#!/usr/bin/env python3
"""Pins, copies and verifies the acquisition units of a same-pool generation run. Usage: python
same_pool_prepare.py pin|materialize|verify <options>."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time

GEN_OUTPUTS = {"entry_evidence_bundle.json", "binding.json", "synthesis_graph.json", "outline_plan.json",
               "report_artifact.json", "integrity.json", "resources.json", "delivery.json"}
CUT = {"trace.jsonl", "manifests.jsonl"}
SCHEMA_PINS = "same_pool_src_pins/1"
SCHEMA_PROV = "same_pool_provenance/1"
MARKER = re.compile(r"^[\s\[\(<{#*_-]*new[\s_-]*paragraph[\s\]\)>}.:#*_-]*$", re.I)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


SEEDS = (0, 1, 2)


def src_unit(root: str, task: str, system: str, seed: int) -> str:
    return os.path.join(root, system, "native_chain", task, f"seed{int(seed)}")


def cut_acq(path: str, name: str) -> bytes:
    raw = open(path, "rb").read()
    lines = raw.splitlines(keepends=True)
    keep, seen_other = [], False
    for ln in lines:
        if not ln.strip():
            raise ValueError(f"{path}: blank line")
        w = json.loads(ln).get("window")
        if w == "acquisition":
            if seen_other:
                raise ValueError(f"{path}: an acquisition row after a non-acquisition row (not a prefix)")
            keep.append(ln)
        else:
            seen_other = True
    if not keep:
        raise ValueError(f"{path}: no acquisition rows")
    if name == "manifests.jsonl" and len(keep) != 1:
        raise ValueError(f"{path}: {len(keep)} acquisition manifest lines (need exactly 1)")
    out = b"".join(keep)
    if not out.endswith(b"\n"):
        raise ValueError(f"{path}: acquisition part does not end with a newline")
    return out


def unit_record(u: str) -> dict:
    if not os.path.isdir(u):
        raise ValueError(f"source unit {u} missing")
    files = {}
    for n in sorted(os.listdir(u)):
        p = os.path.join(u, n)
        if os.path.islink(p) or not os.path.isfile(p):
            raise ValueError(f"{u}/{n}: not a regular file (links / dirs are refused)")
        if n in GEN_OUTPUTS:
            continue
        if n in CUT:
            b = cut_acq(p, n)
            files[n] = {"sha256": sha256_bytes(b), "bytes": len(b), "cut": "acquisition", "src_sha256_full": sha256_file(p)}
        else:
            files[n] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    for need in ("evidence_bundle.json", "pool_doccache.jsonl", "trace.jsonl", "manifests.jsonl", "task_spec.json"):
        if need not in files:
            raise ValueError(f"{u}: no {need}")
    eb = json.load(open(os.path.join(u, "evidence_bundle.json")))
    if not eb.get("papers"):
        raise ValueError(f"{u}: evidence bundle has no papers")
    return {"files": files, "bundle_content_hash": eb.get("content_hash"), "n_papers": len(eb.get("papers") or []),
            "n_evidence": len(eb.get("evidence") or []), "retrieval_status": eb.get("retrieval_status")}


def cmd_pin(a):
    tasks = [t for t in open(a.tasks).read().split() if t]
    rec = {"schema": SCHEMA_PINS, "src_root": os.path.abspath(a.src_root), "seed": a.seed,
           "tasks_file": os.path.abspath(a.tasks), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "src_system": a.src_system, "tasks": {}}
    for t in tasks:
        rec["tasks"][t] = unit_record(src_unit(a.src_root, t, a.src_system, a.seed))
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists (pins are written once)")
    json.dump(rec, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"pinned {len(tasks)} units -> {a.out} sha256 {sha256_file(a.out)}")


def cmd_materialize(a):
    pins = json.load(open(a.pins))
    if pins.get("schema") != SCHEMA_PINS:
        raise SystemExit(f"{a.pins}: not {SCHEMA_PINS}")
    if a.pins_sha256 and sha256_file(a.pins) != a.pins_sha256:
        print(f"REFUSED: pins file sha256 {sha256_file(a.pins)} != pinned {a.pins_sha256}")
        sys.exit(3)
    if not a.runs_root:
        raise SystemExit("--runs-root is required (default: $SCRIBE_RUNS_ROOT)")
    tasks = [t for t in open(a.tasks).read().split() if t]
    lvl = os.path.join(a.runs_root, f"level{a.level}")
    if not os.path.isdir(lvl):
        raise SystemExit(f"{lvl} must exist (claimed by run_same_pool.sh) before materialize")
    bad, recs = [], {}
    for t in tasks:
        pin = (pins["tasks"] or {}).get(t)
        if pin is None:
            bad.append(f"{t}: not in the pins file")
            continue
        try:
            r = unit_record(src_unit(pins["src_root"], t, pins["src_system"], pins["seed"]))
        except (ValueError, OSError) as e:
            bad.append(f"{t}: {e}")
            continue
        if set(r["files"]) != set(pin["files"]):
            bad.append(f"{t}: file set differs from the pins: +{sorted(set(r['files']) - set(pin['files']))} "
                       f"-{sorted(set(pin['files']) - set(r['files']))}")
        for n, f in r["files"].items():
            p = pin["files"].get(n)
            if p and p["sha256"] != f["sha256"]:
                bad.append(f"{t}/{n}: sha256 {f['sha256'][:16]} != pinned {p['sha256'][:16]}")
        recs[t] = r
    if bad:
        print(f"REFUSED: {len(bad)} source problem(s) vs pins {a.pins}:")
        for b in bad[:20]:
            print("  " + b)
        sys.exit(3)
    prov = {"schema": SCHEMA_PROV, "level": int(a.level), "runs_root": os.path.abspath(a.runs_root),
            "src_root": pins["src_root"], "seed": pins["seed"], "pins": os.path.abspath(a.pins),
            "pins_sha256": sha256_file(a.pins), "tasks": {}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    prov["system"] = a.system
    for t in tasks:
        su = src_unit(pins["src_root"], t, pins["src_system"], pins["seed"])
        du = os.path.join(lvl, a.system, "native_chain", t, f"seed{int(pins['seed'])}")
        if os.path.exists(du):
            print(f"REFUSED: {du} exists (a level is materialised once)")
            sys.exit(3)
        os.makedirs(du)
        for n, f in recs[t]["files"].items():
            sp, dp = os.path.join(su, n), os.path.join(du, n)
            if n in CUT:
                with open(dp, "wb") as fh:
                    fh.write(cut_acq(sp, n))
            else:
                shutil.copyfile(sp, dp)
            got = sha256_file(dp)
            if got != f["sha256"]:
                print(f"REFUSED: copy {dp} sha256 {got[:16]} != source {f['sha256'][:16]}")
                sys.exit(3)
        prov["tasks"][t] = {"src_unit": su, "dest_unit": du, **recs[t]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(prov, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"materialised {len(tasks)} units at {lvl} (every copied file == pinned source sha256) -> {a.out}")


def cmd_verify(a):
    prov = json.load(open(a.provenance))
    if prov.get("schema") != SCHEMA_PROV:
        raise SystemExit(f"{a.provenance}: not {SCHEMA_PROV}")
    tasks = [t for t in open(a.tasks).read().split() if t] if a.tasks else sorted(prov["tasks"])
    bad = []
    for t in tasks:
        r = prov["tasks"].get(t)
        if r is None:
            bad.append(f"{t}: not in provenance")
            continue
        du = r["dest_unit"]
        for n, f in r["files"].items():
            p = os.path.join(du, n)
            if not os.path.isfile(p) or os.path.islink(p):
                bad.append(f"{t}/{n}: missing")
                continue
            if n in CUT:
                with open(p, "rb") as fh:
                    head = fh.read(f["bytes"])
                if sha256_bytes(head) != f["sha256"]:
                    bad.append(f"{t}/{n}: copied prefix changed")
            elif sha256_file(p) != f["sha256"]:
                bad.append(f"{t}/{n}: sha256 changed (was {f['sha256'][:16]})")
        if a.phase == "pre":
            present = sorted(n for n in os.listdir(du) if n in GEN_OUTPUTS)
            if present:
                bad.append(f"{t}: generation output already present before generation: {present}")
            for n in CUT:
                if os.path.getsize(os.path.join(du, n)) != r["files"][n]["bytes"]:
                    bad.append(f"{t}/{n}: grew before generation")
        if a.phase == "post" and not os.path.exists(os.path.join(du, "report_artifact.json")):
            bad.append(f"{t}: no report_artifact.json after generation")
        rp = os.path.join(du, "report_artifact.json")
        if a.phase in ("post", "any") and os.path.exists(rp):
            ra = json.load(open(rp))
            mk = [x.get("sentence_id") for sec in ra.get("sections") or [] for x in sec.get("sentences") or []
                  if MARKER.match(str(x.get("text") or "").strip())]
            if mk:
                bad.append(f"{t}: {len(mk)} paragraph-marker text(s) stored as report sentences ({mk[:3]})")
    out = {"schema": "same_pool_verify/1", "phase": a.phase, "provenance": a.provenance, "n_tasks": len(tasks),
           "problems": bad, "ok": not bad, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"verify[{a.phase}] {len(tasks)} units: {len(bad)} problem(s)" + (": " + "; ".join(bad[:5]) if bad else ""))
    sys.exit(3 if bad else 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pin")
    p.add_argument("--src-root", required=True, help="the source acquisition level dir (<runs>/level<n>)")
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--src-system", default="SCRIBE", help="the source level's system dir")
    p.add_argument("--seed", type=int, choices=SEEDS, default=0, help="the source level's seed")
    m = sub.add_parser("materialize")
    m.add_argument("--pins", required=True)
    m.add_argument("--pins-sha256", default="")
    m.add_argument("--tasks", required=True)
    m.add_argument("--level", required=True, type=int)
    m.add_argument("--system", default="SCRIBE", help="the destination system dir")
    m.add_argument("--runs-root", default=os.environ.get("SCRIBE_RUNS_ROOT") or "")
    m.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--provenance", required=True)
    v.add_argument("--tasks", default="")
    v.add_argument("--phase", choices=["pre", "post", "any"], default="any")
    v.add_argument("--out", default="")
    a = ap.parse_args(argv)
    {"pin": cmd_pin, "materialize": cmd_materialize, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
