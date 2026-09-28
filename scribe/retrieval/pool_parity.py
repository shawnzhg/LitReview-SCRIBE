#!/usr/bin/env python3
"""Checks the pool-service configuration: static records the file hashes and environment, health
checks a /healthz response, and routes checks that a call log used only allowed routes. Usage:
python pool_parity.py static|health|routes <args>."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness" / "tools"))
from selection_rule import EDGES_META_SHA256
from pool_retrieval_tool import PIN_N_DOCS, PIN_STATS_FINGERPRINT

SERVICE = Path(__file__).resolve().parents[2] / "biolitbench" / "pool" / "pool_service.py"
EDGES_NPY_BYTES = {"fwd_indptr.npy": 318208504, "fwd_val.npy": 1401925400, "bwd_indptr.npy": 318208504,
                   "bwd_val.npy": 1401925400, "ncit.npy": 159104312}
HEALTH = {"ok": True, "edges_present": True, "edges_kind": "dense-csr", "edges_n_input": 350699023,
          "has_n_citation": True, "citation_count_mode": "global", "s2_pdf_mode": "local",
          "stats_fingerprint": PIN_STATS_FINGERPRINT, "n_docs": PIN_N_DOCS}


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def static(service, env=None):
    env = os.environ if env is None else env
    bad, rec = [], {"schema": "pool_parity/1.0", "service": service}
    try:
        rec["service_md5"] = md5(service)
        rec["pool_common_md5"] = md5(Path(service).parent / "pool_common.py")
    except OSError as e:
        bad.append(f"service files unreadable: {e}")
    if Path(service).resolve() != SERVICE:
        bad.append(f"service {service} != the repository's pool service {SERVICE}")
    edges = env.get("POOL_EDGES") or ""
    rec["env"] = {k: env.get(k) for k in ("POOL_EDGES", "POOL_GOLD_DIR", "POOL_ALLOWLIST_DIR")}
    if not edges or not Path(edges).is_dir():
        bad.append(f"POOL_EDGES={edges!r} is not a directory (the citation edge store)")
    if not env.get("POOL_GOLD_DIR") or not Path(env["POOL_GOLD_DIR"]).is_dir():
        bad.append(f"POOL_GOLD_DIR={env.get('POOL_GOLD_DIR')!r} is not a directory (the gold records naming each review)")
    if env.get("POOL_ALLOWLIST_DIR"):
        bad.append("POOL_ALLOWLIST_DIR is set: the same-pool service runs without a reference-list restriction")
    rec["edges_path"] = edges
    try:
        rec["edges_meta_sha256"] = hashlib.sha256((Path(edges) / "meta.json").read_bytes()).hexdigest()
        if rec["edges_meta_sha256"] != EDGES_META_SHA256:
            bad.append(f"edge store meta.json sha256 {rec['edges_meta_sha256']} != {EDGES_META_SHA256}")
        rec["edges_n_input_meta"] = json.loads((Path(edges) / "meta.json").read_text()).get("n_edges_input")
    except (OSError, ValueError) as e:
        bad.append(f"edge store meta.json unreadable: {e}")
    rec["edges_npy_bytes"] = {}
    for f, n in EDGES_NPY_BYTES.items():
        try:
            rec["edges_npy_bytes"][f] = (Path(edges) / f).stat().st_size
        except OSError:
            rec["edges_npy_bytes"][f] = None
        if rec["edges_npy_bytes"][f] != n:
            bad.append(f"edge store {f} apparent size {rec['edges_npy_bytes'][f]} != {n}")
    rec["expected_healthz"] = HEALTH
    rec["problems"] = bad
    return rec


def health(h, n_tasks=None, env=None):
    env = os.environ if env is None else env
    edges = env.get("POOL_EDGES") or ""
    bad = [f"/healthz {k}={h.get(k)!r} != {v!r}" for k, v in HEALTH.items() if h.get(k) != v]
    try:
        if not edges or Path(str(h.get("edges_path") or "")).resolve() != Path(edges).resolve():
            bad.append(f"/healthz edges_path {h.get('edges_path')!r} != POOL_EDGES {edges!r}")
    except OSError as e:
        bad.append(f"/healthz edges_path unresolvable: {e}")
    if n_tasks is not None and h.get("n_tasks") != n_tasks:
        bad.append(f"/healthz n_tasks={h.get('n_tasks')!r} != {n_tasks}")
    return bad


ALLOWED_ROUTES = ("plain_search", "s2_paper")
ALLOWED_PAPER_FIELDS = {"title", "abstract", "year"}


def routes(paths, min_rows=1):
    bad, counts, fields, n = [], {}, {}, 0
    if not paths:
        bad.append("no service log given")
    for p in paths:
        try:
            lines = Path(p).read_text().splitlines()
        except OSError as e:
            bad.append(f"service log unreadable: {e}")
            continue
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                bad.append(f"{p}:{i + 1}: unparsable row")
                continue
            n += 1
            rt = r.get("route")
            counts[rt] = counts.get(rt, 0) + 1
            if rt not in ALLOWED_ROUTES:
                if counts[rt] == 1:
                    bad.append(f"{p}:{i + 1}: route {rt!r} (task {r.get('task')!r}) is not /search or /paper")
                continue
            if rt == "s2_paper":
                f = str((r.get("params") or {}).get("fields") or "")
                fields[f] = fields.get(f, 0) + 1
                extra = {x.strip().split(".")[0] for x in f.split(",") if x.strip()} - ALLOWED_PAPER_FIELDS
                if extra and fields[f] == 1:
                    bad.append(f"{p}:{i + 1}: /paper fields {f!r} ask for {sorted(extra)} (only title,abstract,year)")
    if n < min_rows:
        bad.append(f"{n} rows in the service log(s) < --min-rows {min_rows}")
    rec = {"schema": "pool_routes/1.0", "logs": list(paths),
           "n_rows": n, "routes": counts, "paper_fields": fields, "allowed_routes": list(ALLOWED_ROUTES),
           "allowed_paper_fields": sorted(ALLOWED_PAPER_FIELDS), "problems": bad}
    return rec, bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--service", required=True)
    s.add_argument("--out")
    h = sub.add_parser("health")
    h.add_argument("healthz")
    h.add_argument("--n-tasks", type=int)
    h.add_argument("--label", default="pool")
    r = sub.add_parser("routes")
    r.add_argument("logs", nargs="*")
    r.add_argument("--min-rows", type=int, default=1)
    r.add_argument("--out")
    r.add_argument("--label", default="pool")
    a = ap.parse_args(argv)
    if a.cmd == "routes":
        rec, bad = routes(a.logs, a.min_rows)
        if a.out:
            Path(a.out).write_text(json.dumps(rec, indent=1))
        print(f"### route guard ({a.label}): " + ("OK " if not bad else "REFUSED ")
              + json.dumps({"n_rows": rec["n_rows"], "routes": rec["routes"], "paper_fields": rec["paper_fields"]}))
        for b in bad[:20]:
            print(f"FATAL: route guard: {a.label}: {b}")
        return 1 if bad else 0
    if a.cmd == "static":
        rec = static(a.service)
        if a.out:
            Path(a.out).write_text(json.dumps(rec, indent=1))
        print(f"### pool parity (static): service md5 {str(rec.get('service_md5'))[:8]} common "
              f"{str(rec.get('pool_common_md5'))[:8]} edges meta sha256 {str(rec.get('edges_meta_sha256'))[:16]} "
              f"env {rec['env']}")
        for b in rec["problems"]:
            print(f"FATAL: pool parity: {b}")
        return 1 if rec["problems"] else 0
    try:
        hz = json.load(open(a.healthz))
    except (OSError, ValueError) as e:
        print(f"FATAL: pool parity: {a.label} /healthz unreadable ({e})")
        return 1
    bad = health(hz, a.n_tasks)
    keys = list(HEALTH) + ["edges_path", "n_tasks", "index_dir"]
    print(f"### pool parity (/healthz {a.label}): " + ("OK " if not bad else "REFUSED ")
          + json.dumps({k: hz.get(k) for k in keys}))
    for b in bad:
        print(f"FATAL: pool parity: {a.label}: {b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
