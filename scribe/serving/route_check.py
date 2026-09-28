#!/usr/bin/env python3
"""Checks that the server of each stage (synthesis, planning, writing) staged the configured carrier
and serves the configured model with prefix caching off, and writes route_check.json. Usage:
python route_check.py --config <config.json> --sh <server dir>."""

import argparse, json, os, re, sys, tempfile


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--sh", required=True)
    a = ap.parse_args()
    from kvskill.export import export_from_artifact
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from carrier_digest import canon_sha
    c = json.load(open(a.config))
    exp_cache, out, bad = {}, {}, []
    tmp = tempfile.mkdtemp(prefix="route_check_")
    for stage in ("synthesis", "planning", "writing"):
        th = c["carriers"][stage]
        k = [i for i, s in enumerate(c["servers"]) if stage in s["stages"]][0]
        if th not in exp_cache:
            p = os.path.join(tmp, f"{len(exp_cache)}.safetensors"); export_from_artifact(th, p); exp_cache[th] = canon_sha(p)
        log = open(os.path.join(a.sh, f"server_{k}.log"), errors="replace").read()
        lines = re.findall(r"\[carrier_digest\] staged canon=([0-9a-f]{64}) file_sha256=[0-9a-f]{64} path=(\S+) buffers=([0-9a-f]{64})", log)
        row = {"server": k, "theta": th, "expected_sha256": exp_cache[th], "n_staged_lines": len(lines)}
        if len(lines) != 1:
            bad.append(f"{stage}: server {k} printed {len(lines)} [carrier_digest] staged lines (need exactly 1)")
        else:
            row["served_sha256"], row["served_path"], row["buffers_sha256"] = lines[0]
            if row["served_sha256"] != exp_cache[th]:
                bad.append(f"{stage}: server {k} staged {row['served_sha256'][:16]} but the stage's carrier {th} exports to {exp_cache[th][:16]}")
        pc = set(re.findall(r"enable_prefix_caching=(True|False)", log))
        row["prefix_caching"] = sorted(pc)
        if pc != {"False"}:
            bad.append(f"{stage}: server {k} prefix caching {sorted(pc) or 'not reported'} (must be exactly False)")
        want_model = (c.get("models") or {}).get(stage)
        if want_model:
            ms = set(re.findall(r"model='([^']+)'", log))
            row["served_models"] = sorted(ms)
            if ms != {want_model}:
                bad.append(f"{stage}: server {k} served model(s) {sorted(ms)} != config models[{stage}] {want_model}")
        out[stage] = row
    rec = {"ok": not bad, "stages": out, "problems": bad}
    json.dump(rec, open(os.path.join(a.sh, "route_check.json"), "w"), indent=1)
    for s, r in out.items():
        print(f"### route check {s}: server {r['server']} served {r.get('served_sha256', '?')[:16]} expected {r['expected_sha256'][:16]}")
    if bad:
        print("FATAL: carrier identity check failed:\n  " + "\n  ".join(bad)); sys.exit(95)
    print("### route check OK: every stage's server staged exactly its own carrier")


if __name__ == "__main__":
    main()
