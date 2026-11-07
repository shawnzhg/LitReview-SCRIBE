#!/usr/bin/env python
"""Builds a scoring overlay of our same-pool runs in which an acquisition stopped at its document
budget counts as a normal exit when its bundle, hashes, audits and later windows pass. Usage:
python budget_stop_overlay.py --level <n> --root <dir> --runs-root <dir>."""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

RUNNERS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scribe", "harness", "runners"))
sys.path.insert(0, RUNNERS)
from common import content_hash

SCHEMA = "overlay_ledger/1.0"
FROM, TO = "budget_exhausted", "ok"
DOWNSTREAM = ("synthesis", "planning", "writing")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def level_checks(src_lvl, L):
    bad, info, audits = [], {}, []
    marks = sorted(glob.glob(os.path.join(src_lvl, "_level_infra_failure_*.json")) + glob.glob(os.path.join(src_lvl, "_level_*failed*.json")))
    if marks:
        bad.append(f"failure markers: {[os.path.basename(m) for m in marks[:3]]}")
    okp = os.path.join(src_lvl, "_level_acq_audit_ok.json")
    try:
        ok = json.load(open(okp))
    except Exception as e:
        return bad + [f"no acquisition audit record {okp} ({type(e).__name__})"], info, audits
    info["audit_ok_record"] = {"path": okp, "md5": md5(okp), **{k: ok.get(k) for k in ("ok", "level", "arm", "job", "audit", "audit_md5", "n_units_audited", "no_bundle", "seeds", "retrieval_budget")}}
    if ok.get("ok") is not True:
        bad.append(f"_level_acq_audit_ok ok={ok.get('ok')}")
    try:
        if int(ok.get("level")) != L:
            bad.append(f"_level_acq_audit_ok level {ok.get('level')} != {L}")
    except Exception:
        bad.append(f"_level_acq_audit_ok level {ok.get('level')!r} unreadable")
    if ok.get("no_bundle"):
        bad.append(f"_level_acq_audit_ok no_bundle {ok.get('no_bundle')[:3]}")
    ap = ok.get("audit") or ""
    if not os.path.isfile(ap):
        return bad + [f"audit file {ap!r} missing"], info, audits
    if md5(ap) != ok.get("audit_md5"):
        bad.append(f"audit md5 {md5(ap)} != recorded {ok.get('audit_md5')}")
    rdir = os.path.dirname(ap)
    files = sorted(glob.glob(os.path.join(rdir, "*audit*.json")))
    if ap not in files:
        bad.append("the recorded audit is not among the run dir's *audit*.json")
    info["run_dir"] = rdir
    info["audits"] = {}
    for f in files:
        n = os.path.basename(f)
        try:
            a = json.load(open(f))
        except Exception as e:
            bad.append(f"{n} unreadable ({type(e).__name__})")
            continue
        info["audits"][n] = {"md5": md5(f), "n_findings": a.get("n_findings"), "findings": len(a.get("findings") or []),
                             "level": a.get("level"), "seeds": a.get("seeds")}
        if "n_findings" not in a and "findings" not in a:
            bad.append(f"{n}: unrecognised audit shape (no n_findings/findings)")
        if a.get("n_findings") not in (None, 0) or a.get("findings"):
            bad.append(f"{n}: {a.get('n_findings')} level-scope findings")
        if a.get("level") is not None and int(a["level"]) != L:
            bad.append(f"{n}: level {a['level']} != {L}")
        if not isinstance(a.get("units"), dict) and not isinstance(a.get("rows"), list):
            bad.append(f"{n}: no per-unit units{{}} / rows[]")
        audits.append((n, a))
    recp = os.path.join(rdir, "level_record.json")
    if os.path.exists(recp):
        try:
            r = json.load(open(recp))
            info["acq_record"] = {k: r.get(k) for k in ("rc", "level", "acq_audit_ok", "retrieval_budget", "job", "arm")}
            if r.get("acq_audit_ok") is not True:
                bad.append("level_record says acq_audit_ok is not true")
        except Exception as e:
            bad.append(f"level_record.json unreadable ({type(e).__name__})")
    return bad, info, audits


def unit_audit(audits, L, t, s, n_papers, system):
    bad, acq_unit = [], None
    for n, a in audits:
        if a.get("seeds") is not None and str(s) not in [str(x) for x in (a["seeds"] if isinstance(a["seeds"], list) else str(a["seeds"]).split(","))]:
            bad.append(f"{n}: seed {s} not audited")
        if isinstance(a.get("units"), dict):
            u = a["units"].get(t)
            if u is None:
                bad.append(f"{n}: unit not audited")
                continue
            if u.get("n_findings") not in (None, 0) or u.get("findings"):
                bad.append(f"{n}: unit has findings {u.get('n_findings')}")
            if n.startswith("acquisition_audit"):
                acq_unit = u
                if u.get("n_papers") != n_papers:
                    bad.append(f"{n}: audited n_papers {u.get('n_papers')} != bundle papers {n_papers}")
        else:
            rows = [r for r in a["rows"] if r.get("task") == t]
            if len(rows) != 1:
                bad.append(f"{n}: {len(rows)} rows for the unit")
                continue
            r = rows[0]
            if r.get("unit") and not str(r["unit"]).rstrip("/").endswith(f"/level{L}/{system}/native_chain/{t}/seed{s}"):
                bad.append(f"{n}: row unit {r['unit']} is another level/unit")
            if r.get("findings"):
                bad.append(f"{n}: unit has {len(r['findings'])} findings")
    if acq_unit is None and not any(b.startswith("acquisition_audit") for b in bad):
        bad.append("no acquisition_audit unit record")
    return bad, acq_unit


def decide(u, L, t, s, level_ok, audits):
    row = {"task": t, "seed": s, "old_status": None, "new_status": None, "changed": False, "verdict": None, "reason": None}
    mp = os.path.join(u, "manifests.jsonl")
    if not os.path.isfile(mp):
        row.update(verdict="refused", reason="no manifests.jsonl")
        return row, None
    raw = open(mp, "rb").read()
    row["src_manifest_md5"] = hashlib.md5(raw).hexdigest()
    lines = raw.decode("utf-8").splitlines(keepends=True)
    mans = []
    for i, l in enumerate(lines):
        if l.strip():
            try:
                mans.append((i, json.loads(l)))
            except Exception:
                row.update(verdict="refused", reason=f"manifest line {i} unparseable")
                return row, raw
    acq = [(i, m) for i, m in mans if m.get("window") == "acquisition"]
    if len(acq) != 1:
        row.update(verdict="refused", reason=f"{len(acq)} acquisition entries (need exactly 1)")
        return row, raw
    ai, a = acq[0]
    old = a.get("status")
    row["old_status"] = row["new_status"] = old
    if old in (None, "ok"):
        row.update(verdict="unchanged", reason="acquisition status already ok")
        return row, raw
    if old != FROM:
        row.update(verdict="refused", reason=f"acquisition status {old!r} is not {FROM!r}")
        return row, raw
    bad, chk = [], {}
    bp = os.path.join(u, "evidence_bundle.json")
    b = None
    if not os.path.isfile(bp):
        bad.append("(a) no evidence_bundle.json")
    elif os.path.getsize(bp) == 0:
        bad.append("(a) evidence_bundle.json is empty")
    else:
        try:
            b = json.load(open(bp))
        except Exception as e:
            bad.append(f"(a) evidence_bundle.json unparseable ({type(e).__name__})")
    if b is not None:
        if not isinstance(b, dict) or not b.get("papers"):
            bad.append("(a) bundle has no papers")
        if not isinstance(b, dict) or not b.get("content_hash"):
            bad.append("(a) bundle has no content_hash (unsealed)")
        elif b["content_hash"] != content_hash(b):
            bad.append("(a) bundle content_hash != hash of its body (seal broken)")
        if isinstance(b, dict) and b.get("retrieval_status") != FROM:
            bad.append(f"(a) bundle retrieval_status {b.get('retrieval_status')!r} != {FROM!r}")
    xh = a.get("exit_hash")
    if not xh:
        bad.append("(b) acquisition manifest has no exit_hash")
    elif b is not None and isinstance(b, dict) and xh != b.get("content_hash"):
        bad.append("(b) exit_hash != bundle content_hash")
    ep = os.path.join(u, "entry_evidence_bundle.json")
    if os.path.isfile(ep):
        try:
            if json.load(open(ep)).get("content_hash") != xh:
                bad.append("(b) entry_evidence_bundle.json hash != acquisition exit_hash")
        except Exception as e:
            bad.append(f"(b) entry_evidence_bundle.json unreadable ({type(e).__name__})")
    if not level_ok:
        bad.append("(c) level audit record not ok")
    npap = len(b.get("papers") or []) if isinstance(b, dict) else -1
    sy = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.normpath(u)))))
    ab, acq_unit = unit_audit(audits, L, t, s, npap, system=sy)
    bad += [f"(c) {x}" for x in ab]
    others = [m for i, m in mans if i != ai]
    byw = {}
    for m in others:
        byw.setdefault(m.get("window"), []).append(m)
    for w in DOWNSTREAM:
        if len(byw.get(w, [])) != 1:
            bad.append(f"(d) {len(byw.get(w, []))} {w} entries (need 1)")
    for m in others:
        if m.get("status") != "ok":
            bad.append(f"(d) window {m.get('window')} status {m.get('status')!r}")
    if all(len(byw.get(w, [])) == 1 for w in DOWNSTREAM):
        chain = [xh] + [x for w in DOWNSTREAM for x in (byw[w][0].get("entry_hash"), byw[w][0].get("exit_hash"))]
        for k, (prev, nxt) in enumerate([(chain[0], chain[1]), (chain[2], chain[3]), (chain[4], chain[5])]):
            if not prev or prev != nxt:
                bad.append(f"(d) hash chain broken into {DOWNSTREAM[k]}")
    lim = []
    br = (b.get("budget_remaining") or {}) if isinstance(b, dict) else {}
    if br.get("document_opens") is not None and br["document_opens"] <= 0:
        lim.append("document_opens")
    if br.get("search_calls") is not None and br["search_calls"] <= 0:
        lim.append("search_calls")
    if acq_unit is not None:
        cap = (acq_unit.get("effective_budget") or {}).get("max_docs_read")
        if cap is not None and acq_unit.get("docs_read") is not None and acq_unit["docs_read"] >= cap:
            lim.append("docs_read")
    chk["budget_stop"] = lim
    chk["budget_remaining"] = br
    if "docs_read" not in lim:
        bad.append("(e) not a document-budget stop: docs_read < the unit's max_docs_read"
                   + (f" (exhausted instead: {lim})" if lim else " (no exhausted budget identified)"))
    row["checks"] = chk
    if bad:
        row.update(verdict="refused", reason="; ".join(bad))
        return row, raw
    orig = lines[ai]
    body, nl = (orig[:-1], "\n") if orig.endswith("\n") else (orig, "")
    if json.dumps(a) != body:
        row.update(verdict="refused", reason="acquisition manifest line does not round-trip through json.dumps (would rewrite other fields)")
        return row, raw
    new = dict(a)
    new["status"] = TO
    new["budget_stop_overlay"] = {"status_from": FROM, "budget_stop": lim, "bundle_content_hash": b["content_hash"],
                          "src_manifest_md5": row["src_manifest_md5"], "rule": "budget-stop overlay (a)-(e)"}
    lines[ai] = json.dumps(new) + nl
    row.update(new_status=TO, changed=True, verdict="changed", reason="budget stop on a sealed, audited bundle; all downstream windows ok (a)-(e)")
    return row, "".join(lines).encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--runs-root", required=True)
    a = ap.parse_args()
    L = a.level
    src_lvl = os.path.join(os.path.abspath(a.runs_root), f"level{L}")
    root = os.path.abspath(a.root)
    dst_lvl = os.path.join(root, f"level{L}")
    led_json = os.path.join(root, f"overlay_ledger_level{L}.json")
    led_tsv = os.path.join(root, f"overlay_ledger_level{L}.tsv")
    rr = os.path.realpath(root)
    for forbidden in (os.path.abspath(a.runs_root), os.path.realpath(a.runs_root)):
        if rr == forbidden or rr.startswith(forbidden.rstrip("/") + "/"):
            print(f"FATAL: --root {root} lies inside {forbidden}; the overlay is never written into a run tree")
            return 2
    if not os.path.isdir(src_lvl):
        print(f"FATAL: {src_lvl} missing")
        return 2
    if os.path.lexists(dst_lvl):
        print(f"FATAL: {dst_lvl} exists")
        return 2
    lbad, linfo, audits = level_checks(src_lvl, L)
    level_ok = not lbad
    units = sorted(glob.glob(os.path.join(src_lvl, "*", "native_chain", "*", "seed*")))
    units = [u for u in units if os.path.isdir(u) and not os.path.islink(u)]
    rows, payload = [], {}
    for u in units:
        t, s = os.path.basename(os.path.dirname(u)), os.path.basename(u)[4:]
        row, data = decide(u, L, t, s, level_ok, audits)
        row["unit"] = u
        rows.append(row)
        payload[u] = data
    cnt = {k: sum(1 for r in rows if r["verdict"] == k) for k in ("changed", "unchanged", "refused")}
    here = os.path.abspath(__file__)
    ledger = {"schema": SCHEMA, "level": L, "src_level": src_lvl, "overlay_level": dst_lvl, "dry_run": False,
              "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
              "script": here, "script_md5": md5(here), "common_py_md5": md5(os.path.join(RUNNERS, "common.py")),
              "rule_e": "document (docs_read) budget stop only",
              "level_ok": level_ok, "level_problems": lbad, "level_info": linfo, "n_units": len(rows), "counts": cnt,
              "units": rows}
    print(f"### budget-stop overlay level{L}: {len(rows)} units: changed {cnt['changed']}, unchanged {cnt['unchanged']}, refused {cnt['refused']}"
          + ("" if level_ok else f" | LEVEL REFUSED: {lbad[:3]}"))
    for r in rows:
        if r["verdict"] == "refused":
            print(f"  refused {r['task']} seed{r['seed']} [{r['old_status']}]: {r['reason'][:300]}")
    os.makedirs(root, exist_ok=True)
    if not level_ok:
        json.dump(ledger, open(led_json, "w"), indent=1)
        print(f"FATAL: level{L} refused; no overlay built (ledger {led_json})")
        return 2
    unit_set = set(units)

    def mirror(src, dst):
        os.mkdir(dst)
        for n in sorted(os.listdir(src)):
            sp, dp = os.path.join(src, n), os.path.join(dst, n)
            if src in unit_set and n == "manifests.jsonl":
                with open(dp, "wb") as f:
                    f.write(payload[src])
            elif os.path.isdir(sp) and not os.path.islink(sp) and any(x == sp or x.startswith(sp + "/") for x in unit_set):
                mirror(sp, dp)
            else:
                os.symlink(sp, dp)

    mirror(src_lvl, dst_lvl)
    for r in rows:
        d = os.path.join(dst_lvl, os.path.relpath(r["unit"], src_lvl), "manifests.jsonl")
        r["overlay_manifest_md5"] = md5(d) if os.path.isfile(d) else None
        if r["verdict"] != "changed" and r["overlay_manifest_md5"] != r.get("src_manifest_md5"):
            print(f"FATAL: {r['task']}: an unchanged manifest copy differs from its source")
            return 2
    tmp = led_json + ".tmp"
    json.dump(ledger, open(tmp, "w"), indent=1)
    os.replace(tmp, led_json)
    with open(led_tsv, "w") as f:
        f.write("task\tseed\told_status\tnew_status\tverdict\treason\n")
        for r in rows:
            f.write(f"{r['task']}\t{r['seed']}\t{r['old_status']}\t{r['new_status']}\t{r['verdict']}\t{r['reason']}\n")
    print(f"### overlay {dst_lvl}; ledger {led_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
