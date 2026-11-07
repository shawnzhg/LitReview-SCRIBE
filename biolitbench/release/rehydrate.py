#!/usr/bin/env python3
"""Restores the dehydrated files of a release snapshot into a data root, gating each file and recording drift."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_format as rf

HERE = Path(__file__).resolve().parent
PARSER = HERE / "parse_jats.py"
STATES = ("ok", "drift", "failed", "not_rehydratable")
DRIFTED = ("drift", "missing")
CTX = {}


class Refused(Exception):
    pass


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_abstract_store(path) -> dict:
    store = {}
    if path:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if "abstract" in r:
                        store[str(r["pmid"])] = r["abstract"]
    return store


def abstract_status(digests: dict, store: dict) -> dict:
    out = {}
    for pmid, (sha, n) in digests.items():
        text = store.get(pmid)
        if pmid not in store:
            out[pmid] = "not_fetched"
        elif text is None:
            out[pmid] = "missing"
        elif len(text) == n and rf.sha256_hex(text) == sha:
            out[pmid] = "verified"
        else:
            out[pmid] = "drifted"
    return out


def locate(path: str) -> str:
    for base in (os.path.join(CTX["snapshot"], "store"), CTX["root"]):
        p = os.path.join(base, rf.safe_relpath(path))
        if os.path.exists(p):
            return p
    raise Refused("sealed_file_missing")


def load_wrapper(row) -> dict:
    p = os.path.join(CTX["snapshot"], "dehydrated", rf.safe_relpath(row["path"]))
    if not os.path.exists(p):
        raise Refused("wrapper_missing")
    w = read_json(p)
    if not rf.is_wrapper(w) or w.get("format") != rf.FORMAT:
        raise Refused("wrapper_format")
    if (w.get("kind"), w.get("task_id"), w.get("path"), w.get("target_sha256"), w.get("serialization")) != \
            (row["kind"], row["task_id"], row["path"], row["sha256"], rf.SERIALIZATION[row["kind"]]):
        raise Refused("wrapper_mismatch")
    return w


def write_file(path: str, data: bytes) -> None:
    dst = os.path.join(CTX["root"], rf.safe_relpath(path))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)


def is_digest(ph) -> bool:
    return isinstance(ph, dict) and "sha256" in ph and "len" in ph and "@" not in ph


def check_text(text, ph, reason: str) -> str:
    if text is None or len(text) != ph["len"] or rf.sha256_hex(text) != ph["sha256"]:
        raise Refused(reason)
    return text


def restore_rs(task_id: str, w: dict) -> dict:
    path = os.path.join(CTX["articles"], task_id, "review_structure_llm.json")
    if not os.path.exists(path):
        raise Refused("article_missing")
    art = read_json(path)
    paras = {p["paragraph_id"]: p["text"] for p in art.get("paragraphs") or []}
    secs = {s["section_id"]: s["title"] for s in art.get("sections") or []}
    d = w["data"]
    for p in d["paragraphs"]:
        if is_digest(p["text"]):
            p["text"] = check_text(paras.get(p["paragraph_id"]), p["text"], "paragraph_digest")
        sp = p["sentences"]
        if isinstance(sp, dict):
            if len(sp["lens"]) != len(sp["sha16"]):
                raise Refused("sentence_digest")
            text, out, a = p["text"], [], 0
            for n, h in zip(sp["lens"], sp["sha16"]):
                s = text[a:a + n]
                if len(s) != n or rf.sha16(s) != h:
                    raise Refused("sentence_digest")
                out.append(s)
                a += n + 1
            p["sentences"] = out
    for s in d["sections"]:
        if is_digest(s["title"]):
            s["title"] = check_text(secs.get(s["section_id"]), s["title"], "section_digest")
    return d


def restore_graph(w: dict, rs: dict) -> dict:
    sentences = {p["paragraph_id"]: p["sentences"] for p in rs["paragraphs"]}
    d = w["data"]
    by_claim = {}
    for c in d.get("claims") or []:
        if c.get("sentence") is None:
            try:
                c["sentence"] = sentences[c["paragraph_id"]][c["sentence_index"]]
            except (KeyError, IndexError, TypeError):
                raise Refused("claim_locator")
        s = c["sentence"]
        c["citation_markers"] = [s[m[0]:m[1]] if isinstance(m, list) else m for m in c.get("citation_markers") or []]
        by_claim[c["claim_id"]] = s
    for name in rf.EDGE_LISTS:
        for e in d.get(name) or []:
            for end in ("source", "target"):
                if e.get(end + "_text") is None and end + "_text" in e:
                    if e.get(end + "_id") not in by_claim:
                        raise Refused("edge_endpoint")
                    e[end + "_text"] = by_claim[e[end + "_id"]]
    return d


def sealed_digests(kind: str, task_id: str) -> dict:
    bundle = read_json(locate(rf.contract_path(rf.UNIT_BUNDLE[kind], task_id)))
    out = {}
    for e in bundle.get("evidence") or []:
        out.setdefault(e["locator"], (e["granularity"], e["text_hash"]))
    return out


def sealed(digests: dict, locator: str, granularity=None):
    hit = digests.get(locator)
    if hit is None or (granularity and hit[0] != granularity):
        raise Refused("no_sealed_digest")
    return hit


def pm_source(pmid: str):
    if pmid not in CTX["digests"]:
        raise Refused("pmid_not_in_manifest")
    if CTX["status"][pmid] == "not_fetched":
        raise Refused("abstract_not_fetched")
    return CTX["store"].get(pmid), CTX["status"][pmid], CTX["digests"][pmid][1]


def clamp(x: int, n: int) -> int:
    return min(max(x, 0), n)


def resolve_spans(text, state: str, len_old: int, spans: list) -> dict:
    out = {}
    if state == "verified":
        for key, s, e, h in spans:
            v = text[s:e]
            if rf.text_hash(v) != h:
                raise Refused("unit_digest")
            out[key] = (v, "ok")
        return out
    if text is None:
        return {key: ("", "missing") for key, _, _, _ in spans}
    shifts = []
    for key, s, e, h in spans:
        n = e - s
        hits = [i for i in range(0, len(text) - n + 1) if rf.text_hash(text[i:i + n]) == h]
        if hits:
            i = min(hits, key=lambda i: (abs(i - s), i))
            out[key] = (text[i:i + n], "relocated")
            shifts.append((s, i - s))
    for key, s, e, h in spans:
        if key in out:
            continue
        before = [x for x in shifts if x[0] < s]
        after = [x for x in shifts if x[0] > s]
        lo = max(before, key=lambda x: x[0])[1] if before else 0
        hi = min(after, key=lambda x: x[0])[1] if after else len(text) - len_old
        a, b = clamp(s + lo, len(text)), clamp(e + hi, len(text))
        out[key] = (text[a:max(a, b)], "drift")
    return out


def join_source(pmid: str, ph: dict, oracle):
    if oracle is None:
        raise Refused("oracle_unavailable")
    values, states = oracle
    parts = sorted((rf.oracle_locator(loc)[1], loc) for loc in values if rf.oracle_locator(loc)[0] == pmid)
    if CTX["status"].get(pmid) == "not_fetched":
        raise Refused("abstract_not_fetched")
    if CTX["status"].get(pmid) == "missing":
        return None, "missing"
    text = " ".join(values[loc] for _, loc in parts)
    if len(text) == ph["len"] and rf.sha256_hex(text) == ph["sha256"]:
        return text, "ok"
    if any(states.get(loc) in DRIFTED for _, loc in parts):
        return text, "drift"
    raise Refused("unit_digest")


def unit(task_id, kind, path, pmid, locator, granularity, state, expected, written) -> dict:
    return {"task_id": task_id, "path": path, "kind": kind, "pmid": pmid, "locator": locator,
            "granularity": granularity, "status": state, "expected_text_hash": expected,
            "written_text_hash": None if written is None else rf.text_hash(written)}


def abstract_unit(text, state: str, expected: str):
    if state == "verified" or state == "ok":
        if rf.text_hash(text) != expected:
            raise Refused("unit_digest")
        return text, "ok"
    if state == "drifted":
        return text, "drift"
    return None, "missing"


def restore_oracle(task_id: str, kind: str, path: str, w: dict):
    digests = sealed_digests(kind, task_id)
    d = w["data"]
    units, states, groups = [], {}, {}
    for loc, v in d.items():
        granularity, h = sealed(digests, loc)
        pmid = rf.oracle_locator(loc)[0]
        if isinstance(v, str):
            if rf.text_hash(v) != h:
                raise Refused("unit_digest")
            states[loc] = "ok"
        else:
            groups.setdefault(pmid, []).append((loc, v[0], v[1], h))
    for pmid, spans in groups.items():
        text, state, len_old = pm_source(pmid)
        for loc, (v, st) in resolve_spans(text, state, len_old, spans).items():
            d[loc] = v
            states[loc] = st
            units.append(unit(task_id, kind, path, pmid, loc, digests[loc][0], st, digests[loc][1], v))
    return d, states, units


def restore_texts(task_id: str, kind: str, path: str, w: dict, oracle):
    digests = sealed_digests(kind, task_id)
    d = w["data"]
    units = []
    for pmid, entry in d.items():
        ab = entry.get("abstract")
        loc_a = f"pmid{pmid}"
        text, span_state, len_old = None, None, 0
        if ab is not None:
            _, expected = sealed(digests, loc_a, "abstract")
            if isinstance(ab, dict) and ab.get("@") == rf.PM:
                text, pm_state, len_old = pm_source(pmid)
                text, st = abstract_unit(text, pm_state, expected)
                span_state = pm_state
            elif isinstance(ab, dict) and ab.get("@") == rf.ORACLE_JOIN:
                text, st = join_source(pmid, ab, oracle)
                text, st = abstract_unit(text, {"ok": "ok", "drift": "drifted"}.get(st, st), expected)
                span_state, len_old = {"ok": "verified", "drift": "drifted"}.get(st, st), ab["len"]
            elif isinstance(ab, str):
                text, st = abstract_unit(ab, "ok", expected)
                span_state = "verified"
            else:
                raise Refused("placeholder")
            entry["abstract"] = text
            units.append(unit(task_id, kind, path, pmid, loc_a, "abstract", st, expected, text))
        title = entry.get("title") or ""
        sents = entry.get("sentences") or {}
        spans = []
        for loc, v in sents.items():
            _, h = sealed(digests, loc, "abstract_sentence")
            if isinstance(v, str):
                if rf.text_hash(v) != h:
                    raise Refused("unit_digest")
            elif v[0] == rf.TITLE_SPAN:
                s = title[v[1]:v[2]]
                if rf.text_hash(s) != h:
                    raise Refused("unit_digest")
                sents[loc] = s
            elif v[0] == rf.ABSTRACT_SPAN:
                if ab is None:
                    raise Refused("placeholder")
                spans.append((loc, v[1], v[2], h))
            else:
                raise Refused("placeholder")
        if spans:
            for loc, (v, st) in resolve_spans(text, span_state, len_old, spans).items():
                sents[loc] = v
                units.append(unit(task_id, kind, path, pmid, loc, "abstract_sentence", st, digests[loc][1], v))
    return d, units


def restore_baseline(task_id: str, kind: str, path: str, w: dict, oracle):
    d = w["data"]
    units = []
    for r in d.get("refs") or []:
        ab = r.get("abstract")
        if ab is None or isinstance(ab, str):
            continue
        pmid = str(r.get("pmid"))
        expected = "sha256:" + ab["sha256"]
        if ab.get("@") == rf.PM:
            text, state, _ = pm_source(pmid)
            text, st = abstract_unit(text, state, expected)
        elif ab.get("@") == rf.ORACLE_JOIN:
            text, st = join_source(pmid, ab, oracle)
            text, st = abstract_unit(text, {"ok": "ok", "drift": "drifted"}.get(st, st), expected)
        else:
            raise Refused("placeholder")
        r["abstract"] = text
        units.append(unit(task_id, kind, path, pmid, f"pmid{pmid}", "baseline_abstract", st, expected, text))
    return d, units


def gate(kind: str, obj, target: str, units=()):
    data = rf.dumps(obj, rf.SERIALIZATION[kind])
    sha = rf.sha256_bytes(data)
    if sha == target:
        return data, sha, "ok", []
    drift = [u for u in units if u["status"] in DRIFTED]
    if drift:
        return data, sha, "drift", drift
    raise Refused("file_digest")


def init_worker(ctx: dict) -> None:
    CTX.update(ctx)


def process_task(job):
    task_id, rows = job
    records, drift, counts = [], [], Counter()
    rs = oracle = None
    for row in sorted(rows, key=lambda r: rf.RESTORE_ORDER.index(r["kind"])):
        kind, path = row["kind"], row["path"]
        rec = {"path": path, "kind": kind, "task_id": task_id, "status": "failed", "target_sha256": row["sha256"],
               "written_sha256": None, "reason": None}
        records.append(rec)
        if kind in rf.REVIEW_KINDS and task_id in CTX["nr"]:
            rec["status"] = "not_rehydratable"
            continue
        try:
            w = load_wrapper(row)
            units = []
            if kind == "rs":
                if not (w.get("source") or {}).get("rehydratable", True):
                    rec["status"] = "not_rehydratable"
                    continue
                obj = restore_rs(task_id, w)
            elif kind == "graph":
                if rs is None:
                    raise Refused("rs_unavailable")
                obj = restore_graph(w, rs)
            elif kind == "oracle_sentences":
                obj, states, units = restore_oracle(task_id, kind, path, w)
            elif kind in ("texts_test", "texts_train"):
                obj, units = restore_texts(task_id, kind, path, w, oracle)
            elif kind == "baseline_refs":
                obj, units = restore_baseline(task_id, kind, path, w, oracle)
            else:
                raise Refused("kind")
            data, sha, status, file_drift = gate(kind, obj, row["sha256"], units)
            write_file(path, data)
            rec.update(status=status, written_sha256=sha)
            for u in units:
                if u["status"] in ("relocated",) + DRIFTED:
                    counts[u["status"]] += 1
            drift.extend(file_drift)
            if kind == "rs":
                rs = obj
            elif kind == "oracle_sentences":
                oracle = (obj, states)
        except Refused as e:
            rec["reason"] = str(e.args[0])
        except Exception as e:
            rec["reason"] = "error:" + type(e).__name__
    return records, drift, counts


def repo_commit():
    try:
        out = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def manifest_sha256() -> str:
    for base in (CTX["root"], CTX["snapshot"]):
        p = os.path.join(base, "manifests", "original_sha256.tsv")
        if os.path.exists(p):
            return rf.file_sha256(p)
    return None


def build_report(profile: str, records: list, drift: list, counts: Counter, kinds: list) -> dict:
    by_kind = {k: {s: 0 for s in STATES} for k in kinds}
    for r in records:
        by_kind.setdefault(r["kind"], {s: 0 for s in STATES})[r["status"]] += 1
    st = Counter(CTX["status"].values())
    return {"schema": rf.REPORT_SCHEMA, "dataset": rf.DATASET, "version": rf.VERSION, "profile": profile,
            "root": os.path.abspath(CTX["root"]), "manifest_sha256": manifest_sha256(),
            "repo_commit": repo_commit(), "parser_sha256": rf.file_sha256(PARSER),
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "abstracts": {"expected": len(CTX["status"]), "verified": st["verified"], "drifted": st["drifted"],
                          "missing": st["missing"], "not_fetched": st["not_fetched"]},
            "files_by_kind": by_kind,
            "units": {"relocated": counts["relocated"], "drift": counts["drift"], "missing": counts["missing"]},
            "tasks_with_drift": len({u["task_id"] for u in drift}),
            "files": sorted((r for r in records if r["status"] != "ok"), key=lambda r: r["path"]),
            "drift": sorted(drift, key=lambda u: (u["task_id"], u["path"], u["locator"]))}


def write_report(report: dict) -> None:
    os.makedirs(CTX["root"], exist_ok=True)
    dst = os.path.join(CTX["root"], "rehydrate_report.json")
    tmp = f"{dst}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=1, ensure_ascii=False))
    os.replace(tmp, dst)


def plan(snapshot: str, profile: str, tasks) -> dict:
    jobs = {}
    for r in rf.read_originals(snapshot):
        if r.get("origin") != "rehydrate" or r["kind"] not in rf.RESTORE_ORDER:
            continue
        if profile not in rf.parse_profiles(r.get("profiles")) or (tasks is not None and r["task_id"] not in tasks):
            continue
        jobs.setdefault(r["task_id"], []).append(r)
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--profile", choices=rf.PROFILES, required=True)
    ap.add_argument("--articles", required=True)
    ap.add_argument("--abstracts")
    ap.add_argument("--tasks")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    if rf.read_abstract_digests(a.snapshot, a.profile) and not (a.abstracts and os.path.isfile(a.abstracts)):
        ap.error("--abstracts must name the abstracts.jsonl written by fetch_abstracts.py for this profile")
    CTX.update(snapshot=a.snapshot, root=a.root, articles=a.articles, status={}, digests={}, store={}, nr=set())
    records, drift, counts, kinds = [], [], Counter(), []
    try:
        tasks = {line.strip() for line in open(a.tasks) if line.strip()} if a.tasks else None
        jobs = plan(a.snapshot, a.profile, tasks)
        kinds = [k for k in rf.RESTORE_ORDER if any(r["kind"] == k for rows in jobs.values() for r in rows)]
        CTX["nr"] = {r["task_id"] for r in rf.read_article_sources(a.snapshot)
                     if not rf.flag(r.get("rehydratable", "true"))}
        CTX["digests"] = rf.read_abstract_digests(a.snapshot, a.profile)
        CTX["store"] = load_abstract_store(a.abstracts)
        CTX["status"] = abstract_status(CTX["digests"], CTX["store"])
        items = sorted(jobs.items())
        if a.workers > 1:
            with Pool(a.workers, initializer=init_worker, initargs=(dict(CTX),)) as pool:
                results = pool.imap_unordered(process_task, items, chunksize=4)
                for rec, dr, c in results:
                    records.extend(rec)
                    drift.extend(dr)
                    counts.update(c)
        else:
            for item in items:
                rec, dr, c = process_task(item)
                records.extend(rec)
                drift.extend(dr)
                counts.update(c)
    finally:
        report = build_report(a.profile, records, drift, counts, kinds)
        write_report(report)
        print(json.dumps({"files_by_kind": report["files_by_kind"], "abstracts": report["abstracts"],
                          "units": report["units"], "tasks_with_drift": report["tasks_with_drift"]}))
    return 1 if any(r["status"] == "failed" for r in records) or report["abstracts"]["not_fetched"] else 0


if __name__ == "__main__":
    sys.exit(main())
