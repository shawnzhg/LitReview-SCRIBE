#!/usr/bin/env python3
"""Builds the fixed-input evidence bundle of every task from its reference list, keeping only papers
published before the cutoff and never the evaluated review, or checks the built bundles. Usage:
python build_canonical_bundles.py [--split dev|train] [--tasks <ids>] [--check]."""

from __future__ import annotations
import argparse
import csv
import functools
import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scribe" / "harness" / "runners"))
from common import RUNS, seal, sha256_str, validate, content_hash
import runner as R


def _site(var):
    return Path(os.environ.get(var) or f"/nonexistent/{var}")


SQLITE = _site("SCRIBE_POOL_META_SQLITE")
REF_YEARS = _site("SCRIBE_REF_YEARS")
REFS = _site("SCRIBE_DATASET_ROOT") / "refs_enriched"
EVAL_TASKS = _site("SCRIBE_EVAL_TASKS")
BUILDER_VERSION = "canonical_allowlist/1.0"

SPLIT_DEFAULTS = {
    "dev": {
        "allowlist_dir": _site("SCRIBE_ALLOWLISTS"),
        "taskspecs_dir": RUNS / "taskspecs" / "dev",
        "gold_dir": RUNS / "gold" / "dev",
        "refs_dir": REFS,
        "texts_dir": None,
        "sentences_dir": RUNS / "oracle" / "dev" / "sentences",
        "split_ref_years": None,
        "out": RUNS / "canonical" / "campaign50_ref",
    },
    "train": {
        "allowlist_dir": RUNS / "canonical" / "train" / "allowlists",
        "taskspecs_dir": RUNS / "taskspecs" / "train",
        "gold_dir": RUNS / "gold" / "train",
        "refs_dir": REFS,
        "texts_dir": RUNS / "canonical" / "train" / "texts",
        "sentences_dir": None,
        "split_ref_years": RUNS / "canonical" / "train" / "ref_years.json",
        "out": RUNS / "canonical" / "train" / "bundles",
    },
}


class SplitCtx:

    FIELDS = ("allowlist_dir", "taskspecs_dir", "gold_dir", "refs_dir", "texts_dir",
              "sentences_dir", "split_ref_years", "out")

    def __init__(self, split="dev", **over):
        if split not in SPLIT_DEFAULTS:
            raise ValueError(f"unknown split {split!r}")
        unknown = sorted(set(over) - set(self.FIELDS))
        if unknown:
            raise TypeError(f"SplitCtx: unknown override(s) {unknown}; valid: {list(self.FIELDS)}")
        self.split = split
        d = dict(SPLIT_DEFAULTS[split])
        for k, v in over.items():
            if v is not None:
                d[k] = Path(v)
        for k in self.FIELDS:
            setattr(self, k, d[k])

    def allowlist_path(self, task_id) -> Path:
        return self.allowlist_dir / f"{task_id}.json"

    def load_allowlist(self, task_id):
        p = self.allowlist_path(task_id)
        return None if not p.exists() else [str(x) for x in json.loads(p.read_text())]

    def load_spec(self, task_id):
        return json.loads((self.taskspecs_dir / f"{task_id}.json").read_text())

    def load_gold(self, task_id):
        p = self.gold_dir / f"{task_id}.json" if self.gold_dir else None
        return json.loads(p.read_text()) if p is not None and p.exists() else None

    def review_pmid(self, task_id):
        rp = (self.load_gold(task_id) or {}).get("review_pmid")
        if rp in (None, ""):
            raise FileNotFoundError(f"{task_id}: no review_pmid in the gold record under {self.gold_dir}; "
                                    f"the evaluated review cannot be excluded")
        return str(rp)

    def tasks(self):
        if self.split == "dev":
            return campaign50_tasks()
        return sorted(p.stem for p in self.taskspecs_dir.glob("*.json"))


DEV = None


def dev_ctx():
    global DEV
    if DEV is None:
        DEV = SplitCtx("dev")
    return DEV


def campaign50_tasks():
    return sorted(json.loads(EVAL_TASKS.read_text())["tasks"])


@functools.lru_cache(maxsize=1)
def ref_years():
    if not REF_YEARS.exists():
        raise FileNotFoundError(f"ref_years manifest missing at {REF_YEARS}")
    return {str(k): int(v) for k, v in json.loads(REF_YEARS.read_text()).items() if v is not None}


def _sqlite_rows(db, pmids):
    out = {}
    if db is None:
        return out
    for p in pmids:
        row = db.execute("select title, year, abstract from meta where pmid=?", (int(p),)).fetchone()
        if row:
            out[p] = {"title": row[0], "year": row[1], "abstract": row[2]}
    return out


@functools.lru_cache(maxsize=4)
def split_ref_years(path: Path):
    if path is None:
        return {}
    if not Path(path).exists():
        raise FileNotFoundError(f"split ref_years missing: {path} -- build this split's "
                                f"ref_years.json first")
    return {str(k): int(v) for k, v in json.loads(Path(path).read_text()).items() if v is not None}


def _refs_by_pmid(task_id, refs_dir=None):
    p = (refs_dir or REFS) / f"{task_id}.json"
    if not p.exists():
        return {}
    out = {}
    for _, v in json.loads(p.read_text()).items():
        pid = v.get("pmid")
        if pid:
            out.setdefault(str(pid), v)
    return out


_UNSET = object()


def _sentences_by_pmid(task_id, sentences_dir=_UNSET):
    if sentences_dir is _UNSET:
        sentences_dir = R.ORACLE / "sentences"
    out = {}
    if sentences_dir is None:
        return out
    p = Path(sentences_dir) / f"{task_id}.json"
    if not p.exists():
        return out
    for loc, txt in json.loads(p.read_text()).items():
        pid = loc.split("_")[0][4:]
        out.setdefault(pid, []).append((int(loc.rsplit("_", 1)[1]), loc, txt))
    for pid in out:
        out[pid].sort()
    return out


def _texts_source(task_id, texts_dir):
    if not texts_dir:
        return {}
    p = Path(texts_dir) / f"{task_id}.json"
    if not p.exists():
        return {}
    return {str(k): v for k, v in json.loads(p.read_text()).items()}


def _nonblank(s):
    return s if isinstance(s, str) and s.strip() else None


def build_task(task_id, db=None, ctx=None):
    ctx = ctx or dev_ctx()
    allow_file = ctx.load_allowlist(task_id)
    if allow_file is None:
        raise FileNotFoundError(f"allowlist missing: {ctx.allowlist_path(task_id)}")
    if len(set(allow_file)) != len(allow_file):
        raise ValueError(f"{task_id}: allowlist has duplicate pmids")
    if not all(p.isdigit() for p in allow_file):
        raise ValueError(f"{task_id}: non-numeric pmid in allowlist")
    review_pmid = ctx.review_pmid(task_id)
    excluded_review_pmid = review_pmid if review_pmid in set(allow_file) else None
    pmids = sorted((p for p in allow_file if p != excluded_review_pmid), key=int)
    spec = ctx.load_spec(task_id)
    cutoff = int(spec["publication_cutoff"])
    sq = _sqlite_rows(db, pmids)
    ry = ref_years()
    sry = split_ref_years(ctx.split_ref_years) if ctx.split_ref_years else {}
    refs = _refs_by_pmid(task_id, ctx.refs_dir)
    sents = _sentences_by_pmid(task_id, ctx.sentences_dir)
    tx_src = _texts_source(task_id, ctx.texts_dir)
    if ctx.texts_dir and not tx_src:
        raise FileNotFoundError(f"{task_id}: texts source missing/empty at "
                                f"{Path(ctx.texts_dir) / (task_id + '.json')} -- build the "
                                f"{ctx.split} split's texts first")

    year_of, late, post_excluded = {}, [], []
    year_sources = {"ref_years": 0, "split_ref_years": 0, "sqlite": 0}
    for pid in pmids:
        if ry.get(pid) is not None:
            y, src = int(ry[pid]), "ref_years"
        elif sry.get(pid) is not None:
            y, src = int(sry[pid]), "split_ref_years"
        else:
            sy = (sq.get(pid) or {}).get("year")
            y, src = (int(sy) if sy is not None else None), "sqlite"
        if y is not None and y >= cutoff and src != "sqlite":
            late.append(pid)
        elif y is None or y >= cutoff:
            post_excluded.append(pid)
        else:
            year_of[pid] = y
            year_sources[src] += 1
    if late:
        raise ValueError(f"{task_id}: {len(late)} allowlisted papers dated in or after the cutoff year "
                         f"{cutoff}: {late[:5]} -- the allowlist is not cutoff-clean")
    pmids = [p for p in pmids if p in year_of]

    papers, evidence, texts, without = [], [], {}, []
    for rank, pid in enumerate(pmids, 1):
        s, r, x = sq.get(pid) or {}, refs.get(pid) or {}, tx_src.get(pid) or {}
        title = (_nonblank(x.get("title")) if x else None) or \
            _nonblank(s.get("title")) or _nonblank(r.get("title"))
        abstract = (_nonblank(x.get("abstract")) if x else None) or \
            _nonblank(s.get("abstract")) or _nonblank(r.get("abstract"))
        if abstract is None and pid in sents:
            abstract = " ".join(t for _, _, t in sents[pid])
        papers.append({"paper_id": pid, "doi": _nonblank(r.get("doi")), "title": title,
                       "year": year_of[pid], "rank": rank, "score": None, "first_seen_step": None,
                       "retrieval_provenance": {"query": "", "tool": "canonical_allowlist",
                                                "rank_from_tool": None, "route": "cited_by_human"},
                       "decision": "include", "decision_reason": f"{ctx.split} canonical allowlist",
                       "post_cutoff": False})
        sd = {loc: txt for _, loc, txt in sents.get(pid, [])}
        texts[pid] = {"title": title, "abstract": abstract, "sentences": sd}
        if abstract is not None:
            evidence.append({"evidence_id": f"e_{pid}", "paper_id": pid, "granularity": "abstract",
                             "locator": f"pmid{pid}", "text_hash": sha256_str(abstract),
                             "condition": None})
        else:
            without.append(pid)
        for k, loc, txt in sents.get(pid, []):
            evidence.append({"evidence_id": f"e_{pid}_{k}", "paper_id": pid,
                             "granularity": "abstract_sentence", "locator": loc,
                             "text_hash": sha256_str(txt), "condition": None})

    al_sha = R.allowlist_sha256(pmids)
    validation = {
        "builder": BUILDER_VERSION,
        "split": ctx.split,
        "allowlist_path": str(ctx.allowlist_path(task_id)),
        "allowlist_sha256": al_sha,
        "allowlist_sha256_file": R.allowlist_sha256(allow_file),
        "n_allowlisted_file": len(allow_file),
        "n_allowlisted": len(pmids),
        "n_with_abstract": len(pmids) - len(without),
        "n_without_abstract": len(without),
        "without_abstract_pmids": without,
        "cutoff": cutoff,
        "year_sources": year_sources,
        "ref_years_path": str(REF_YEARS),
        "split_ref_years_path": str(ctx.split_ref_years) if ctx.split_ref_years else None,
        "texts_source_path": str(ctx.texts_dir) if ctx.texts_dir else None,
        "review_pmid": review_pmid,
        "review_pmid_excluded": excluded_review_pmid,
        "n_review_pmid_excluded": 1 if excluded_review_pmid else 0,
        "post_cutoff_excluded_pmids": post_excluded,
        "n_post_cutoff_excluded": len(post_excluded),
    }
    bundle = seal({
        "schema_version": "evidence_bundle/1.0", "task_id": task_id,
        "task_spec_hash": spec["content_hash"], "provenance_tier": "reference_derived",
        "retrieval_status": "done", "papers": papers, "evidence": evidence,
        "discovered_union": [], "failures": [], "budget_remaining": {},
        "validation": validation})
    row = {"task_id": task_id, "split": ctx.split, "cutoff": cutoff, "n_allowlisted": len(pmids),
           "n_with_abstract": len(pmids) - len(without), "n_without_abstract": len(without),
           "n_year_ref_years": year_sources["ref_years"], "n_year_split_ref_years": year_sources["split_ref_years"],
           "n_year_sqlite": year_sources["sqlite"], "n_evidence": len(evidence),
           "n_sentence_evidence": sum(e["granularity"] == "abstract_sentence" for e in evidence),
           "review_pmid": review_pmid, "n_review_pmid_excluded": validation["n_review_pmid_excluded"],
           "n_post_cutoff_excluded": len(post_excluded),
           "content_hash": bundle["content_hash"], "texts_sha256": R.texts_sha256(texts),
           "allowlist_sha256": al_sha}
    return bundle, texts, row


def _dump(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False)


def _write_if_changed(p: Path, text: str) -> bool:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.read_text() == text:
        return False
    p.write_text(text)
    return True


SUMMARY_COLS = ["task_id", "split", "cutoff", "n_allowlisted", "n_with_abstract", "n_without_abstract",
                "n_year_ref_years", "n_year_split_ref_years", "n_year_sqlite", "n_evidence",
                "n_sentence_evidence", "review_pmid", "n_review_pmid_excluded", "n_post_cutoff_excluded",
                "content_hash", "texts_sha256", "allowlist_sha256"]


def _read_index(out):
    p = out / "index.jsonl"
    rows = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["task_id"]] = r
    return rows


def _write_index(out, rows):
    rows = dict(sorted(rows.items()))
    (out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
    with (out / "_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows.values():
            w.writerow(r)


def build(tasks, out: Path, db=None, ctx=None):
    ctx = ctx or dev_ctx()
    out.mkdir(parents=True, exist_ok=True)
    index = _read_index(out)
    n_changed, errors, n_excluded, n_post = 0, [], 0, 0
    for t in tasks:
        bundle, texts, row = build_task(t, db, ctx)
        n_excluded += row["n_review_pmid_excluded"]
        n_post += row["n_post_cutoff_excluded"]
        errs = validate("evidence_bundle", bundle)
        if errs:
            errors.append((t, errs))
            print(f"[canonical] {t}: SCHEMA INVALID: {errs[:3]}", file=sys.stderr)
            continue
        bp, tp = out / "evidence_bundle" / f"{t}.json", out / "texts" / f"{t}.json"
        ch = _write_if_changed(bp, _dump(bundle)) | _write_if_changed(tp, _dump(texts))
        n_changed += ch
        row.update(path=str(bp), texts_path=str(tp))
        index[t] = row
        print(f"[canonical] {t}: {row['n_allowlisted']} papers, {row['n_with_abstract']} with "
              f"abstract, {row['n_evidence']} evidence, {bundle['content_hash'][:23]}... "
              f"{'written' if ch else 'unchanged'}")
    _write_index(out, index)
    print(f"[canonical] split={ctx.split} {len(tasks)} tasks, {n_changed} files changed, "
          f"{len(errors)} schema failures, {n_excluded} review_pmid excluded, "
          f"{n_post} not dated before the cutoff excluded -> {out}")
    return errors


def check(tasks, out: Path, db=None, ctx=None):
    ctx = ctx or dev_ctx()
    index = _read_index(out)
    problems = []
    for t in tasks:
        bp, tp = out / "evidence_bundle" / f"{t}.json", out / "texts" / f"{t}.json"
        if not bp.exists() or not tp.exists():
            problems.append(f"{t}: missing {bp if not bp.exists() else tp}")
            continue
        disk = json.loads(bp.read_text())
        texts = json.loads(tp.read_text())
        errs = validate("evidence_bundle", disk)
        if errs:
            problems.append(f"{t}: schema errors {errs[:3]}")
        if content_hash(disk) != disk.get("content_hash"):
            problems.append(f"{t}: on-disk bundle does not re-hash")
        bundle, texts_mem, row = build_task(t, db, ctx)
        if bundle["content_hash"] != disk.get("content_hash"):
            problems.append(f"{t}: DRIFT rebuild {bundle['content_hash'][:23]} != disk "
                            f"{str(disk.get('content_hash'))[:23]}")
        if R.texts_sha256(texts) != row["texts_sha256"] or texts != texts_mem:
            problems.append(f"{t}: texts file differs from rebuild")
        ir = index.get(t)
        if not ir:
            problems.append(f"{t}: not in index.jsonl")
        elif ir.get("content_hash") != disk.get("content_hash") or \
                ir.get("texts_sha256") != R.texts_sha256(texts):
            problems.append(f"{t}: index.jsonl hashes disagree with the files")
        derrs, _ = R.check_bundle_delivery(t, ctx.load_spec(t), disk, texts, canonical=disk,
                                           allowlist=[p["paper_id"] for p in bundle["papers"]])
        if derrs:
            problems.append(f"{t}: delivery assertion fails on the canonical files: {derrs[:2]}")
    for pr in problems:
        print(f"[canonical --check] {pr}", file=sys.stderr)
    print(f"[canonical --check] {len(tasks)} tasks, {len(problems)} problems")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=sorted(SPLIT_DEFAULTS),
                    help="which split's inputs/outputs to use (default dev, the evaluation tasks)")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: the split's)")
    ap.add_argument("--allowlist-dir", default=None)
    ap.add_argument("--taskspecs-dir", default=None)
    ap.add_argument("--gold-dir", default=None, help="gold records naming each task's review_pmid")
    ap.add_argument("--refs-enriched-dir", default=None)
    ap.add_argument("--texts-dir", default=None,
                    help="title/abstract source for the train split (canonical/train/texts)")
    ap.add_argument("--sentences-dir", default=None, help="abstract sentences (dev only)")
    ap.add_argument("--split-ref-years", default=None,
                    help="canonical/<split>/ref_years.json, consulted after SCRIBE_REF_YEARS")
    ap.add_argument("--out", default=None, help=f"output root (default {R.CANON} for dev)")
    ap.add_argument("--check", action="store_true", help="verify, write nothing; non-zero on drift")
    a = ap.parse_args(argv)
    ctx = SplitCtx(a.split, allowlist_dir=a.allowlist_dir, taskspecs_dir=a.taskspecs_dir,
                   gold_dir=a.gold_dir, refs_dir=a.refs_enriched_dir, texts_dir=a.texts_dir,
                   sentences_dir=a.sentences_dir, split_ref_years=a.split_ref_years, out=a.out)
    tasks = a.tasks.split(",") if a.tasks else ctx.tasks()
    out = ctx.out
    db = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True) if SQLITE.exists() else None
    if db is None:
        print(f"[canonical] WARNING sqlite missing at {SQLITE}; refs_enriched/sentences only",
              file=sys.stderr)
    if a.check:
        sys.exit(1 if check(tasks, out, db, ctx) else 0)
    sys.exit(1 if build(tasks, out, db, ctx) else 0)


if __name__ == "__main__":
    main()
