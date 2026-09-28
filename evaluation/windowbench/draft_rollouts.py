"""Builds the draft-exit rollouts of the published pipelines from their window extractors and merges
their conformance rows. Usage: python -m windowbench.draft_rollouts build|conformance [--arms ...]."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import re
import time
from pathlib import Path

from . import config as C
from .window_extractors.surveyforge import final_reference_maps

NATIVE = ("autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr", "lira")
REF = ("autosurvey", "surveyg", "llmxmr", "sgi")


def task_dir(arm: str, task: str, dataset: str) -> Path:
    if dataset == "ref":
        return C.RUN_ROOT_FIXED_INPUT / arm / task
    return C.RUNS / "campaign50" / arm / task


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", t.lower()).strip()


def _title_resolver(title2pmid: dict[str, str]):
    from rapidfuzz import fuzz, process
    keys = list(title2pmid)
    cache: dict = {}

    def one(t: str):
        n = _norm(t)
        if n in title2pmid:
            return title2pmid[n]
        if not keys:
            return None
        best = process.extractOne(n, keys, scorer=fuzz.token_set_ratio, score_cutoff=90)
        return title2pmid[best[0]] if best else None

    def resolve(payload: str):
        if re.match(r"^[\d,\s;–-]+$", payload):
            return None
        if payload in cache:
            return cache[payload]
        out, ok = [], False
        for part in re.split(r"\s*[;|]\s*", payload):
            if not part.strip():
                continue
            pm = one(part.strip())
            if pm:
                ok = True
                out.append(pm)
        cache[payload] = out if ok else None
        return cache[payload]
    return resolve


def _map_from_zip(titles_by_sec: dict, pmids_by_sec: dict) -> dict[str, str]:
    m: dict[str, str] = {}
    for k, titles in (titles_by_sec or {}).items():
        pm = (pmids_by_sec or {}).get(k) or []
        if titles and pm and len(titles) == len(pm):
            for t, p in zip(titles, pm):
                if t and re.match(r"^\d{4,9}$", str(p)):
                    m.setdefault(_norm(str(t)), str(p))
    return m


def cite_and_resolver(arm: str, td: Path, ex: dict):
    if arm == "autosurvey":
        refs = final_reference_maps(td)
        m = {_norm(t): str(p).split(".")[-1] for t, p in refs["title_to_pmid"].items()}
        m.update({k: v for k, v in _map_from_zip(ex.get("section_evidence"), ex.get("section_evidence_pmids")).items() if k not in m})
        return re.compile(r"\[([^\[\]\n]{6,400})\]"), _title_resolver(m), f"title map {len(m)} (References list + prompt/tap zip)"
    if arm == "surveyforge":
        refs = final_reference_maps(td)
        m = {_norm(t): str(p).split(".")[-1] for t, p in refs["title_to_pmid"].items()}
        m.update({k: v for k, v in _map_from_zip(ex.get("section_evidence_intended"), ex.get("section_evidence")).items() if k not in m})
        return re.compile(r"\[([^\[\]\n]{6,400})\]"), _title_resolver(m), f"title map {len(m)} (References list + intended/resolved zip)"
    if arm == "surveyg":
        key2pmid: dict[str, list[str]] = {}
        dp = td / "literature_review_data.json"
        if dp.exists():
            for fname, key in (json.load(open(dp)).get("citations_map") or {}).items():
                mm = re.search(r"PMID[:_]?(\d+)", fname)
                if mm:
                    key2pmid[key] = [mm.group(1)]

        def r_surveyg(payload: str):
            out, ok = [], False
            for tok in re.split(r"[,;]", payload):
                tok = tok.strip()
                if tok in key2pmid:
                    ok = True
                    out.extend(key2pmid[tok])
            return out if ok else None
        return re.compile(r"\[((?:paper[0-9A-Za-z_]+\s*[,;]?\s*)+)\]"), r_surveyg, f"citations_map {len(key2pmid)} keys"
    if arm == "sgi":
        return re.compile(r"\[((?:pmid\d+\s*[;,]?\s*)+)\]"), (lambda p: re.findall(r"pmid(\d+)", p) or None), "pmid marks"
    if arm == "llmxmr":
        bk = {p["bibkey"]: str(p["pmid"]) for p in (ex.get("papers") or []) if p.get("pmid") and p.get("bibkey")}

        def r_llmxmr(payload: str):
            keys = re.findall(r"'([^']+)'", payload)
            out = [bk[k] for k in keys if k in bk]
            return out or None
        return re.compile(r"\[((?:'[^'\]]+'\s*,?\s*)+)\]"), r_llmxmr, f"bibkey map {len(bk)}"
    if arm == "lira":
        m: dict[str, str] = {}
        inp = td / "data" / "scireviewgen" / "full_data_abs.json"
        if inp.exists():
            d = json.load(open(inp))
            for rec in (d if isinstance(d, list) else [d]):
                for r in rec.get("references", []):
                    if r.get("id") is not None and r.get("title"):
                        m[_norm(r["title"])] = str(r["id"])
        return re.compile(r"\[([^\[\]\n]{6,400})\]"), _title_resolver(m), f"given bibliography {len(m)} titles"
    raise KeyError(arm)


def _first_refine_seq(arm: str, ex: dict) -> int | None:
    seqs = []
    for p in ex.get("refinement_passes") or []:
        role = str(p.get("role") or "")
        if not (role.startswith("refine") or role in ("select_best", "review", "resolve_conflict", "style_rewrite")):
            continue
        rng = p.get("seq_range")
        if isinstance(rng, str) and "-" in rng:
            seqs.append(int(rng.split("-")[0]))
        elif isinstance(rng, (list, tuple)) and rng:
            seqs.append(int(rng[0]))
    if arm == "autosurvey":
        ds = [p.get("draft_seq") for p in (ex.get("per_subsection") or []) if p.get("draft_seq") is not None]
        if ds:
            return max(ds) + 1
    if arm == "sgi" and ex.get("first_write_seq") is not None and seqs:
        return min(seqs)
    return min(seqs) if seqs else None


def build_one(arm: str, task: str, dataset: str, mod, common, adapters, build_mod, redo: bool = False):
    key = arm + (".ref" if dataset == "ref" else "")
    td = task_dir(arm, task, dataset)
    t0 = time.time()
    if not td.is_dir():
        return None, {"task": task, "key": key, "error": "no_task_dir"}
    dest0 = build_mod.rollout_path(key + ".draft", "campaign", task)
    if dest0.exists() and not redo:
        from ccbench.model import Rollout
        ro = Rollout.from_json(dest0)
        row = build_mod._row(ro, 0.0)
        row.update({"draft_is_final": ro.meta.get("draft_is_final"), "n_llm_to_draft": ro.meta.get("n_llm_to_draft"),
                    "final_words": ro.meta.get("final_words"), "cached": True})
        return ro, row
    final = adapters[arm](task, td)
    try:
        ex = mod.extract(td)
    except Exception as e:
        ro = common.bot(arm, task, td, f"draft_unavailable:{type(e).__name__}:{str(e)[:80]}")
        ex = None
    if ex is not None:
        draft = ex.get("draft_report") or ""
        if not draft.strip() or "[no draft]" in draft[:200]:
            ro = common.bot(arm, task, td, "draft_unavailable:empty")
        else:
            cite_re, resolver, note = cite_and_resolver(arm, td, ex)
            drop = ("references", "bibliography", "papers included") if arm == "surveyg" else ("references", "bibliography")
            report = common.report_from_markdown(draft, cite_re, resolver, drop_after_header=drop)
            outline = final.outline if (arm in ("surveyg", "lira", "llmxmr") and final.outline) else common.outline_from_markdown(draft)
            ro = common.assemble(arm, task, td, papers=list(final.papers), outline=outline, report=report, graph=None,
                                 final_ok=not final.is_bot, bot_reason=final.bot_reason,
                                 extra_meta={"draft_source": "extractor.draft_report", "citation_map": note,
                                             "draft_is_final": False})
            fr = _first_refine_seq(arm, ex)
            n_llm_to_draft = sum(1 for e in ro.events if e.kind == "llm" and (fr is None or e.seq < fr))
            ro.meta.update({"first_refine_seq": fr, "n_llm_to_draft": n_llm_to_draft,
                            "final_words": final.report.words if final.report else None,
                            "final_marks": final.report.total_citation_marks if final.report else None,
                            "final_resolution": (1 - final.report.unresolved_citations / final.report.total_citation_marks) if final.report and final.report.total_citation_marks else None})
            if final.is_bot:
                ro.status, ro.bot_reason = final.status, final.bot_reason
    ro.system = key + ".draft"
    ro.mode = "campaign"
    if dataset == "ref":
        ro.meta["campaign"] = "campaign50_ref"
        ro.meta["provenance_tier"] = "reference_derived"
    dest = build_mod.rollout_path(ro.system, "campaign", task)
    ro.to_json(dest)
    row = build_mod._row(ro, time.time() - t0)
    row.update({"draft_is_final": ro.meta.get("draft_is_final"), "n_llm_to_draft": ro.meta.get("n_llm_to_draft"),
                "final_words": ro.meta.get("final_words")})
    return ro, row


def build(arms: list[str], tasks: list[str], datasets: list[str], out: Path, redo: bool = False) -> None:
    C.bootstrap_env()
    from ccbench import build as build_mod
    from ccbench.adapters import BASELINE_ADAPTERS, common
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"build_log_{'+'.join(arms)}_{'+'.join(datasets)}.jsonl"
    rows: list[dict] = []
    with open(log, "a") as f:
        for ds in datasets:
            for arm in arms:
                if ds == "ref" and arm not in REF:
                    continue
                mod = importlib.import_module(f".window_extractors.{arm}", __package__)
                for t in tasks:
                    try:
                        ro, row = build_one(arm, t, ds, mod, common, BASELINE_ADAPTERS, build_mod, redo=redo)
                    except Exception as e:
                        row = {"task": t, "key": arm + (".ref" if ds == "ref" else ""), "error": f"{type(e).__name__}: {str(e)[:200]}"}
                        ro = None
                    if ro is not None:
                        rows.append(row)
                    f.write(json.dumps({"dataset": ds, "arm": arm, **row}) + "\n")
                    f.flush()
                    st = row.get("status", "ERR")
                    print(f"  {row.get('system', row.get('key')):24s} {t}  {st:4s} words={row.get('words', '-')} marks={row.get('citation_marks', '-')} "
                          f"res={row.get('resolution_rate', '-')} llm2draft={row.get('n_llm_to_draft', '-')} {'(cached)' if row.get('cached') else ''}{row.get('bot_reason', '') or row.get('error', '')}", flush=True)
    print(f"done: {len(rows)} rollouts for {arms} x {datasets}; run `draft_rollouts conformance` to merge E1/conformance.csv")


def conformance() -> None:
    C.bootstrap_env()
    from ccbench import build as build_mod
    from ccbench.model import Rollout
    root = Path(C.CCB_OUT / "rollouts")
    rows = []
    for d in sorted(root.glob("*.draft")):
        for fp in sorted(d.glob("pmcid_*.json")):
            ro = Rollout.from_json(fp)
            r = build_mod._row(ro, 0.0)
            r["n_llm_to_draft"] = ro.meta.get("n_llm_to_draft")
            r["final_words"] = ro.meta.get("final_words")
            rows.append(r)
    conf_p = C.SOURCES["conformance"]
    old = list(csv.DictReader(open(conf_p))) if conf_p.exists() else []
    fields = list(old[0].keys()) if old else list(rows[0].keys())
    new_keys = {r["system"] for r in rows}
    kept = [r for r in old if r["system"] not in new_keys]
    with open(conf_p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
        w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["system"], [0, 0])
        by[r["system"]][0] += 1
        by[r["system"]][1] += (r["status"] == "ok")
    print(f"conformance.csv: {len(kept)} rows kept + {len(rows)} draft rows -> {conf_p}")
    for k, (n, ok) in sorted(by.items()):
        print(f"  {k:24s} {ok}/{n} ok")
    side = C.OUT_DEFAULT / "draft" / "draft_resources.csv"
    side.parent.mkdir(parents=True, exist_ok=True)
    with open(side, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["system", "task", "status", "n_llm", "n_llm_to_draft", "words", "final_words"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print(f"draft prefix resources -> {side}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--arms", default=",".join(NATIVE))
    b.add_argument("--tasks", default="all", help="'all' | integer = first N campaign50 tasks | comma list")
    b.add_argument("--dataset", default="both", choices=["native", "ref", "both"])
    b.add_argument("--out", default=str(C.OUT_DEFAULT / "draft"))
    b.add_argument("--redo", action="store_true")
    sub.add_parser("conformance")
    a = ap.parse_args(argv)
    C.bootstrap_env()
    if a.cmd == "conformance":
        conformance()
        return
    from ccbench.ingest import gold
    tasks = gold.campaign50_tasks()
    if a.tasks != "all":
        tasks = tasks[: int(a.tasks)] if a.tasks.isdigit() else a.tasks.split(",")
    datasets = ["native", "ref"] if a.dataset == "both" else [a.dataset]
    build(a.arms.split(","), tasks, datasets, Path(a.out), redo=a.redo)


if __name__ == "__main__":
    main()
