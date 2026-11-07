"""Converts the Gemini Deep Research reports in $COMMERCIAL_OUT into scorer runs under a fixed-input
and a same-pool key and builds their rollouts. Usage: python gemini_deep_research.py convert
--out <dir> | build --runs <dir>."""

from __future__ import annotations

import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shared as S
from ccbench.adapters.common import HEADER_RE

KEYS = {"fixinput": "gemini_web_dr.ref", "samepool": "gemini_web_dr"}
REF_HEADS = ("references", "bibliography", "works cited", "sources")
PRODUCT = "Gemini Deep Research (web app)"
VENDOR_MODEL = "gemini-3.8-flash"

CHIP_ADJ = re.compile(r"(?<=\])\[(\d+)\]")
CITE_ALL = re.compile(r"(\[[\d,\s;–-]+\]|\[chip:\d+\])")
GEM_RESIDUE = re.compile(r"\[cite:\s*[^\]\n]*\]")
PMC_URL = re.compile(r"(PMC\d{4,9})")


def head_kind(line: str) -> str | None:
    m = HEADER_RE.match(line.strip())
    if not m:
        return None
    t = S.NUM_TITLE.sub("", m.group(2)).strip().lower().rstrip(":").strip("*").strip()
    return t if t in REF_HEADS else "other"


def ref_sections(text: str) -> list[dict]:
    lines = text.splitlines()
    heads = [(i, head_kind(l)) for i, l in enumerate(lines)]
    heads = [(i, k) for i, k in heads if k is not None]
    out = []
    for j, (i, k) in enumerate(heads):
        if k == "other":
            continue
        end = next((i2 for i2, _ in heads[j + 1:]), len(lines))
        ents = []
        for l in lines[i + 1:end]:
            m = S.REF_LINE.match(l) or S.REF_LINE_DOT.match(l)
            if not m:
                continue
            body = m.group(2)
            pm = list(dict.fromkeys(S.PMID_TOK.findall(body))) or list(dict.fromkeys(S.PMID_URL.findall(body)))
            pmc = PMC_URL.findall(body)
            title = re.sub(r"https?://\S+", "", S.PMID_TOK.sub("", body)).strip(" .,\t")
            ents.append({"n": m.group(1), "pmids": [str(int(p)) for p in pm], "pmc": pmc[0] if pmc else None, "title": title,
                         "raw": body, "uploaded": "(uploaded file)" in body})
        out.append({"kind": "works" if k == "works cited" else "model", "heading": k, "start": i, "end": end, "entries": ents})
    return out


def refmap_of(sec: dict | None) -> tuple[dict[str, list[str]], dict]:
    st = {"n_entries": 0, "n_entries_no_pmid": 0, "n_pmc_only": 0, "n_uploaded_file": 0, "n_duplicate_numbers": 0,
          "n_multi_pmid_entries": 0}
    rm: dict[str, list[str]] = {}
    if sec is None:
        return rm, st
    for e in sec["entries"]:
        st["n_entries"] += 1
        if not e["pmids"]:
            st["n_entries_no_pmid"] += 1
            st["n_pmc_only"] += int(bool(e["pmc"]) and not e["uploaded"])
            st["n_uploaded_file"] += int(e["uploaded"])
            continue
        if len(e["pmids"]) > 1:
            st["n_multi_pmid_entries"] += 1
        if e["n"] in rm:
            st["n_duplicate_numbers"] += 1
            continue
        rm[e["n"]] = e["pmids"]
    return rm, st


def body_outside_refs(text: str) -> str:
    keep, skip = [], False
    for l in text.splitlines():
        k = head_kind(l)
        if k is not None:
            skip = k != "other"
        if not skip:
            keep.append(l)
    return "\n".join(keep)


def without_works_cited(text: str) -> str:
    keep, skip = [], False
    for l in text.splitlines():
        k = head_kind(l)
        if k is not None:
            skip = k == "works cited"
        if not skip:
            keep.append(l)
    return "\n".join(keep) + ("\n" if text.endswith("\n") else "")


def mark_resolver(maps: dict, mode: str, stats: Counter):
    br = S.cite_resolver(maps["works"] if mode == "chips" else maps["model"], stats)
    ch = S.cite_resolver(maps["works"], stats, "chip_")

    def _r(mark: str):
        if mark.startswith("[chip:"):
            stats["marks_chip_adjacent"] += 1
            return ch(mark[6:-1])
        stats["marks_bracket"] += 1
        return br(mark[1:-1])

    return _r


def analyse(raw: str) -> dict:
    text, pre = S.strip_preamble(raw)
    secs = ref_sections(text)
    model = [s for s in secs if s["kind"] == "model"]
    works = [s for s in secs if s["kind"] == "works"]
    if len(works) > 1:
        raise SystemExit("INGEST FAILED: more than one 'Works cited' section")
    msec, wsec = (model[-1] if model else None), (works[0] if works else None)
    adj = len(CHIP_ADJ.findall(body_outside_refs(text)))
    mode = "model" if wsec is None else "chips" if msec is None else "mixed"
    scored = text
    if mode == "mixed":
        out, skip = [], False
        for l in text.splitlines():
            k = head_kind(l)
            if k is not None:
                skip = k != "other"
            out.append(l if skip else CHIP_ADJ.sub(lambda m: f"[chip:{m.group(1)}]", l))
        scored = "\n".join(out) + ("\n" if text.endswith("\n") else "")
    mm, mst = refmap_of(msec)
    wm, wst = refmap_of(wsec)
    P: list[str] = []
    for rm in ((wm, mm) if mode == "chips" else (mm, wm)):
        for n in sorted(rm, key=int):
            for p in rm[n]:
                if p not in P:
                    P.append(p)
    return {"text": scored, "preamble": pre, "mode": mode, "sections": secs, "model_sec": msec, "works_sec": wsec,
            "maps": {"model": mm, "works": wm}, "model_stats": mst, "works_stats": wst, "adjacent_marks": adj, "P": P}


def inline_cited(a: dict) -> tuple[set[str], int, int, Counter]:
    cst: Counter = Counter()
    rsv = mark_resolver(a["maps"], a["mode"], cst)
    cited: set[str] = set()
    n = un = 0
    for mk in CITE_ALL.findall(body_outside_refs(a["text"])):
        n += 1
        r = rsv(mk)
        if r is None:
            un += 1
        else:
            cited.update(r)
    return cited, n, un, cst


def sources_summary(sj: dict, rp: str, rpmc: str) -> dict:
    su, sr = sj.get("sources_used") or [], sj.get("sources_read_not_used") or []
    pm = lambda L: sorted({m.group(1) for s in L for m in [S.PMID_URL.search(s.get("url") or "")] if m})
    return {"n_used": len(su), "n_read_not_used": len(sr), "domains_used": dict(Counter(s.get("domain") for s in su)),
            "domains_read_not_used": dict(Counter(s.get("domain") for s in sr)),
            "pubmed_pmids_used": len(pm(su)), "pubmed_pmids_read_not_used": len(pm(sr)),
            "target_review_opened": any(rp in (s.get("url") or "") or rpmc in (s.get("url") or "") for s in su + sr),
            "scored": False}


def convert(out: Path) -> int:
    tasks = S.run_tasks()
    summary = {"schema": "gemini_deep_research_convert/3", "source": str(S.X), "arms": {},
               "source_sha256": S.source_hashes(("audit.json",))}
    for cond, key in KEYS.items():
        (out / cond / "_arm").mkdir(parents=True)
        st_lines, per = [], {}
        for task in tasks:
            sm = S.stem(tasks, task)
            md, sources = S.X / "runs" / cond / f"{sm}.md", S.X / "runs" / cond / f"{sm}.sources.json"
            td = out / cond / task
            td.mkdir()
            raw = md.read_text(errors="replace")
            a = analyse(raw)
            cited, n_marks, n_unres, cst = inline_cited(a)
            call = {"seq": 0, "t_req": None, "t_done": None, "wall_ms": None, "status": 200,
                    "path": f"{PRODUCT}: one new chat, Deep Research on, plan accepted unedited",
                    "model": VENDOR_MODEL, "requested_model": VENDOR_MODEL, "finish_reason": "completed (exported by hand)",
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None},
                    "cost_usd": None, "prompt": "", "completion": raw,
                    "params": {"cost_note": "web app: no token or cost metering; timing not logged"}}
            (td / "_calls.jsonl").write_text(json.dumps(call) + "\n")
            shutil.copy2(md, td / "report.md")
            shutil.copy2(sources, td / "sources.json")
            sj = json.load(open(sources))
            rp, rpmc = S.review_pmid(task), task.replace("pmcid_", "")
            ref_p = {p for m in a["maps"].values() for v in m.values() for p in v}
            prov = {"key": key, "source_md": str(md), "report_sha256": S.sha256(md), "sources_json": str(sources),
                    "sources_json_sha256": S.sha256(sources), "vendor_model": VENDOR_MODEL,
                    "citation_mode": a["mode"], "adjacent_marks": a["adjacent_marks"],
                    "lists": {"model": a["model_stats"], "works": a["works_stats"]},
                    "P_rule": "the PMIDs of the report's own reference lists, the list the marks index first; sources.json is not scored",
                    "P": len(a["P"]), "marks": n_marks, "marks_unresolved": n_unres, "citation_tokens": dict(cst),
                    "gemini_cite_residue": len(GEM_RESIDUE.findall(body_outside_refs(a["text"]))),
                    "preamble": a["preamble"], "sources_summary": sources_summary(sj, rp, rpmc),
                    "target_review": {"pmid": rp, "in_lists": rp in ref_p, "cited_inline": rp in cited}}
            (td / "provenance.json").write_text(json.dumps(prov, indent=1))
            ok = bool(raw.strip())
            st_lines.append({"task": task, "status": "ok" if ok else "failed", "wall_s": None,
                             **({} if ok else {"note": "no_report: exported .md is empty"})})
            per[task] = {"stem": sm[:2], "mode": a["mode"], "P": len(a["P"]), "marks": n_marks, "unresolved": n_unres,
                         "has_report": ok}
        S.write_status(out / cond, st_lines)
        v = list(per.values())
        summary["arms"][cond] = {"key": key, "tasks": len(v), "no_report": [x["stem"] for x in v if not x["has_report"]],
                                 "modes": {m: sorted(x["stem"] for x in v if x["mode"] == m) for m in ("chips", "mixed")},
                                 "marks": sum(x["marks"] for x in v), "unresolved": sum(x["unresolved"] for x in v),
                                 "unresolved_by_run": {x["stem"]: x["unresolved"] for x in v if x["unresolved"]},
                                 "per_task": per}
    S.write_tasks(out, tasks)
    (out / "convert_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({a: {k: v for k, v in s.items() if k != "per_task"} for a, s in summary["arms"].items()}, indent=1))
    return 0


def adapt(task: str, task_dir: Path, key: str, cond: str):
    from ccbench.adapters import common
    from ccbench.ingest import logs
    status = logs.load_status(task_dir.parent).get(task, {})
    if status.get("status") == "failed":
        note = str(status.get("note") or "failed")
        return common.bot(key, task, task_dir, note if note.startswith("no_report") else "no_report: " + note)
    rp = task_dir / "report.md"
    if not rp.exists():
        return common.bot(key, task, task_dir, "no_report: report.md missing")
    raw = rp.read_text(errors="replace")
    if not raw.strip():
        return common.bot(key, task, task_dir, "no_report: report.md empty")
    prov = json.load(open(task_dir / "provenance.json"))
    a = analyse(raw)
    if a["mode"] != prov["citation_mode"] or len(a["P"]) != prov["P"]:
        raise SystemExit(f"INGEST FAILED: {task_dir}: adapter analysis differs from the conversion's provenance")
    cst: Counter = Counter()
    report = common.report_from_markdown(a["text"], CITE_ALL, mark_resolver(a["maps"], a["mode"], cst), drop_after_header=REF_HEADS)
    outline = common.outline_from_markdown(without_works_cited(a["text"]))
    return common.assemble(key, task, task_dir, papers=a["P"], outline=outline, report=report, graph=None, final_ok=True,
                           bot_reason=None,
                           extra_meta={"citation_mode": a["mode"], "lists": {"model": a["model_stats"], "works": a["works_stats"]},
                                       "n_reference_entries": a["model_stats"]["n_entries"] + a["works_stats"]["n_entries"],
                                       "preamble": a["preamble"], "citation_tokens": dict(cst),
                                       "gemini_cite_residue_uncounted": prov["gemini_cite_residue"],
                                       "adapter": "evaluation/board/ingest/gemini_deep_research.py:adapt", "draft_is_final": True,
                                       "backbone": VENDOR_MODEL, "vendor_model": VENDOR_MODEL, "P_rule": prov["P_rule"],
                                       "agent": PRODUCT})


def main(argv=None) -> int:
    return S.main(argv, __doc__, convert, adapt, KEYS)


if __name__ == "__main__":
    raise SystemExit(main())
