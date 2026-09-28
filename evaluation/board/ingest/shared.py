"""Code shared by the ingest scripts: the task list, hashes, MCP logs, reference lists, citation
resolution, and the build of rollouts and the conformance table."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "evaluation"))
from ccbench.adapters.common import HEADER_RE

X = Path(os.environ.get("COMMERCIAL_OUT", ""))
SHARED_OUT = Path(os.environ.get("CCBENCH_ROOT", "")) / "out"
TASKS_FILE = "tasks.json"
MCP_K = 20
PMID_MIN_DIGITS = 5

REF_LINE = re.compile(r"^\s*(?:[-*]\s*)?\[(\d+)\]\s*(.*)$")
REF_LINE_DOT = re.compile(r"^\s*(\d+)\.\s+(.*)$")
PMID_TOK = re.compile(r"PMID\s*:?\s*(\d{4,9})", re.I)
PMID_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d{4,9})")
RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
NUM_TITLE = re.compile(r"^\s*(?:[\dIVXivx]+[.\d]*\)?\s+)+")


def _pool_fingerprint() -> str:
    spec = importlib.util.spec_from_file_location("pool_retrieval_tool_pin", REPO / "scribe/harness/tools/pool_retrieval_tool.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PIN_STATS_FINGERPRINT


POOL_FINGERPRINT = _pool_fingerprint()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def epoch(ts: str) -> float:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def run_tasks() -> list[str]:
    p = os.environ.get("COMMERCIAL_TASKS")
    if not p:
        raise SystemExit("set COMMERCIAL_TASKS to the JSON task list the runners used")
    t = json.loads(Path(p).read_text())
    if not t or len(set(t)) != len(t):
        raise SystemExit(f"INGEST FAILED: {p} is empty or lists a task twice")
    return list(t)


def stem(tasks: list[str], task: str) -> str:
    return f"{tasks.index(task) + 1:02d}_{task}"


def write_tasks(out: Path, tasks: list[str]) -> None:
    (out / TASKS_FILE).write_text(json.dumps(tasks, indent=1))


def converted_tasks(runs: Path) -> list[str]:
    return json.loads((Path(runs) / TASKS_FILE).read_text())


def source_hashes(names) -> dict:
    out = {"COMMERCIAL_TASKS": sha256(Path(os.environ["COMMERCIAL_TASKS"]))}
    for f in names:
        if (X / f).exists():
            out[f] = sha256(X / f)
    return out


def jsonl(p: Path) -> list[dict]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def review_pmid(task: str) -> str:
    from ccbench import paths
    return str(json.load(open(paths.gold_path(task)))["review_pmid"])


def load_mcp_log(name: str) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    with open(X / "mcp_call_logs" / name) as f:
        for i, l in enumerate(f):
            d = json.loads(l)
            if not d.get("task"):
                continue
            d["_line"] = i
            by.setdefault(d["task"], []).append(d)
    return by


def is_ref_heading(line: str, heads=("references", "bibliography")) -> bool:
    m = HEADER_RE.match(line.strip())
    return bool(m) and NUM_TITLE.sub("", m.group(2)).strip().lower().rstrip(":") in heads


def references(text: str) -> tuple[dict[str, list[str]], dict]:
    lines = text.splitlines()
    start = None
    for i, l in enumerate(lines):
        if is_ref_heading(l):
            start = i + 1
    st = {"has_references_heading": start is not None, "n_entries": 0, "n_entries_no_pmid": 0, "n_duplicate_numbers": 0,
          "n_multi_pmid_entries": 0}
    refmap: dict[str, list[str]] = {}
    if start is None:
        return refmap, st
    for l in lines[start:]:
        if HEADER_RE.match(l.strip()):
            break
        m = REF_LINE.match(l) or REF_LINE_DOT.match(l)
        if not m:
            continue
        n, body = m.group(1), m.group(2)
        st["n_entries"] += 1
        pm = list(dict.fromkeys(PMID_TOK.findall(body))) or list(dict.fromkeys(PMID_URL.findall(body)))
        if not pm:
            st["n_entries_no_pmid"] += 1
            continue
        if len(pm) > 1:
            st["n_multi_pmid_entries"] += 1
        if n in refmap:
            st["n_duplicate_numbers"] += 1
            continue
        refmap[n] = pm
    return refmap, st


def strip_preamble(text: str) -> tuple[str, dict]:
    lines = text.splitlines()
    first = next((i for i, l in enumerate(lines) if HEADER_RE.match(l.strip())), None)
    if first is None or first == 0:
        return text, {"stripped": False, "reason": "no heading" if first is None else "starts with a heading", "words": 0}
    pre = [l for l in lines[:first] if l.strip() and not RULE_RE.match(l)]
    if not pre:
        return text, {"stripped": False, "reason": "only blank / rule lines before the first heading", "words": 0}
    if len(pre) != 1:
        return text, {"stripped": False, "reason": f"{len(pre)} text lines before the first heading (kept, flagged)",
                      "words": sum(len(l.split()) for l in pre), "text": " | ".join(pre)[:300]}
    return "\n".join(lines[first:]) + ("\n" if text.endswith("\n") else ""), {
        "stripped": True, "words": len(pre[0].split()), "text": pre[0][:300], "lines_dropped": first}


def cite_resolver(refmap: dict[str, list[str]], stats: Counter, pfx: str = ""):

    def _r(payload: str):
        out: list[str] = []
        any_ok = False
        for tok in re.split(r"[,;]", payload):
            tok = tok.strip()
            if not tok:
                continue
            m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", tok)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if max(len(m.group(1)), len(m.group(2))) >= PMID_MIN_DIGITS or b - a > 200:
                    stats[pfx + "tok_range_rejected"] += 1
                    continue
                stats[pfx + "tok_range"] += 1
                ids = [str(n) for n in range(a, b + 1)]
            elif tok.isdigit():
                if len(tok) >= PMID_MIN_DIGITS:
                    stats[pfx + "tok_pmid"] += 1
                    any_ok = True
                    out.append(str(int(tok)))
                    continue
                stats[pfx + "tok_num"] += 1
                ids = [str(int(tok))]
            else:
                stats[pfx + "tok_other"] += 1
                ids = []
            for n in ids:
                r = refmap.get(n)
                if r:
                    any_ok = True
                    out.extend(r)
                else:
                    stats[pfx + "tok_num_not_in_refs"] += 1
        return out if any_ok else None

    return _r


def pool_health(td: Path) -> None:
    (td / "pool_health.json").write_text(json.dumps({"ok": True, "stats_fingerprint": POOL_FINGERPRINT,
                                                     "source": "the pool service"}))


def write_status(arm: Path, lines: list[dict]) -> None:
    with open(arm / "_arm/task_status.jsonl", "w") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")


def build(runs: Path, keys_by_cond: dict[str, str], adapt, keys: list[str]) -> int:
    from ccbench import paths
    from ccbench.build import _row, rollout_path
    out = Path(paths.OUT).resolve()
    if out == SHARED_OUT.resolve() or str(out).startswith(str(SHARED_OUT.resolve())):
        print(f"INGEST REFUSED: CCBENCH_OUT={out} is the SHARED out; build writes only into a private out", file=sys.stderr)
        return 2
    tasks = converted_tasks(runs)
    rows = []
    for cond, key in keys_by_cond.items():
        if key not in keys:
            continue
        if (out / "rollouts" / key).exists():
            print(f"INGEST REFUSED: {out}/rollouts/{key} exists", file=sys.stderr)
            return 2
        for task in tasks:
            ro = adapt(task, runs / cond / task, key, cond)
            ro.system = key
            ro.meta["campaign"] = cond
            if cond == "fixinput":
                ro.meta["provenance_tier"] = "reference_derived"
            ro.to_json(rollout_path(key, "campaign", task))
            rows.append(_row(ro, 0.0))
            r = rows[-1]
            print(f"{key:24s} {task} {ro.status:4s} P={len(ro.papers):4d} words={r['words']:6d} bib={r['n_bib']:4d} "
                  f"marks={r['citation_marks']} unres={r['unresolved']}", flush=True)
    e1 = out / "E1"
    e1.mkdir(parents=True, exist_ok=True)
    if (e1 / "conformance.csv").exists():
        print(f"INGEST REFUSED: {e1}/conformance.csv exists", file=sys.stderr)
        return 2
    hdr = open(SHARED_OUT / "E1/conformance.csv").readline().strip().split(",")
    if hdr != list(rows[0].keys()):
        print(f"INGEST FAILED: conformance header differs from the shared one: {list(rows[0].keys())} vs {hdr}", file=sys.stderr)
        return 3
    with open(e1 / "conformance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return 0


def main(argv, doc: str, convert, adapt, keys_by_cond: dict[str, str]) -> int:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--runs", required=True)
    b.add_argument("--keys", default=",".join(keys_by_cond.values()))
    a = ap.parse_args(argv)
    if a.cmd == "convert":
        if not os.environ.get("COMMERCIAL_OUT"):
            raise SystemExit("set COMMERCIAL_OUT to the agent's run bundle")
        out = Path(a.out)
        if out.exists():
            print(f"INGEST REFUSED: {out} exists (conversion dirs are never overwritten)", file=sys.stderr)
            return 2
        return convert(out)
    return build(Path(a.runs), keys_by_cond, adapt, a.keys.split(","))
