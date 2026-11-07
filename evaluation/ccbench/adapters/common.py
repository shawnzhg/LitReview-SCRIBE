"""Shared adapter code: markdown outline parsing, sentence splitting, citation-marker resolution to
PMIDs and rollout assembly."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ccbench.config import prereg
from ccbench.ingest import gold, history, logs
from ccbench.model import BOT, Context, OutlineNode, Report, Rollout, Sentence, bot_rollout

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
WORD_RE = re.compile(r"\b\w+\b")
PMID_RE = re.compile(r"^\d{4,9}$")


def load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def outline_from_markdown(text: str) -> list[OutlineNode]:
    nodes: list[OutlineNode] = []
    stack: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines()):
        m = HEADER_RE.match(line.strip())
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        if not title:
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        nid = f"h{i}"
        nodes.append(OutlineNode(id=nid, title=title, level=level, parent=stack[-1][1] if stack else None))
        stack.append((level, nid))
    return nodes


def strip_title_number(t: str) -> str:
    return re.sub(r"^\s*(?:[\dIVXivx]+[.\d]*\)?\s+)+", "", t).strip()


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s for s in SENT_SPLIT.split(text) if s.strip()]


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def report_from_markdown(text: str, cite_regex: re.Pattern, resolver, drop_after_header: tuple[str, ...] = ("references", "bibliography")) -> Report:
    sections: list[dict] = []
    sentences: list[Sentence] = []
    bib: dict[str, None] = {}
    unresolved = 0
    total_marks = 0
    cur = {"id": "s0", "title": "", "level": 0, "text": []}
    in_refs = False
    for line in text.splitlines():
        m = HEADER_RE.match(line.strip())
        if m:
            if cur["text"] or cur["title"]:
                sections.append({**cur, "text": "\n".join(cur["text"])})
            title = m.group(2).strip()
            in_refs = strip_title_number(title).lower().rstrip(":") in drop_after_header
            cur = {"id": f"s{len(sections)}", "title": title, "level": len(m.group(1)), "text": []}
            continue
        if in_refs:
            continue
        cur["text"].append(line)
    if cur["text"] or cur["title"]:
        sections.append({**cur, "text": "\n".join(cur["text"])})

    words = 0
    for sec in sections:
        if strip_title_number(sec["title"]).lower().rstrip(":") in drop_after_header:
            continue
        sents, w, marks, unres = section_sentences(sec["id"], sec["text"], cite_regex, resolver, bib)
        sentences += sents
        words += w
        total_marks += marks
        unresolved += unres
    return Report(sections=[{k: v for k, v in s.items()} for s in sections], sentences=sentences, bibliography=list(bib), words=words, unresolved_citations=unresolved, total_citation_marks=total_marks)


def section_sentences(sec_id: str, body: str, cite_regex: re.Pattern, resolver, bib: dict[str, None]) -> tuple[list[Sentence], int, int, int]:
    sentences: list[Sentence] = []
    marks = unresolved = 0
    for j, s in enumerate(split_sentences(body)):
        cites: list[str] = []
        for mk in cite_regex.findall(s):
            marks += 1
            r = resolver(mk)
            if r is None:
                unresolved += 1
                continue
            for p in r:
                cites.append(p)
                bib.setdefault(p, None)
        clean = cite_regex.sub("", s).strip()
        if clean:
            sentences.append(Sentence(sid=f"{sec_id}#{j}", section=sec_id, text=clean, cites=cites))
    return sentences, word_count(cite_regex.sub(" ", body)), marks, unresolved


def numeric_resolver(refmap: dict[str, list[str]]):

    def _r(payload: str):
        out: list[str] = []
        any_ok = False
        for tok in re.split(r"[,;]", payload):
            tok = tok.strip()
            if not tok:
                continue
            m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", tok)
            ids = list(range(int(m.group(1)), int(m.group(2)) + 1)) if m else ([int(tok)] if tok.isdigit() else [])
            for n in ids:
                r = refmap.get(str(n))
                if r:
                    any_ok = True
                    out.extend(r)
        return out if any_ok else None

    return _r


def baseline_context(task: str, task_dir: Path, tools: list[str]) -> Context:
    fp = None
    ph = task_dir / "pool_health.json"
    if ph.exists():
        try:
            fp = load_json(ph).get("stats_fingerprint")
        except Exception:
            fp = None
    return gold.context_for(task, pool_fingerprint=fp, tools=tools)


def assemble(arm: str, task: str, task_dir: Path, *, papers: list[str], outline, report: Report | None, graph: dict | None, final_ok: bool, bot_reason: str | None, extra_meta: dict | None = None) -> Rollout:
    K = int(prereg()["readouts"]["K"])
    status = logs.load_status(task_dir.parent).get(task, {})
    events, totals = history.build_events(task_dir, K)
    tools = sorted({e.route for e in events if e.route})
    ctx = baseline_context(task, task_dir, tools)
    union = history.retrieved_union(events)
    ok = final_ok and (status.get("status", "ok") == "ok")
    ro = Rollout(
        system=arm,
        task=task,
        panel="A",
        context=ctx,
        status="ok" if ok else BOT,
        bot_reason=None if ok else (bot_reason or f"status={status.get('status')} note={status.get('note')}"),
        mode="campaign",
        events=events,
        papers=[p for p in papers if PMID_RE.match(str(p))],
        outline=outline,
        report=report,
        graph=graph,
        resources={**totals, "wall_s": status.get("wall_s"), "n_attempts": status.get("n_attempts", 1)},
        meta={"task_dir": str(task_dir), "retrieved_union_n": len(union), "n_events": len(events), **(extra_meta or {})},
    )
    return ro


def bot(arm: str, task: str, task_dir: Path, reason: str) -> Rollout:
    ctx = baseline_context(task, task_dir, [])
    return bot_rollout(arm, task, "A", ctx, reason, mode="campaign", meta={"task_dir": str(task_dir)})
