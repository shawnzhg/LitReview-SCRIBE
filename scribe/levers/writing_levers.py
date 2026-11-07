#!/usr/bin/env python3
"""Lever writer installed as windows.writing: paragraph breaks, round-robin per-section evidence up
to a cap and a length retry, as fixed in writing_levers.json; install() applies it."""

from __future__ import annotations

import hashlib
import json
import math
import re

SCHEMA = "writing_levers/1"
RELEASE_LEVERS = {"paragraphs": True, "evidence": {"order": "round_robin", "cap": 30},
                  "length": {"tokens_per_word": 3.5, "slack": 0.5, "first_attempt": True}}
PARA_TEXT = ('PARAGRAPHS: organise this section into paragraphs, each developing one point over several sentences. '
             'Begin a new paragraph whenever the point changes. To start a new paragraph, add "new_paragraph": true to '
             'the FIRST sentence object of that paragraph (never to the first paragraph), for example '
             '{"text": "<first sentence of the new paragraph>", "claim_ids": ["c7"], "citations": ["<PMID>"], '
             '"new_paragraph": true}. Never output a separate object for a paragraph break.')
_MARK = re.compile(r"^[\s\[\(<{#*_-]*new[\s_-]*paragraph[\s\]\)>}.:#*_-]*$", re.I)


def is_marker(sent) -> bool:
    return (isinstance(sent, dict) and bool(_MARK.match(str(sent.get("text") or "").strip()))
            and not sent.get("claim_ids") and not sent.get("citations"))


def split_markers(sents):
    out, brk = [], False
    for s in sents:
        if is_marker(s):
            brk = True
            continue
        flag = s.get("new_paragraph") in (True, "true", "True")
        out.append((s, brk or flag))
        brk = False
    return out


def load(path: str) -> dict:
    d = json.load(open(path))
    if d.get("schema") != SCHEMA or d.get("levers") != RELEASE_LEVERS:
        raise ValueError(f"{path}: not the {SCHEMA} lever set {RELEASE_LEVERS}")
    return d["levers"]


def section_caps(W, words):
    g = RELEASE_LEVERS["length"]
    cap = max(1, int(math.ceil(g["tokens_per_word"] * (1 + g["slack"]) * int(words))))
    first = min(W.WRITE_MAX_TOKENS, cap) if g["first_attempt"] else W.WRITE_MAX_TOKENS
    return first, cap


def round_robin_papers(cl, ev):
    per = []
    for c in cl:
        lst, own = [], set()
        for eid in c["evidence_ids"]:
            pid = (ev.get(eid) or {}).get("paper_id")
            if pid and pid not in own:
                own.add(pid)
                lst.append(pid)
        per.append(lst)
    pids, seen = [], set()
    for depth in range(max((len(x) for x in per), default=0)):
        for lst in per:
            if depth < len(lst) and lst[depth] not in seen:
                seen.add(lst[depth])
                pids.append(lst[depth])
    return pids


def make_writing(W, write_user: str):
    ecap = RELEASE_LEVERS["evidence"]["cap"]

    def levers_writing(spec, graph, plan, texts, bundle, client, log, seed=0):
        claims = {c["claim_id"]: c for c in graph["claims"]}
        ev = {e["evidence_id"]: e for e in bundle["evidence"]}
        known_papers = {p["paper_id"] for p in bundle["papers"]}
        budgets, budget_src = W.section_word_budgets(spec, plan)
        sections_out, s_map, c_map, failures = [], [], [], []
        bib, sent_n = [], 0
        empty_sections = []
        sections_salvaged = []
        lever_rows = {}
        for s in plan["sections"]:
            cl = [claims[c] for c in s["claim_ids"] if c in claims]
            cl_txt = "\n".join(f"  {c['claim_id']}: {W.clip(c['text'], W.MAXCH['claim_text'], '...')}"
                               for c in cl) or "  (none: write only connective material)"
            pids = round_robin_papers(cl, ev)
            ev_txt = "\n".join(
                f"  PMID {pid}: {W.clip((texts.get(pid, {}) or {}).get('abstract') or '', 600, '...')}"
                for pid in pids[:ecap]) or "  (none)"
            words = budgets[s["section_id"]]
            first_tok, retry_tok = section_caps(W, words)
            messages = [{"role": "system", "content": W.WRITE_SYS},
                        {"role": "user", "content": write_user.format(
                            question=spec["question"], audience=spec["audience"], title=s["title"],
                            objective=s.get("objective") or "-", words=words,
                            claims=cl_txt, evidence=ev_txt)}]
            salv = {"fired": False}
            obj, draw, err = W.ask_json(client, log, "writing", messages, max_tokens=first_tok, seed=seed,
                                        salvage_keys=("sentences",), salvage=salv, retry_budget=retry_tok)
            sents = W._nonempty_sentences(obj)
            n_markers = 0
            if sents:
                n_markers = sum(is_marker(x) for x in sents)
                if all(is_marker(x) for x in sents):
                    sents = []
            if sents and salv.get("fired"):
                sections_salvaged.append(s["section_id"])
            lever_rows[s["section_id"]] = {"n_papers_available": len(pids), "n_papers_shown": min(len(pids), ecap),
                                           "first_max_tokens": first_tok, "retry_max_tokens": retry_tok,
                                           "budget_words": words}
            if not sents:
                empty_sections.append(s["section_id"])
                failures.append(f"section {s['section_id']}: "
                                f"{err if obj is None else 'no non-empty sentence text'}")
                sections_out.append({"section_id": s["section_id"], "text": "",
                                     "title": s["title"], "sentences": []})
                continue
            paras, sentences = [[]], []
            for sent, brk in split_markers(sents):
                txt = str(sent.get("text") or "").strip()
                sid = f"{s['section_id']}#{sent_n}"
                sent_n += 1
                if brk and paras[-1]:
                    paras.append([])
                paras[-1].append(txt)
                cids = [str(c) for c in (sent.get("claim_ids") or []) if str(c) in claims]
                if cids:
                    s_map.append({"sentence_id": sid, "claim_ids": cids})
                cites = []
                for pid in (sent.get("citations") or []):
                    pid = str(pid).strip().replace("PMID", "").strip()
                    eids = [e["evidence_id"] for e in bundle["evidence"] if e["paper_id"] == pid]
                    c_map.append({"sentence_id": sid, "paper_id": pid, "evidence_ids": eids})
                    cites.append(pid)
                    if pid not in bib:
                        bib.append(pid)
                sentences.append({"sentence_id": sid, "text": txt, "claim_ids": cids,
                                  "citations": cites})
            lever_rows[s["section_id"]]["n_paragraphs"] = len(paras)
            lever_rows[s["section_id"]]["n_marker_objects"] = n_markers
            lever_rows[s["section_id"]]["paragraph_sizes"] = [len(p) for p in paras]
            sec_text = "\n\n".join(" ".join(p) for p in paras)
            sections_out.append({"section_id": s["section_id"], "text": sec_text,
                                 "title": s["title"], "sentences": sentences})
        body = " ".join(x["text"] for x in sections_out)
        n_words = len(body.split())
        fabricated = sorted({c["paper_id"] for c in c_map if c["paper_id"] not in known_papers})
        written = [x["section_id"] for x in sections_out]
        report = W.seal({
            "schema_version": "report_artifact/1.0", "task_id": spec["task_id"],
            "outline_plan_hash": plan["content_hash"], "sections": sections_out,
            "sentence_claim_map": s_map, "citation_evidence_map": c_map,
            "bibliography": bib, "uncertainty_statements": [],
            "terminal_audit": {"n_words": n_words,
                               "target_words": spec["output_spec"]["target_words"],
                               "n_sentences": sent_n,
                               "n_sentences_with_claims": len(s_map),
                               "n_citations": len(c_map),
                               "n_cited_papers": len(bib),
                               "fabricated_citations": fabricated,
                               "fabricated_citation_rate": (round(len(fabricated) / len(bib), 4)
                                                            if bib else None),
                               "section_word_budgets": {k: budgets[k] for k in written},
                               "word_budget_source": {k: budget_src[k] for k in written},
                               "word_budget_rule": W.BUDGET_RULE,
                               "word_budget_sum": sum(budgets[k] for k in written),
                               "sections_empty": empty_sections,
                               "sections_salvaged": sections_salvaged,
                               "write_max_tokens": W.WRITE_MAX_TOKENS,
                               "sentence_source": "stored",
                               "writing_levers": {"levers": RELEASE_LEVERS, "sections": lever_rows}},
            "failures": failures,
            "resource_summary": {}})
        return W._sealed(report, "report_artifact", log, "writing", W.utcnow())
    return levers_writing


def install(W, R, levers: dict) -> dict:
    if levers != RELEASE_LEVERS:
        raise ValueError(f"lever set {levers} is not {RELEASE_LEVERS}")
    wu = W.WRITE_USER + "\n\n" + PARA_TEXT.replace("{", "{{").replace("}", "}}")
    fn = make_writing(W, wu)
    fn._writing_levers = dict(RELEASE_LEVERS)
    W.writing = fn
    ph_before = getattr(R, "PROMPT_HASH", None)
    R.PROMPT_HASH = R.prompt_hash(write_user=wu)
    return {"installed": sorted(RELEASE_LEVERS), "levers": dict(RELEASE_LEVERS),
            "write_user_sha256": hashlib.sha256(wu.encode()).hexdigest(), "para_text": PARA_TEXT,
            "prompt_hash_before": ph_before, "prompt_hash": R.PROMPT_HASH}
