#!/usr/bin/env python3
"""The SCRIBE windows as functions: acquisition ranking and bundle, synthesis (claim graph with
relations) and planning (outline assigning claims to sections), plus the writing prompts and word
budgets of the lever writer; each window maps a sealed entry artifact to a sealed exit artifact."""

from __future__ import annotations

from calllog import Timer
from common import utcnow
from common import seal, validate

MAXCH = {"paper_block": 1400, "claim_text": 300, "claim_listed": 180}

CTX_CHARS = 100_000


def fit(items, render, budget=CTX_CHARS):
    out, used = [], 0
    for it in items:
        t = render(it)
        if used + len(t) > budget and out:
            break
        out.append(it)
        used += len(t)
    return out, len(items) - len(out)


def clip(s, n, marker="\n...[CLIPPED]"):
    s = s or ""
    return s if len(s) <= n else s[:n] + marker


def ask_json(client, log, window, messages, max_tokens, seed, expect="object", retries=1,
             artifact_delta=None, salvage_keys=None, salvage=None, retry_budget=None):
    msgs = list(messages)
    last_err, last_draw = None, None
    budget = max_tokens
    truncated_draws = []
    for attempt in range(retries + 1):
        with Timer() as t:
            draw = client.chat(msgs, max_tokens=budget, seed=seed)
        log.llm(window, t.started, draw.wall_ms, draw.model, draw.prompt, draw.completion,
                draw.params, draw.finish_reason, draw.n_prompt_tokens, draw.n_completion_tokens,
                role_sequence=draw.role_sequence, artifact_delta=artifact_delta)
        last_draw = draw
        if draw.error:
            log.failure(window, t.started, "LLMTransportError", draw.error, recovered=False)
            return None, draw, draw.error
        from llm import parse_json_block
        obj, err = parse_json_block(draw.completion)
        if obj is not None and (expect != "object" or isinstance(obj, dict)):
            return obj, draw, None
        last_err = err or f"expected a JSON {expect}, got {type(obj).__name__}"
        if draw.truncated:
            truncated_draws.append((attempt, budget, draw))
        if salvage_keys and expect == "object" and attempt >= retries and truncated_draws:
            from llm import salvage_truncated_arrays
            best = None
            for s_attempt, s_budget, s_draw in truncated_draws:
                got = salvage_truncated_arrays(s_draw.completion, salvage_keys)
                n_got = sum(len(v) for v in got.values()) if got else 0
                if got and (best is None or n_got > best[0]):
                    best = (n_got, s_attempt, s_budget, s_draw, got)
            if best is not None:
                _, s_attempt, s_budget, s_draw, got = best
                counts = {k: len(v) for k, v in got.items()}
                if isinstance(salvage, dict):
                    salvage.update({"fired": True, "attempt": s_attempt, "max_tokens": s_budget,
                                    "attempts_made": attempt + 1,
                                    "keys": sorted(got), "n_items": counts,
                                    "parse_error": last_err})
                log.failure(window, t.started, "TruncatedJSONSalvaged",
                            f"finish_reason=length at max_tokens={s_budget} (attempt "
                            f"{s_attempt + 1} of {attempt + 1}); every retry failed to parse; "
                            f"kept complete elements only: {counts}",
                            recovered=True, kind="retry")
                return got, s_draw, None
        log.failure(window, t.started, "JSONParseFailure", last_err,
                    recovered=attempt < retries, kind="retry" if attempt < retries else "failure")
        if attempt < retries:
            if draw.truncated:
                budget = retry_budget or min(max_tokens * 3, 12000)
                msgs = msgs + [
                    {"role": "user", "content":
                        "Your previous reply was cut off by the length limit. Reply again with "
                        f"ONLY the JSON {expect}, on a single line, no whitespace or "
                        "indentation, no prose, no code fence."}]
            else:
                msgs = msgs + [
                    {"role": "assistant", "content": clip(draw.completion, 500)},
                    {"role": "user", "content":
                        f"That did not parse as JSON ({last_err}). Reply with ONLY the JSON "
                        f"{expect}, no prose, no code fence, no trailing text."}]
    return None, last_draw, last_err


def _sealed(artifact, name, log, window, started):
    errs = validate(name, artifact)
    log.validate_event(window, started, name, errs)
    return artifact, errs


ACQ_SYS = (
    "You are a biomedical literature-review agent doing the EVIDENCE ACQUISITION stage. "
    "Your only job right now is to find the papers a review on this topic must be built from. "
    "You do not write prose and you do not draw conclusions in this stage.")

ACQ_RANK = """TOPIC: {question}
PUBLICATION CUTOFF: {cutoff}

You retrieved these candidate papers. Select the ones a literature review on this topic should
be built from, and put them in order of importance to that review (most important first).
Select at most {K}. Do not invent identifiers: every id you return must appear below. Do not
repeat an id.

{candidates}

Reply with ONLY one JSON object, on a SINGLE LINE, no indentation:
  {{"selected": ["<paper_id>", "<paper_id>", ...]}}"""


def _rank_once(client, log, spec, cands, K, seed, known):
    listing = "\n".join(
        f"  {p['paper_id']} [{p.get('year')}] {clip(p.get('title') or '', 120, '...')}"
        for p in cands)
    obj, draw, err = ask_json(
        client, log, "acquisition",
        [{"role": "system", "content": ACQ_SYS},
         {"role": "user", "content": ACQ_RANK.format(
             question=spec["question"], cutoff=spec["publication_cutoff"], K=K,
             candidates=listing)}],
        max_tokens=900, seed=seed)
    out, seen, fabricated = [], set(), []
    if obj:
        for pid in (obj.get("selected") or []):
            pid = str(pid).strip()
            if pid not in known:
                fabricated.append(pid)
            elif pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out, fabricated, (draw.completion if draw else ""), err


def acquisition(spec, tool, client, log, seed=0):
    K = spec["budget"]["max_ranked_output_K"]
    history, stop_reason = [], "model_stop"
    pool = sorted(tool.pool.values(), key=lambda r: -(r.get("score") or 0))
    BATCH, PER_BATCH = 120, max(K // 2, 20)
    selected, all_fabricated = [], []
    if pool:
        known = {p["paper_id"] for p in pool}
        batches = [pool[i:i + BATCH] for i in range(0, len(pool), BATCH)]
        survivors = []
        for bi, batch in enumerate(batches):
            got, fab, _c, _e = _rank_once(client, log, spec, batch, PER_BATCH,
                                          seed + 100 + bi, known)
            all_fabricated += fab
            survivors += got
        by_id = {p["paper_id"]: p for p in pool}
        if len(survivors) > K and len(batches) > 1:
            finals, fab, _c, _e = _rank_once(
                client, log, spec, [by_id[x] for x in survivors if x in by_id], K,
                seed + 1, set(survivors))
            all_fabricated += fab
            selected = finals or survivors
        else:
            selected = survivors
        if all_fabricated:
            log.failure("acquisition", utcnow(), "FabricatedPaperIds",
                        f"{len(all_fabricated)} selected ids were never retrieved: "
                        f"{all_fabricated[:10]}", recovered=True)
    selected = selected[:K]
    if not selected:
        selected = [p["paper_id"] for p in pool[:K]]
        rank_source = "tool_score_fallback"
    else:
        rank_source = "agent_ranking"

    evidence = []
    for i, pid in enumerate(selected):
        if tool.remaining("open") == 0:
            log.failure("acquisition", utcnow(), "BudgetExhausted",
                        f"document-open budget spent after {i} of {len(selected)} selected",
                        kind="budget_refusal", recovered=True)
            break
        rec = tool.open_document(pid, step=i)
        if not rec or rec.get("refused"):
            continue
        if rec.get("abstract"):
            evidence.append({"evidence_id": f"e_{pid}", "paper_id": pid,
                             "granularity": "abstract", "locator": f"pmid{pid}",
                             "text_hash": tool.docs.span_hash(rec.get("abstract") or ""),
                             "condition": None})
        for s in (rec.get("sentences") or []):
            evidence.append({"evidence_id": f"e_{pid}_{s['locator'].rsplit('_', 1)[1]}",
                             "paper_id": pid, "granularity": "abstract_sentence",
                             "locator": s["locator"],
                             "text_hash": tool.docs.span_hash(s.get("text") or ""),
                             "condition": None})

    papers = []
    for rank, pid in enumerate(selected, 1):
        p = tool.pool.get(pid, {})
        papers.append({
            "paper_id": pid, "doi": p.get("doi"), "title": p.get("title"),
            "year": int(p["year"]) if p.get("year") is not None else None,
            "rank": rank, "score": float(p.get("score") or 0.0),
            "first_seen_step": p.get("first_seen_step"),
            "retrieval_provenance": {"query": p.get("query", ""), "tool": "pool_search",
                                     "rank_from_tool": p.get("rank"),
                                     "route": p.get("route") or "both"},
            "decision": "include", "decision_reason": rank_source,
            "post_cutoff": bool(p.get("year") and int(p["year"]) >= spec["publication_cutoff"])})

    rep = tool.report()
    status = "budget_exhausted" if (tool.exhausted["search"] or tool.exhausted["open"]) else "done"
    if stop_reason == "query_generation_failed" and not papers:
        status = "failed"
    bundle = seal({
        "schema_version": "evidence_bundle/1.0", "task_id": spec["task_id"],
        "task_spec_hash": spec["content_hash"], "provenance_tier": "agent_retrieved",
        "retrieval_status": status, "papers": papers, "evidence": evidence,
        "discovered_union": sorted(tool.pool.keys()),
        "failures": [stop_reason] if stop_reason != "model_stop" else [],
        "budget_remaining": {"search_calls": tool.remaining("search"),
                             "document_opens": tool.remaining("open")},
        "validation": {"ranking": rank_source, "queries": history,
                       "stop_reason": stop_reason, "pool_size": len(tool.pool),
                       "post_cutoff_suppressed": rep["used"]["post_cutoff_suppressed"],
                       "n_selected": len(selected), "K": K,
                       "fabricated_ids": all_fabricated[:50],
                       "n_fabricated_ids": len(all_fabricated)}})
    return _sealed(bundle, "evidence_bundle", log, "acquisition", utcnow())


SYN_SYS = (
    "You are doing the EVIDENCE SYNTHESIS stage of a biomedical literature review. You have a "
    "SEALED evidence set: you cannot search, and you may not use any paper or fact that is not "
    "in the material given to you. Every claim you state must cite the evidence ids it rests on.")

SYN_EXTRACT = """TOPIC: {question}

EVIDENCE (each paper's abstract, with the evidence id to cite):
{papers}

Extract the substantive findings THIS material supports. One claim per finding. Do not
generalise beyond what the text says, and do not add background knowledge of your own.

Reply with ONLY one JSON object:
{{"claims": [
   {{"text": "<the finding, one sentence>",
     "type": "study_finding",
     "polarity": "positive|negative|mixed|unknown",
     "evidence_ids": ["<id from above>", ...],
     "confidence": 0.0-1.0}}
]}}"""

SYN_REL = """TOPIC: {question}

Claims extracted from separate papers:
{claims}

Which of these claims stand in a relation to each other? Most pairs worth recording are two
studies addressing THE SAME QUESTION -- in parallel, or with contrasting results. Those are
`compares`. Use `supports` when one claim is evidence for another, and `qualifies` when one
narrows or conditions another.

Work through the claims systematically. A claim can appear in several relations. Use claim ids
exactly as listed. One relation per line, no prose, no JSON:

    c12 compares c88
    c3 supports c41
    c57 qualifies c9
"""

SYN_CROSS = """TOPIC: {question}

You extracted these claims from separate papers:
{claims}

Now the higher-level claims a review adds on top of them:
  - cross_study_synthesis: a finding that holds across several papers
  - conflict: a place where the papers genuinely disagree
  - limitation: what this evidence base cannot support
  - gap: what is missing

Give AT MOST {max_new} new claims. Each names the claims it rests on in `from_claims`: AT MOST
5, the most important ones. Do not list every claim id you can see.

Reply with ONLY one JSON object:
{{"claims": [{{"text": "...", "type": "cross_study_synthesis|conflict|limitation|gap",
              "polarity": "positive|negative|mixed|unknown",
              "from_claims": ["<claim_id>", "..."], "confidence": 0.0-1.0}}],
  "unresolved_conflicts": ["<claim_id>", ...],
  "missing_information": ["<short phrase>", ...]}}"""


def parse_relation_lines(text, known):
    import re
    out, seen, bad = [], set(), 0
    for line in (text or "").splitlines():
        m = re.match(r"\s*[-*]?\s*(c\d+)\s+(supports|contradicts|qualifies|compares)\s+(c\d+)\s*$",
                     line.strip(), re.I)
        if not m:
            if line.strip():
                bad += 1
            continue
        a, ty, b = m.group(1), m.group(2).lower(), m.group(3)
        if a in known and b in known and a != b and (a, b, ty) not in seen:
            seen.add((a, b, ty))
            out.append({"source": a, "target": b, "type": ty})
    return out, bad

CLAIM_TYPES = {"study_finding", "cross_study_synthesis", "conflict", "limitation", "gap"}
POLARITY = {"positive", "negative", "mixed", "unknown"}


def _paper_blocks(bundle, texts, chunk):
    out = []
    ev_by_paper = {}
    for e in bundle["evidence"]:
        ev_by_paper.setdefault(e["paper_id"], []).append(e)
    for p in chunk:
        pid = p["paper_id"]
        evs = ev_by_paper.get(pid) or []
        pref = next((e for e in evs if e["granularity"] == "abstract"), None)
        eid = pref["evidence_id"] if pref else (evs[0]["evidence_id"] if evs else f"e_{pid}")
        body = texts.get(pid, {}).get("abstract") or ""
        if not body and evs:
            body = " ".join((texts.get(pid, {}).get("sentences") or {}).get(e["locator"], "")
                            for e in evs[:12]).strip()
        out.append(f"[{eid}] ({p.get('year')}) {clip(p.get('title') or '', 200, '...')}\n"
                   f"{clip(body, MAXCH['paper_block'])}")
    return "\n\n".join(out)


def synthesis(spec, bundle, texts, client, log, seed=0, chunk_size=8):
    papers = bundle["papers"]
    valid_ev = {e["evidence_id"] for e in bundle["evidence"]}
    chunks = [papers[i:i + chunk_size] for i in range(0, len(papers), chunk_size)]

    claims, n_dropped_ev, failures = [], 0, []
    for ci, chunk in enumerate(chunks):
        obj, draw, err = ask_json(
            client, log, "synthesis",
            [{"role": "system", "content": SYN_SYS},
             {"role": "user", "content": SYN_EXTRACT.format(
                 question=spec["question"], papers=_paper_blocks(bundle, texts, chunk))}],
            max_tokens=1800, seed=seed + ci,
            artifact_delta={"claims_added": [], "relations_added": 0})
        if obj is None:
            failures.append(f"extract_chunk_{ci}: {err}")
            continue
        for c in (obj.get("claims") or []):
            if not isinstance(c, dict) or not str(c.get("text") or "").strip():
                continue
            eids = [str(e) for e in (c.get("evidence_ids") or []) if str(e) in valid_ev]
            n_dropped_ev += len(c.get("evidence_ids") or []) - len(eids)
            cid = f"c{len(claims)}"
            claims.append({"claim_id": cid, "text": str(c["text"]).strip(),
                           "type": c.get("type") if c.get("type") in CLAIM_TYPES else "study_finding",
                           "polarity": c.get("polarity") if c.get("polarity") in POLARITY else "unknown",
                           "evidence_ids": eids,
                           "confidence": _num01(c.get("confidence"))})

    relations, unresolved, missing = [], [], []
    n_cross_batches, cross_dropped = 0, 0
    cross_batch_of, batches, n_salvaged, n_bad_rel_lines = {}, [], 0, 0
    if claims:
        def render(c):
            return f"  {c['claim_id']}: {clip(c['text'], MAXCH['claim_listed'], '...')}\n"
        pending, batches = list(claims), []
        while pending:
            grp, _ = fit(pending, render, CTX_CHARS // 2)
            if not grp:
                break
            batches.append(grp)
            pending = pending[len(grp):]
        n_cross_batches = len(batches)
        cross_batch_of = {c["claim_id"]: bi for bi, grp in enumerate(batches) for c in grp}
        known = {c["claim_id"] for c in claims}
        for bi, grp in enumerate(batches):
            listing = "".join(render(c) for c in grp)

            with Timer() as t:
                draw = client.chat(
                    [{"role": "system", "content": SYN_SYS},
                     {"role": "user", "content": SYN_REL.format(question=spec["question"],
                                                                claims=listing)}],
                    max_tokens=4000, seed=seed + 500 + bi)
            log.llm("synthesis", t.started, draw.wall_ms, draw.model, draw.prompt, draw.completion,
                    draw.params, draw.finish_reason, draw.n_prompt_tokens,
                    draw.n_completion_tokens, role_sequence=draw.role_sequence)
            if draw.error:
                failures.append(f"relations_{bi}: {draw.error}")
            else:
                rels, bad = parse_relation_lines(draw.completion, known)
                n_bad_rel_lines += bad
                relations += rels
                if draw.truncated:
                    n_salvaged += 1
                    log.failure("synthesis", utcnow(), "TruncatedRelationList",
                                f"batch {bi}: relation list cut off at the token cap after "
                                f"{len(rels)} parsed relations", recovered=True, kind="retry")

            obj, draw2, err = ask_json(
                client, log, "synthesis",
                [{"role": "system", "content": SYN_SYS},
                 {"role": "user", "content": SYN_CROSS.format(
                     question=spec["question"], claims=listing, max_new=12)}],
                max_tokens=3200, seed=seed + 900 + bi)
            if obj is None:
                failures.append(f"cross_claims_{bi}: {err}")
                continue
            by_cid = {cc["claim_id"]: cc for cc in claims}
            for c in (obj.get("claims") or [])[:12]:
                if not isinstance(c, dict) or not str(c.get("text") or "").strip():
                    continue
                eids = []
                for x in [str(y) for y in (c.get("from_claims") or [])][:5]:
                    for e in (by_cid.get(x) or {}).get("evidence_ids", []):
                        if e in valid_ev and e not in eids:
                            eids.append(e)
                for e in (c.get("evidence_ids") or []):
                    if str(e) in valid_ev and str(e) not in eids:
                        eids.append(str(e))
                cid = f"c{len(claims)}"
                claims.append({"claim_id": cid, "text": str(c["text"]).strip(),
                               "type": c.get("type") if c.get("type") in CLAIM_TYPES else "cross_study_synthesis",
                               "polarity": c.get("polarity") if c.get("polarity") in POLARITY else "unknown",
                               "evidence_ids": eids, "confidence": _num01(c.get("confidence"))})
                known.add(cid)
            unresolved += [str(x) for x in (obj.get("unresolved_conflicts") or []) if str(x) in known]
            missing += [str(x)[:300] for x in (obj.get("missing_information") or [])]
        missing = missing[:60]

    graph = seal({
        "schema_version": "synthesis_graph/1.0", "task_id": spec["task_id"],
        "evidence_bundle_hash": bundle["content_hash"], "role": "agent",
        "claims": claims, "relations": relations,
        "unresolved_conflicts": unresolved, "missing_information": missing,
        "hard_violations": [], "budget_remaining": {},
        "validation": {"n_chunks": len(chunks), "chunk_size": chunk_size,
                       "n_papers_in": len(papers),
                       "n_papers_available": len(bundle["papers"]),
                       "papers_dropped_for_context": len(bundle["papers"]) - len(papers),
                       "n_cross_batches": n_cross_batches,
                       "cross_batch_of": cross_batch_of,
                       "cross_batch_sizes": [len(g) for g in batches] if claims else [],
                       "cross_batches_salvaged_from_truncation": n_salvaged,
                       "unparsed_relation_lines": n_bad_rel_lines,
                       "claims_dropped_from_cross_pass": cross_dropped,
                       "failures": failures,
                       "dropped_evidence_refs": n_dropped_ev,
                       "note": "evidence_ids not present in the sealed bundle are DROPPED and "
                               "counted, never invented into existence"}})
    return _sealed(graph, "synthesis_graph", log, "synthesis", utcnow())


def _num01(x):
    try:
        v = float(x)
        return min(1.0, max(0.0, v))
    except (TypeError, ValueError):
        return None


PLAN_SYS = ("You are doing the STRUCTURAL PLANNING stage of a biomedical literature review. "
            "You organise claims into a section hierarchy. You do not write prose and you do "
            "not add claims that are not in the graph.")

PLAN_USER = """TOPIC: {question}
AUDIENCE: {audience}
TARGET LENGTH: {words} words total

CLAIMS AVAILABLE:
{claims}

Design the section structure of the review and allocate every claim that belongs to a section.
Use only the claim ids above. Word budgets must sum to about the target length.

Reply with ONLY one JSON object:
{{"sections": [
   {{"section_id": "s1", "parent_id": null, "title": "...", "objective": "...",
     "claim_ids": ["c0", "c3"], "word_budget": 600}}
 ]}}"""

PLAN_MAX_TOKENS = 8000
WRITE_MAX_TOKENS = 4000


def cross_claims_first(graph):
    claims = graph["claims"]
    fed = (graph.get("validation") or {}).get("cross_batch_of")
    if not isinstance(fed, dict):
        return list(claims)
    return [c for c in claims if c["claim_id"] not in fed] + [c for c in claims if c["claim_id"] in fed]


def planning(spec, graph, client, log, seed=0):
    claims = graph["claims"]

    def _render(c):
        return f"  {c['claim_id']} [{c['type']}]: {clip(c['text'], MAXCH['claim_listed'], '...')}\n"
    shown, n_dropped = fit(cross_claims_first(graph), _render)
    listing = "".join(_render(c) for c in shown)
    salvage = {"fired": False}
    obj, draw, err = ask_json(
        client, log, "planning",
        [{"role": "system", "content": PLAN_SYS},
         {"role": "user", "content": PLAN_USER.format(
             question=spec["question"], audience=spec["audience"],
             words=spec["output_spec"]["target_words"], claims=listing or "  (none)")}],
        max_tokens=PLAN_MAX_TOKENS, seed=seed,
        salvage_keys=("sections",), salvage=salvage)

    known = {c["claim_id"] for c in claims}
    sections, used, failures = [], set(), []
    if obj is None:
        failures.append(f"plan: {err}")
    else:
        for i, s in enumerate(obj.get("sections") or []):
            if not isinstance(s, dict) or not str(s.get("title") or "").strip():
                continue
            sid = str(s.get("section_id") or f"s{i+1}")
            cids = [str(c) for c in (s.get("claim_ids") or []) if str(c) in known]
            used.update(cids)
            wb = s.get("word_budget")
            sections.append({"section_id": sid, "parent_id": s.get("parent_id") or None,
                             "title": str(s["title"]).strip(),
                             "objective": (str(s.get("objective")) if s.get("objective") else None),
                             "claim_ids": cids, "evidence_ids": [],
                             "word_budget": int(wb) if isinstance(wb, (int, float)) else None,
                             "children": []})
    seen_ids, dedup = set(), []
    for s in sections:
        if s["section_id"] in seen_ids:
            s["section_id"] = f"{s['section_id']}_{len(dedup)}"
        seen_ids.add(s["section_id"])
        dedup.append(s)
    sections = dedup
    dupes = [c for c in known if sum(c in s["claim_ids"] for s in sections) > 1]

    plan = seal({
        "schema_version": "outline_plan/1.0", "task_id": spec["task_id"],
        "synthesis_graph_hash": graph["content_hash"], "sections": sections,
        "precedence_edges": [[sections[i]["section_id"], sections[i + 1]["section_id"]]
                             for i in range(len(sections) - 1)],
        "coverage_map": {"n_claims": len(known), "n_assigned": len(used),
                         "coverage": round(len(used) / len(known), 4) if known else None},
        "unassigned_claims": sorted(known - used), "duplicated_claims": sorted(dupes),
        "budget_remaining": {},
        "validation": {"failures": failures,
                       "word_budget_sum": sum(s["word_budget"] or 0 for s in sections),
                       "target_words": spec["output_spec"]["target_words"],
                       "claims_shown": len(shown), "claims_available": len(claims),
                       "claims_dropped_for_context": n_dropped,
                       "max_tokens_first_attempt": PLAN_MAX_TOKENS,
                       "finish_reason": draw.finish_reason if draw else None,
                       "salvaged_from_truncation": salvage}})
    return _sealed(plan, "outline_plan", log, "planning", utcnow())


WRITE_SYS = ("You are doing the GROUNDED WRITING stage of a biomedical literature review. You "
             "write only from the claims and evidence given to you. You cannot search. Every "
             "sentence that states a finding must carry the claim ids it realises and the "
             "papers it cites.")

WRITE_USER = """TOPIC: {question}
AUDIENCE: {audience}
SECTION: {title}
OBJECTIVE: {objective}
LENGTH: about {words} words

CLAIMS TO REALISE IN THIS SECTION:
{claims}

SUPPORTING EVIDENCE:
{evidence}

Write this section. Output it as sentences so the grounding stays explicit.

Reply with ONLY one JSON object:
{{"sentences": [
   {{"text": "<one sentence of the review>",
     "claim_ids": ["c3"],
     "citations": ["<PMID>", ...]}}
]}}
A sentence that is pure connective tissue may have empty claim_ids and citations. A sentence
that states a finding must not."""


BUDGET_FLOOR, BUDGET_CAP = 250, 900
BUDGET_RULE = ("target_words * (1 + n_claims_in_section) / sum(1 + n_claims) over all plan "
               f"sections, floor {BUDGET_FLOOR}, cap {BUDGET_CAP}; a positive word_budget "
               "carried by the plan itself takes precedence")


def section_word_budgets(spec, plan):
    secs = plan["sections"]
    target = int(spec["output_spec"]["target_words"])
    weights = {s["section_id"]: 1 + len(s.get("claim_ids") or []) for s in secs}
    wsum = sum(weights.values()) or 1
    budgets, source = {}, {}
    for s in secs:
        wb = s.get("word_budget")
        if isinstance(wb, (int, float)) and not isinstance(wb, bool) and wb > 0:
            budgets[s["section_id"]], source[s["section_id"]] = int(wb), "plan"
        else:
            raw = target * weights[s["section_id"]] / wsum
            budgets[s["section_id"]] = int(min(BUDGET_CAP, max(BUDGET_FLOOR, round(raw))))
            source[s["section_id"]] = "rule"
    return budgets, source


def _nonempty_sentences(obj):
    out = []
    for sent in ((obj or {}).get("sentences") or []):
        if not isinstance(sent, dict):
            continue
        if str(sent.get("text") or "").strip():
            out.append(sent)
    return out
