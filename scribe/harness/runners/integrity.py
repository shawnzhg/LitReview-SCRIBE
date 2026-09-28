#!/usr/bin/env python3
"""Referential-integrity checks across the artifact chain (evidence bundle, synthesis graph,
outline, report) that JSON Schema cannot express; they report findings and repair nothing."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import content_hash


def _f(code, detail, n=1):
    return {"code": code, "detail": detail, "n": n}


def check_bundle(bundle, spec=None):
    out = []
    pids = [p["paper_id"] for p in bundle["papers"]]
    if len(pids) != len(set(pids)):
        out.append(_f("bundle.duplicate_paper_id", "same paper appears twice in the ranked set",
                      len(pids) - len(set(pids))))
    ranks = sorted(p["rank"] for p in bundle["papers"])
    if ranks and ranks != list(range(1, len(ranks) + 1)):
        out.append(_f("bundle.rank_not_contiguous", f"ranks are {ranks[:5]}...; expected 1..n"))
    eids = [e["evidence_id"] for e in bundle["evidence"]]
    if len(eids) != len(set(eids)):
        out.append(_f("bundle.duplicate_evidence_id", "evidence ids collide",
                      len(eids) - len(set(eids))))
    known = set(pids)
    orphan = [e["evidence_id"] for e in bundle["evidence"] if e["paper_id"] not in known]
    if orphan:
        out.append(_f("bundle.evidence_orphan_paper",
                      f"evidence whose paper is not in papers[]: {orphan[:5]}", len(orphan)))
    if bundle["provenance_tier"] == "agent_retrieved" and not bundle.get("task_spec_hash"):
        out.append(_f("bundle.agent_bundle_lost_its_spec",
                      "agent_retrieved bundle with null task_spec_hash: it was produced under "
                      "some spec and lost it"))
    if spec is not None:
        if bundle.get("task_spec_hash") and bundle["task_spec_hash"] != spec["content_hash"]:
            out.append(_f("bundle.spec_hash_mismatch",
                          "bundle is bound to a different TaskSpec than the one supplied"))
        cut = spec["publication_cutoff"]
        late = [p["paper_id"] for p in bundle["papers"]
                if p.get("year") is not None and int(p["year"]) >= cut]
        if late:
            out.append(_f("bundle.post_cutoff_paper",
                          f"papers not published before the cutoff {cut}: {late[:5]}", len(late)))
    if bundle.get("content_hash") != content_hash(bundle):
        out.append(_f("bundle.hash_stale", "content_hash does not match the body"))
    return out


def check_graph(graph, bundle):
    out = []
    cids = [c["claim_id"] for c in graph["claims"]]
    if len(cids) != len(set(cids)):
        out.append(_f("graph.duplicate_claim_id", "claim ids collide", len(cids) - len(set(cids))))
    if graph.get("evidence_bundle_hash") != bundle.get("content_hash"):
        out.append(_f("graph.bundle_hash_mismatch",
                      "graph was built from a different bundle than the one supplied"))
    valid = {e["evidence_id"] for e in bundle["evidence"]}
    dangling, total = [], 0
    for c in graph["claims"]:
        for e in c.get("evidence_ids") or []:
            total += 1
            if e not in valid:
                dangling.append((c["claim_id"], e))
    if dangling:
        out.append(_f("graph.dangling_evidence_id",
                      f"claim citations pointing at nothing ({len(dangling)}/{total}): "
                      f"{dangling[:5]}", len(dangling)))
    known = set(cids)
    bad = [(r["source"], r["target"]) for r in graph["relations"]
           if r["source"] not in known or r["target"] not in known]
    if bad:
        out.append(_f("graph.relation_unknown_claim",
                      f"relations referencing unknown claims: {bad[:5]}", len(bad)))
    self_r = [r for r in graph["relations"] if r["source"] == r["target"]]
    if self_r:
        out.append(_f("graph.self_relation", "a claim related to itself", len(self_r)))
    if graph.get("content_hash") != content_hash(graph):
        out.append(_f("graph.hash_stale", "content_hash does not match the body"))
    return out


def check_plan(plan, graph):
    out = []
    if plan.get("synthesis_graph_hash") != graph.get("content_hash"):
        out.append(_f("plan.graph_hash_mismatch",
                      "plan was built from a different graph than the one supplied"))
    sids = [s["section_id"] for s in plan["sections"]]
    if len(sids) != len(set(sids)):
        out.append(_f("plan.duplicate_section_id", "section ids collide", len(sids) - len(set(sids))))
    known = {c["claim_id"] for c in graph["claims"]}
    bad = [c for s in plan["sections"] for c in s["claim_ids"] if c not in known]
    if bad:
        out.append(_f("plan.unknown_claim_id",
                      f"sections allocate claims that are not in the graph: {bad[:5]}", len(bad)))
    parents = {s.get("parent_id") for s in plan["sections"] if s.get("parent_id")}
    orphan = [p for p in parents if p not in set(sids)]
    if orphan:
        out.append(_f("plan.parent_not_a_section", f"parent_id pointing at nothing: {orphan[:5]}",
                      len(orphan)))
    if plan.get("content_hash") != content_hash(plan):
        out.append(_f("plan.hash_stale", "content_hash does not match the body"))
    return out


def check_report(report, plan, graph, bundle):
    out = []
    if report.get("outline_plan_hash") != plan.get("content_hash"):
        out.append(_f("report.plan_hash_mismatch",
                      "report was written from a different plan than the one supplied"))
    sec_ids = {s["section_id"] for s in report["sections"]}
    plan_ids = {s["section_id"] for s in plan["sections"]}
    extra = sec_ids - plan_ids
    if extra:
        out.append(_f("report.section_not_in_plan",
                      f"report contains sections the plan does not: {sorted(extra)[:5]}", len(extra)))
    known_c = {c["claim_id"] for c in graph["claims"]}
    bad = [c for m in report["sentence_claim_map"] for c in m.get("claim_ids", []) if c not in known_c]
    if bad:
        out.append(_f("report.unknown_claim_in_map",
                      f"sentence_claim_map cites claims not in the graph: {bad[:5]}", len(bad)))
    valid_e = {e["evidence_id"] for e in bundle["evidence"]}
    bade = [e for m in report["citation_evidence_map"] for e in m.get("evidence_ids", [])
            if e not in valid_e]
    if bade:
        out.append(_f("report.unknown_evidence_in_map",
                      f"citation_evidence_map cites evidence not in the bundle: {bade[:5]}",
                      len(bade)))
    sent_ids = {m["sentence_id"] for m in report["sentence_claim_map"]}
    sent_ids |= {m["sentence_id"] for m in report["citation_evidence_map"]}
    bad_prefix = [s for s in sent_ids if s.split("#")[0] not in sec_ids]
    if bad_prefix:
        out.append(_f("report.sentence_id_orphan_section",
                      f"sentence ids whose section does not exist: {bad_prefix[:5]}",
                      len(bad_prefix)))
    if report.get("content_hash") != content_hash(report):
        out.append(_f("report.hash_stale", "content_hash does not match the body"))
    return out


def check_chain(spec, bundle, graph=None, plan=None, report=None):
    f = {"bundle": check_bundle(bundle, spec)}
    if graph is not None:
        f["graph"] = check_graph(graph, bundle)
    if plan is not None and graph is not None:
        f["plan"] = check_plan(plan, graph)
    if report is not None and plan is not None and graph is not None:
        f["report"] = check_report(report, plan, graph, bundle)
    f["n_findings"] = sum(len(v) for k, v in f.items() if isinstance(v, list))
    return f
