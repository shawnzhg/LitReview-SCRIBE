"""Tests that every referential-integrity check of the artifact chain fires on a corrupted chain and
that a clean chain has no findings."""

from __future__ import annotations

import harness_env

from common import seal
from integrity import check_bundle, check_chain, check_graph, check_plan, check_report


def test_every_integrity_check_fires_on_a_corrupted_chain():
    spec = seal({"content_hash": "", "publication_cutoff": 2020, "task_id": "t"})
    bundle = seal({"schema_version": "evidence_bundle/1.0", "task_id": "t",
                   "task_spec_hash": spec["content_hash"], "provenance_tier": "agent_retrieved",
                   "retrieval_status": "done",
                   "papers": [{"paper_id": "1", "rank": 1, "decision": "include", "year": 2019}],
                   "evidence": [{"evidence_id": "e1", "paper_id": "1",
                                 "granularity": "abstract", "locator": "pmid1",
                                 "text_hash": "sha256:" + "0" * 64}],
                   "budget_remaining": {}, "validation": {}})
    graph = seal({"schema_version": "synthesis_graph/1.0", "task_id": "t",
                  "evidence_bundle_hash": bundle["content_hash"],
                  "claims": [{"claim_id": "c0", "text": "x", "type": "study_finding",
                              "evidence_ids": ["e1"]}],
                  "relations": [], "budget_remaining": {}, "validation": {}})
    plan = seal({"schema_version": "outline_plan/1.0", "task_id": "t",
                 "synthesis_graph_hash": graph["content_hash"],
                 "sections": [{"section_id": "s1", "title": "T", "claim_ids": ["c0"]}],
                 "budget_remaining": {}, "validation": {}})
    report = seal({"schema_version": "report_artifact/1.0", "task_id": "t",
                   "outline_plan_hash": plan["content_hash"],
                   "sections": [{"section_id": "s1", "text": "hello"}],
                   "sentence_claim_map": [{"sentence_id": "s1#0", "claim_ids": ["c0"]}],
                   "citation_evidence_map": [{"sentence_id": "s1#0", "paper_id": "1",
                                              "evidence_ids": ["e1"]}],
                   "bibliography": ["1"], "terminal_audit": {}, "resource_summary": {}})
    base = check_chain(spec, bundle, graph, plan, report)

    def mut(obj, fn):
        import copy
        o = copy.deepcopy(obj)
        fn(o)
        return o

    probes = [
        ("bundle.duplicate_paper_id",
         lambda: check_bundle(mut(bundle, lambda b: b["papers"].append(dict(b["papers"][0], rank=2))), spec)),
        ("bundle.evidence_orphan_paper",
         lambda: check_bundle(mut(bundle, lambda b: b["evidence"].append(dict(b["evidence"][0], evidence_id="e9", paper_id="999"))), spec)),
        ("bundle.agent_bundle_lost_its_spec",
         lambda: check_bundle(mut(bundle, lambda b: b.update(task_spec_hash=None)), spec)),
        ("bundle.post_cutoff_paper",
         lambda: check_bundle(mut(bundle, lambda b: b["papers"][0].update(year=2099)), spec)),
        ("bundle.hash_stale",
         lambda: check_bundle(mut(bundle, lambda b: b["papers"][0].update(rank=1, title="changed")), spec)),
        ("graph.dangling_evidence_id",
         lambda: check_graph(mut(graph, lambda g: g["claims"][0]["evidence_ids"].append("e_MISSING")), bundle)),
        ("graph.relation_unknown_claim",
         lambda: check_graph(mut(graph, lambda g: g["relations"].append({"source": "c0", "target": "cX", "type": "supports"})), bundle)),
        ("graph.bundle_hash_mismatch",
         lambda: check_graph(mut(graph, lambda g: g.update(evidence_bundle_hash="sha256:" + "1" * 64)), bundle)),
        ("plan.unknown_claim_id",
         lambda: check_plan(mut(plan, lambda p: p["sections"][0]["claim_ids"].append("cZ")), graph)),
        ("plan.parent_not_a_section",
         lambda: check_plan(mut(plan, lambda p: p["sections"][0].update(parent_id="sNOPE")), graph)),
        ("report.unknown_evidence_in_map",
         lambda: check_report(mut(report, lambda r: r["citation_evidence_map"][0]["evidence_ids"].append("eNOPE")), plan, graph, bundle)),
        ("report.sentence_id_orphan_section",
         lambda: check_report(mut(report, lambda r: r["sentence_claim_map"].append({"sentence_id": "sZZ#0", "claim_ids": []})), plan, graph, bundle)),
        ("report.section_not_in_plan",
         lambda: check_report(mut(report, lambda r: r["sections"].append({"section_id": "sX", "text": "y"})), plan, graph, bundle)),
    ]
    assert base["n_findings"] == 0
    missed = [code for code, fn in probes if code not in [f["code"] for f in fn()]]
    assert missed == []
