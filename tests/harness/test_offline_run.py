"""Runs SCRIBE offline with a mock model on a synthetic fixed-input task and checks the artifacts,
manifests, traces, receipt chain and lever prompt hash."""

from __future__ import annotations

import json

import pytest

from harness_env import ROOT, TASK, R, MockClient

SCHEMAS = ROOT / "biolitbench" / "schemas"


def _validator(name):
    import jsonschema
    return jsonschema.Draft202012Validator(json.loads((SCHEMAS / f"{name}.schema.json").read_text()))


def _errs(name, obj):
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in _validator(name).iter_errors(obj)]


def test_repo_schemas_are_the_ones_the_runner_uses():
    from common import SCHEMAS as RUNNER_SCHEMAS
    assert RUNNER_SCHEMAS.resolve() == SCHEMAS.resolve()
    for name in ("call_log", "evidence_bundle", "outline_plan", "report_artifact", "run_manifest",
                 "synthesis_graph", "task_spec"):
        assert (SCHEMAS / f"{name}.schema.json").is_file(), name


@pytest.mark.parametrize("relations", [True, False], ids=["mock_relations", "mock_no_relations"])
def test_offline_run_fixed_input(relations, levers_installed, task_runs, tmp_path):
    from common import content_hash
    out_root = tmp_path
    c = MockClient(relations=relations)
    entry = R.load_canonical(TASK, "evidence_bundle")
    out, mans = R.run_generation(TASK, "SCRIBE", 0, c, level=7, mode="bundle_entry", entry_bundle=entry,
                                 entry_mode="bundle_entry")
    d = R.run_dir(7, "SCRIBE", "bundle_entry", TASK, 0)
    assert d.is_relative_to(task_runs)

    files = {"evidence_bundle": "entry_evidence_bundle.json", "synthesis_graph": "synthesis_graph.json",
             "outline_plan": "outline_plan.json", "report_artifact": "report_artifact.json"}
    art = {}
    for schema, fn in files.items():
        p = d / fn
        assert p.is_file(), fn
        art[schema] = json.loads(p.read_text())
        assert _errs(schema, art[schema]) == [], (fn, _errs(schema, art[schema])[:5])
        assert art[schema]["content_hash"] == content_hash(art[schema]), f"{fn}: seal broken"
    assert art["evidence_bundle"]["content_hash"] == entry["content_hash"]
    lines = [json.loads(l) for l in (d / "manifests.jsonl").read_text().splitlines()]
    assert lines == mans and [m["window"] for m in mans] == ["synthesis", "planning", "writing"]
    for m in mans:
        assert _errs("run_manifest", m) == [] and "_manifest_validation_errors" not in m
        assert m["status"] == "ok" and m["system"] == "SCRIBE" and m["retrieval_check"]["findings"] == []
        assert m["prompt_hash"] == levers_installed["prompt_hash"]
    trace = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines()]
    bad = [(i, _errs("call_log", e)[:2]) for i, e in enumerate(trace) if _errs("call_log", e)]
    assert not bad, bad[:3]

    g = art["synthesis_graph"]
    n_chunks = -(-len(entry["papers"]) // 8)
    assert c.calls["extract"] == n_chunks and g["validation"]["n_chunks"] == n_chunks
    assert c.calls["relations"] == c.calls["cross"] == g["validation"]["n_cross_batches"]
    if relations:
        assert len(g["relations"]) > 0
        known = {x["claim_id"] for x in g["claims"]}
        assert all(r["source"] in known and r["target"] in known for r in g["relations"])
    else:
        assert g["relations"] == []
    assert any(x["type"] != "study_finding" for x in g["claims"])

    hb, hg, hp, hr = (art[k]["content_hash"] for k in ("evidence_bundle", "synthesis_graph", "outline_plan",
                                                        "report_artifact"))
    syn, plan, wri = mans
    assert (syn["entry_hash"], syn["exit_hash"]) == (hb, hg)
    assert (plan["entry_hash"], plan["exit_hash"]) == (hg, hp)
    assert (wri["entry_hash"], wri["exit_hash"]) == (hp, hr)
    assert g["evidence_bundle_hash"] == hb
    assert art["outline_plan"]["synthesis_graph_hash"] == hg
    assert art["report_artifact"]["outline_plan_hash"] == hp
    integ = json.loads((d / "integrity.json").read_text())
    assert integ["n_findings"] == 0, integ
    delivery = json.loads((d / "delivery.json").read_text())
    assert delivery["ok"] is True

    ta = art["report_artifact"]["terminal_audit"]
    assert ta["writing_levers"]["levers"] == levers_installed["levers"]
    assert any("\n\n" in s["text"] for s in art["report_artifact"]["sections"])

    summary = {"task": TASK, "run_dir": str(d), "mock_relations": relations, "calls_by_kind": c.calls,
               "n_papers": len(entry["papers"]), "n_chunks": n_chunks, "n_cross_batches": g["validation"]["n_cross_batches"],
               "n_claims": len(g["claims"]), "n_cross_claims": sum(x["type"] != "study_finding" for x in g["claims"]),
               "n_relations": len(g["relations"]), "n_sections": len(art["outline_plan"]["sections"]),
               "n_words": ta["n_words"], "prompt_hash": syn["prompt_hash"],
               "receipt_chain": [hb, hg, hp, hr], "integrity_findings": integ["n_findings"],
               "schemas": "biolitbench/schemas", "checks": "all passed"}
    (out_root / "offline_run_summary.json").write_text(json.dumps(summary, indent=1))
    print("\nOFFLINE RUN", json.dumps({k: summary[k] for k in ("calls_by_kind", "n_claims", "n_cross_claims", "n_relations",
                                                          "n_sections", "n_words")}))
