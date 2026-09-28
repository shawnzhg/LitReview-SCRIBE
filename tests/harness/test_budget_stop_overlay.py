"""Tests that only a document-read budget stop counts as a normal acquisition exit in the scoring
overlay."""

from __future__ import annotations

import importlib.util
import json

import pytest

from harness_env import ROOT

OV_PATH = ROOT / "evaluation" / "board" / "budget_stop_overlay.py"


@pytest.fixture(scope="module")
def OV():
    spec = importlib.util.spec_from_file_location("budget_stop_overlay_test", OV_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _unit(tmp, t, *, opens_left, search_left, docs_read, docs_cap, system="SCRIBE"):
    from common import seal
    u = tmp / "level5" / system / "native_chain" / t / "seed0"
    u.mkdir(parents=True)
    b = seal({"task_id": t, "retrieval_status": "budget_exhausted", "papers": [{"paper_id": "1"}, {"paper_id": "2"}],
              "budget_remaining": {"document_opens": opens_left, "search_calls": search_left}})
    (u / "evidence_bundle.json").write_text(json.dumps(b))
    hs = [b["content_hash"], "sha256:g", "sha256:p", "sha256:r"]
    mans = [{"window": "acquisition", "status": "budget_exhausted", "exit_hash": hs[0]}]
    for w, (e, x) in zip(("synthesis", "planning", "writing"), zip(hs[:-1], hs[1:])):
        mans.append({"window": w, "status": "ok", "entry_hash": e, "exit_hash": x})
    (u / "manifests.jsonl").write_text("".join(json.dumps(m) + "\n" for m in mans))
    audits = [("acquisition_audit.json", {"seeds": "0", "units": {t: {"n_findings": 0, "n_papers": 2, "docs_read": docs_read,
                                                                    "effective_budget": {"max_docs_read": docs_cap}}}})]
    return u, audits


@pytest.mark.parametrize("case,opens,search,docs,verdict", [
    ("docs_read_stop", 3, 2, 100, "changed"),
    ("docs_read_and_opens_stop", 0, 2, 100, "changed"),
    ("opens_only_stop", 0, 2, 40, "refused"),
    ("search_only_stop", 3, 0, 40, "refused"),
    ("no_stop", 3, 2, 40, "refused"),
])
def test_condition_e_docs_read_only(OV, case, opens, search, docs, verdict, tmp_path):
    tmp = tmp_path / case
    u, audits = _unit(tmp, "pmcid_TEST1", opens_left=opens, search_left=search, docs_read=docs, docs_cap=100)
    row, data = OV.decide(str(u), 5, "pmcid_TEST1", "0", True, audits)
    assert row["verdict"] == verdict, row["reason"]
    if verdict == "refused":
        assert row["reason"].startswith("(e) not a document-budget stop")
    else:
        new = [json.loads(l) for l in data.decode().splitlines()]
        assert new[0]["status"] == "ok" and "docs_read" in new[0]["budget_stop_overlay"]["budget_stop"]


def test_unit_path_system_is_the_units_own(OV, tmp_path):
    t = "pmcid_TEST2"
    tmp = tmp_path / "rows_path"
    u, audits = _unit(tmp, t, opens_left=3, search_left=2, docs_read=100, docs_cap=100)
    audits.append(("ranking_budget_audit.json", {"seeds": "0", "rows": [{"task": t, "unit": str(u), "findings": []}]}))
    row, _ = OV.decide(str(u), 5, t, "0", True, audits)
    assert row["verdict"] == "changed", row["reason"]
