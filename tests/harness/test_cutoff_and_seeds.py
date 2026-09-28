"""Tests the strictly-before cutoff in the harness checks, the retrieval client and the selection
rule's windows, and the seeds 0, 1 and 2 of the campaign runner and the same-pool driver."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from harness_env import ROOT, TASK, R

sys.path.insert(0, str(ROOT / "scribe" / "retrieval"))
sys.path.insert(0, str(ROOT / "scribe" / "harness" / "tools"))


def test_bundle_delivery_and_integrity_refuse_a_paper_of_the_cutoff_year(task_runs):
    from common import seal
    from integrity import check_bundle
    spec = R.load_spec(TASK)
    bundle = R.load_canonical(TASK, "evidence_bundle")
    texts = R.texts_from_canonical(TASK)
    errors, summary = R.check_bundle_delivery(TASK, spec, bundle, texts, split="dev")
    assert errors == [] and summary["cutoff_ok"] is True
    late = copy.deepcopy(bundle)
    late.pop("content_hash")
    late["papers"][0]["year"] = spec["publication_cutoff"]
    late = seal(late)
    errors, summary = R.check_bundle_delivery(TASK, spec, late, texts, split="dev")
    assert summary["cutoff_ok"] is False and any("year >= cutoff" in e for e in errors)
    assert "bundle.post_cutoff_paper" in [f["code"] for f in check_bundle(late, spec)]
    early = copy.deepcopy(late)
    early.pop("content_hash")
    early["papers"][0]["year"] = spec["publication_cutoff"] - 1
    assert "bundle.post_cutoff_paper" not in [f["code"] for f in check_bundle(seal(early), spec)]


@pytest.fixture
def pool_tool(tmp_path):
    import pool_retrieval_tool as PRT
    cuts = tmp_path / "cutoffs.json"
    cuts.write_text(json.dumps({"t1": 2020}))
    cfg = PRT.PoolConfig(url="http://127.0.0.1:9", cutoffs_path=str(cuts), index_dir=str(tmp_path))
    tool = PRT.PoolRetrievalTool(PRT.POOL_SNAPSHOT_ID, 2020, {"max_search_calls": 5, "max_document_opens": 5},
                                 task_id="t1", cfg=cfg, calls_path=tmp_path / "calls.jsonl")
    yield PRT, tool
    sys.modules.pop("pool_retrieval_tool", None)


@pytest.mark.parametrize("year,ok", [(2019, True), (2020, False), (2021, False), (None, False)])
def test_retrieval_client_accepts_only_papers_dated_before_the_cutoff(pool_tool, monkeypatch, year, ok):
    PRT, tool = pool_tool
    hit = {"pmid": "11", "score": 3.0, "year": year, "title": "t", "abstract": "a"}
    monkeypatch.setattr(PRT, "http_get", lambda url, **k: (200, [hit], 1.0, 1))
    if ok:
        assert [r["paper_id"] for r in tool.search("q", k=5)] == ["11"]
    else:
        with pytest.raises(PRT.PoolContractViolation, match="not before the cutoff"):
            tool.search("q", k=5)
    monkeypatch.setattr(PRT, "http_get", lambda url, **k: (200, {"title": "t", "abstract": "a", "year": year}, 1.0, 1))
    if ok:
        assert tool.open_document("11")["year"] == year
    else:
        with pytest.raises(PRT.PoolContractViolation, match="not before the cutoff"):
            tool.open_document("11")


class _Store:

    def __init__(self, years, citers):
        self.years, self.citers = years, citers

    def counts(self, ids, cut, excl):
        st = np.array([self.citers[i] for i in ids], dtype=np.int64)
        yr = np.array([self.years[i] for i in ids], dtype=np.int16)
        return st, np.zeros(len(ids), np.int64), yr

    def provenance(self):
        return {"edges_meta_sha256": "synthetic"}


def test_selection_windows_are_the_two_years_before_the_cutoff():
    import selection_rule as SR
    years = {"a": 2019, "b": 2018, "c": 2017, "d": 2016, "e": 2019, "f": 2015}
    citers = {"a": 1, "b": 9, "c": 5, "d": 2, "e": 0, "f": 7}
    P, rec = SR.select_papers(list(years), 3, 2020, store=_Store(years, citers))
    wins = {w["window"]: w["years"] for w in rec["windows"]}
    assert wins == {0: [2018, 2019], 1: [2016, 2017], 2: [2014, 2015]}
    assert {g["id"]: g["window"] for g in rec["gate_papers"]} == {"a": 0, "b": 0, "e": 0, "c": 1, "d": 1, "f": 2}
    assert rec["citer_year_rule"] == "< cutoff" and P[0] == "b"


def test_campaign_runner_accepts_seeds_zero_one_two_only(task_runs, monkeypatch):
    import run_campaign as RC
    monkeypatch.setattr(RC, "RUNS", task_runs)
    monkeypatch.setattr(RC, "IDX", task_runs / "_index")
    for seeds, ok in (("0", True), ("1", True), ("0,1,2", True), ("3", False), ("0,0", False)):
        monkeypatch.setattr(sys, "argv", ["run_campaign.py", "--phase", "generation", "--mode", "bundle_entry",
                                          "--level", "7", "--seeds", seeds, "--tasks", TASK])
        monkeypatch.delenv("SCRIBE_RENDEZVOUS", raising=False)
        with pytest.raises(SystemExit) as e:
            RC.main()
        assert ("SCRIBE_RENDEZVOUS" in str(e.value)) == ok, (seeds, e.value)


def test_same_pool_driver_checks_the_units_of_the_requested_seed(task_runs, tmp_path):
    for seed in (1, 2):
        d = R.run_dir(9, "SCRIBE", "native_chain", TASK, seed)
        d.mkdir(parents=True)
        (d / "evidence_bundle.json").write_text("{}")
    tf = tmp_path / "tasks.txt"
    tf.write_text(TASK + "\n")
    code = (f"import runpy, sys; sys.path.insert(0, {str(ROOT / 'tests' / 'harness')!r}); import harness_env; "
            f"from pathlib import Path; import runner as R; R.RUNS = Path({str(task_runs)!r}); "
            f"sys.argv = ['generate.py'] + sys.argv[1:]; "
            f"runpy.run_path({str(ROOT / 'scribe' / 'launchers' / 'generate.py')!r}, run_name='__main__')")
    env = {"PATH": "/usr/bin:/bin", "SCRIBE_LEVERS": str(ROOT / "scribe" / "levers" / "writing_levers.json"),
           "SCRIBE_DRIVER_RECORD": str(tmp_path / "driver.json"), "PYTHONDONTWRITEBYTECODE": "1"}
    args = ["--phase", "generation", "--systems", "SCRIBE", "--mode", "native_chain", "--level", "9", "--tasks-file", str(tf)]
    r0 = subprocess.run([sys.executable, "-c", code, *args, "--seeds", "0"], env=env, capture_output=True, text=True)
    assert r0.returncode != 0 and "1 units without a bundle" in r0.stderr + r0.stdout
    r1 = subprocess.run([sys.executable, "-c", code, *args, "--seeds", "1,2"], env=env, capture_output=True, text=True)
    assert "units without a bundle" not in r1.stderr + r1.stdout and "2 units selected" in r1.stdout, r1.stderr[-800:]
