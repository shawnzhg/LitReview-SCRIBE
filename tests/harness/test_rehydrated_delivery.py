"""Tests the delivery check on rehydrated data: text drift is accepted only for the units the rehydration report lists, the drift of other kinds is read separately, and dehydrated files are refused."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json

import pytest

from harness_env import ROOT, TASK, R, MockClient
from common import DehydratedDataError, sha256_str

WRAPPER = {"format": "biolit-dehydrated/1", "kind": "texts_test", "task_id": TASK, "path": "x", "data": {}}


def write_report(runs, drift, schema="biolit-rehydrate-report/1", root=None, tsv=b"path\tkind\n", manifest=None):
    base = runs.parent
    (base / "manifests").mkdir(exist_ok=True)
    (base / "manifests" / "original_sha256.tsv").write_bytes(tsv)
    rep = {"schema": schema, "root": str(root or base),
           "manifest_sha256": manifest or hashlib.sha256(tsv).hexdigest(), "drift": drift}
    p = base / "rehydrate_report.json"
    p.write_text(json.dumps(rep, indent=1))
    return p


def unit(kind, locator, granularity, expected, written):
    return {"task_id": TASK, "path": "x", "kind": kind, "pmid": "1", "locator": locator, "granularity": granularity,
            "status": "drift" if written else "missing", "expected_text_hash": expected, "written_text_hash": written}


def drifted(bundle, texts):
    hashes = {e["locator"]: e["text_hash"] for e in bundle["evidence"]}
    p1, p2 = bundle["papers"][0]["paper_id"], bundle["papers"][1]["paper_id"]
    loc = f"pmid{p1}_1"
    new = copy.deepcopy(texts)
    new[p1]["sentences"][loc] = "Study 0 measured outcome 0, revised today."
    new[p2]["abstract"] = None
    drift = [unit("texts_test", loc, "abstract_sentence", hashes[loc], sha256_str(new[p1]["sentences"][loc])),
             unit("texts_test", f"pmid{p2}", "abstract", hashes[f"pmid{p2}"], None)]
    return new, drift, sorted([loc, f"pmid{p2}"]), p1


def test_delivery_accepts_drift_only_for_the_units_the_report_lists(task_runs, monkeypatch):
    monkeypatch.setattr(R, "_DRIFT_CACHE", {})
    spec, bundle, texts = R.load_spec(TASK), R.load_canonical(TASK, "evidence_bundle"), R.texts_from_canonical(TASK)
    new, drift, locs, p1 = drifted(bundle, texts)
    monkeypatch.delenv("SCRIBE_REHYDRATE_REPORT", raising=False)
    errors, s = R.check_bundle_delivery(TASK, spec, bundle, new, split="dev")
    assert s["n_text_hash_mismatch"] == 2 and s["n_text_drift"] == 0 and s["rehydrate_report_sha256"] is None
    assert any("text_hash" in e for e in errors)
    path = write_report(task_runs, drift + [unit("oracle_sentences", f"pmid{p1}_2", "abstract_sentence", "sha256:0", "sha256:1")])
    monkeypatch.setenv("SCRIBE_REHYDRATE_REPORT", str(path))
    errors, s = R.check_bundle_delivery(TASK, spec, bundle, new, split="dev")
    assert errors == [] and s["ok"] and s["n_text_hash_mismatch"] == 0
    assert s["n_text_drift"] == 2 and s["text_drift_locators"] == locs
    assert s["rehydrate_report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    fields = R.delivery_fields(s)
    assert fields["n_text_drift"] == 2 and fields["rehydrate_report_sha256"] == s["rehydrate_report_sha256"]
    allow, sha = R.text_drift_allowance(("oracle_sentences",))
    assert list(allow) == [(TASK, f"pmid{p1}_2", "abstract_sentence", "sha256:0")] and sha == s["rehydrate_report_sha256"]
    assert len(R.text_drift_allowance()[0]) == 2
    errors, s = R.check_bundle_delivery(TASK, spec, bundle, texts, split="dev")
    assert errors == [] and s["n_text_drift"] == 0
    other = copy.deepcopy(new)
    other[p1]["sentences"][f"pmid{p1}_2"] = "Study 0 measured outcome 1, edited."
    errors, s = R.check_bundle_delivery(TASK, spec, bundle, other, split="dev")
    assert s["n_text_hash_mismatch"] == 1 and s["n_text_drift"] == 2 and not s["ok"]
    wrong = copy.deepcopy(new)
    wrong[p1]["sentences"][locs[0]] = "Study 0 measured outcome 0, revised twice."
    errors, s = R.check_bundle_delivery(TASK, spec, bundle, wrong, split="dev")
    assert s["n_text_hash_mismatch"] == 1 and s["n_text_drift"] == 1


@pytest.mark.parametrize("case", ["schema", "root", "manifest", "json"])
def test_an_invalid_report_is_an_error_and_an_absent_one_is_strict(task_runs, monkeypatch, tmp_path, case):
    monkeypatch.setattr(R, "_DRIFT_CACHE", {})
    spec, bundle, texts = R.load_spec(TASK), R.load_canonical(TASK, "evidence_bundle"), R.texts_from_canonical(TASK)
    monkeypatch.setenv("SCRIBE_REHYDRATE_REPORT", str(tmp_path / "absent.json"))
    errors, s = R.check_bundle_delivery(TASK, spec, bundle, texts, split="dev")
    assert errors == [] and s["rehydrate_report_sha256"] is None
    kw = {"schema": {"schema": "other/1"}, "root": {"root": tmp_path / "elsewhere"},
          "manifest": {"manifest": "0" * 64}, "json": {}}[case]
    path = write_report(task_runs, [], **kw)
    if case == "json":
        path.write_text("{not json")
    monkeypatch.setenv("SCRIBE_REHYDRATE_REPORT", str(path))
    with pytest.raises(R.RehydrateReportError):
        R.check_bundle_delivery(TASK, spec, bundle, texts, split="dev")


def test_a_bundle_entry_run_records_the_drift_it_accepted(levers_installed, task_runs, monkeypatch):
    monkeypatch.setattr(R, "_DRIFT_CACHE", {})
    bundle, texts = R.load_canonical(TASK, "evidence_bundle"), R.texts_from_canonical(TASK)
    new, drift, locs, _ = drifted(bundle, texts)
    (task_runs / "canonical" / "campaign50_ref" / "texts" / f"{TASK}.json").write_text(json.dumps(new, indent=1))
    path = write_report(task_runs, drift)
    monkeypatch.setenv("SCRIBE_REHYDRATE_REPORT", str(path))
    out, mans = R.run_generation(TASK, "SCRIBE", 0, MockClient(), level=7, mode="bundle_entry", entry_bundle=bundle,
                                 entry_mode="bundle_entry")
    d = R.run_dir(7, "SCRIBE", "bundle_entry", TASK, 0)
    delivery = json.loads((d / "delivery.json").read_text())
    binding = json.loads((d / "binding.json").read_text())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert delivery["ok"] and delivery["summary"]["text_drift_locators"] == locs
    assert binding["n_text_drift"] == 2 and binding["rehydrate_report_sha256"] == sha
    assert mans and all(m["n_text_drift"] == 2 and m["rehydrate_report_sha256"] == sha for m in mans)
    assert "report_artifact" in out


def test_runner_and_builder_loaders_refuse_dehydrated_files(task_runs):
    canon = task_runs / "canonical" / "campaign50_ref"
    (canon / "texts" / f"{TASK}.json").write_text(json.dumps(WRAPPER, separators=(",", ":")))
    with pytest.raises(DehydratedDataError, match="rehydrate"):
        R.texts_from_canonical(TASK)
    oracle = task_runs / "oracle" / "dev"
    for kind in ("sentences", "evidence_bundle"):
        (oracle / kind).mkdir(parents=True, exist_ok=True)
        (oracle / kind / f"{TASK}.json").write_text(json.dumps(WRAPPER))
    with pytest.raises(DehydratedDataError):
        R.texts_from_oracle(TASK)
    with pytest.raises(DehydratedDataError):
        R.load_oracle(TASK, "evidence_bundle")
    spec = importlib.util.spec_from_file_location("bb_canonical_guard", ROOT / "biolitbench" / "tasks" / "build_canonical_bundles.py")
    B = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(B)
    with pytest.raises(DehydratedDataError):
        B._sentences_by_pmid(TASK, oracle / "sentences")
    with pytest.raises(DehydratedDataError):
        B._texts_source(TASK, canon / "texts")
    (oracle / "sentences" / f"{TASK}.json").write_text(json.dumps({"pmid5_1": "Synthetic sentence."}))
    assert B._sentences_by_pmid(TASK, oracle / "sentences") == {"5": [(1, "pmid5_1", "Synthetic sentence.")]}
    assert R.texts_from_oracle(TASK) == {"5": {"sentences": {"pmid5_1": "Synthetic sentence."}, "abstract": "Synthetic sentence."}}
