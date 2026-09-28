"""Tests the fail-closed checks around the SCRIBE windows: entry hash and receipt checks, the
finalize checks, the egress-guard listener census and the generation context of every launcher."""

from __future__ import annotations

import importlib.util
import json
import subprocess

import pytest

from harness_env import ROOT, TASK, R, MockClient

from common import seal

L = ROOT / "scribe" / "launchers"


def _spec():
    return R.load_spec(TASK, R.split_of(TASK))


def _native_unit(spec, status="ok", exit_hash=None, n_acq=1):
    d = R.run_dir(7, "SCRIBE", "native_chain", TASK, 0)
    d.mkdir(parents=True, exist_ok=True)
    b = seal({"schema_version": "1", "task_id": TASK, "task_spec_hash": spec["content_hash"], "papers": [],
              "evidence": [], "retrieval_status": "done", "provenance_tier": "agent_retrieved"})
    (d / "evidence_bundle.json").write_text(json.dumps(b))
    m = {"run_id": "x", "window": "acquisition", "status": status, "entry_hash": spec["content_hash"],
         "exit_hash": exit_hash or b["content_hash"]}
    (d / "manifests.jsonl").write_text("".join(json.dumps(m) + "\n" for _ in range(n_acq)))
    return d, b


def test_native_entry_check_accepts_a_sealed_entry_and_refuses_tampering(task_runs):
    spec = _spec()
    d, b = _native_unit(spec)
    assert R.check_native_entry(d, spec, b) == []
    assert R.check_native_entry(d, spec, dict(b, papers=[{"paper_id": "1"}]))
    d, b = _native_unit(spec, exit_hash="sha256:other")
    assert any("exit_hash" in e for e in R.check_native_entry(d, spec, b))
    d, b = _native_unit(spec, status="failed")
    assert any("status" in e for e in R.check_native_entry(d, spec, b))
    d, b = _native_unit(spec, n_acq=2)
    assert any("acquisition manifests" in e for e in R.check_native_entry(d, spec, b))
    d, b = _native_unit(spec, status="budget_exhausted")
    assert R.check_native_entry(d, spec, b) == []


def test_native_chain_run_fails_before_the_first_model_call_on_a_bad_entry(levers_installed, task_runs):
    spec = _spec()
    d, b = _native_unit(spec)
    (d / "evidence_bundle.json").write_text(json.dumps(dict(b, papers=[{"paper_id": "1"}])))
    c = MockClient()
    out, mans = R.run_generation(TASK, "SCRIBE", 0, c, level=7, mode="native_chain", entry_mode="native_chain")
    assert out == {} and c.calls == {}
    assert len(mans) == 1 and mans[0]["status"] == "failed" and mans[0]["failure_reason"] == "harness_failure:entry_hash"
    assert not (d / "synthesis_graph.json").exists()


def _finalize():
    spec = importlib.util.spec_from_file_location("same_pool_finalize_t", ROOT / "scribe" / "launchers" / "same_pool_finalize.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit(tmp_path, status=None, tamper=None, break_chain=False):
    F = _finalize()
    u = tmp_path / "level1" / "SCRIBE" / "native_chain" / TASK / "seed0"
    u.mkdir(parents=True)
    prev, rows = "sha256:spec", []
    for w in F.WINDOWS:
        art = {"window": w, "body": [w]}
        art["content_hash"] = F.content_hash(art)
        if tamper == w:
            art["body"] = ["edited"]
        (u / F.EXITS[w]).write_text(json.dumps(art))
        rows.append({"window": w, "status": (status or {}).get(w, "ok"), "entry_hash": prev, "exit_hash": art["content_hash"]})
        prev = art["content_hash"]
    if break_chain:
        rows[2]["entry_hash"] = "sha256:wrong"
    (u / "manifests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return F, str(u)


def test_finalize_checks_window_status_exit_hashes_and_the_receipt_chain(tmp_path):
    F, u = _unit(tmp_path / "a")
    assert F.unit_problems(u) == []
    F, u = _unit(tmp_path / "b", status={"acquisition": "budget_exhausted"})
    assert F.unit_problems(u) == []
    F, u = _unit(tmp_path / "c", status={"planning": "failed"})
    assert any("planning status" in p for p in F.unit_problems(u))
    F, u = _unit(tmp_path / "d", tamper="writing")
    assert any("does not re-hash" in p for p in F.unit_problems(u))
    F, u = _unit(tmp_path / "e", break_chain=True)
    assert any("planning entry_hash" in p for p in F.unit_problems(u))


def test_egress_guard_opens_no_lane_without_a_listener_census():
    g = (ROOT / "scribe" / "harness" / "runners" / "egress_guard.sh").read_text()
    assert subprocess.run(["bash", "-n", str(ROOT / "scribe" / "harness" / "runners" / "egress_guard.sh")]).returncode == 0
    assert "if lane_sock and not census_src:" in g and "unshare -rn" in g
    assert "python3" not in g and '"$EG_PY"' in g
    fixed = (L / "run_fixed_input.slurm").read_text()
    assert '$RUNNERS/egress_guard.sh "$RUN/egress_proof.json"' in fixed and "RUNNERS=$LITREVIEW_ROOT/scribe/harness/runners" in fixed


@pytest.mark.parametrize("path", ["scribe/launchers/run_same_pool_acquisition.slurm", "scribe/launchers/run_fixed_input.slurm"])
def test_scribe_generation_launchers_serve_a_65536_token_context(path):
    s = (ROOT / path).read_text()
    assert "MAXLEN=65536" in s and "--max-model-len $MAXLEN" in s


def test_training_rollout_server_serves_a_65536_token_context():
    s = (ROOT / "scribe" / "training" / "launch" / "03_grpo.slurm").read_text()
    assert "--max_model_len 65536" in s
