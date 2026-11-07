"""Tests the retrieval budgets on synthetic caps and K_cap files: hash checks, rows per call,
refusals and the audits' expected rows."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest

from harness_env import ROOT

sys.path.insert(0, str(ROOT / "scribe" / "retrieval"))
import pool_backend as PB
import budget as PBK
import ranking_budget_audit as KA

RPC = {"pmcid_SYN1": 1500, "pmcid_SYN2": 1200, "pmcid_SYN3": 1000, "pmcid_SYN4": 1800, "pmcid_SYN5": 400}
KCAP_OF = {"pmcid_SYN1": 395, "pmcid_SYN2": 610, "pmcid_SYN3": 280, "pmcid_SYN4": 746, "pmcid_SYN5": 120}
CALLS, DOCS = 160, 9000
SPEC_BUDGET = {"max_document_opens": 60}


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _spec_budget(t):
    return dict(SPEC_BUDGET)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("budget")
    caps = {t: {"search_calls": {"arm": "a", "value": CALLS}, "results_per_call_max": {"arm": "b", "value": r},
                "docs_read": {"arm": "c", "value": DOCS}} for t, r in RPC.items()}
    eff_cap = {t: {"max_search_calls": CALLS, "max_ranked_output_K": min(r, 1000), "max_document_opens": max(60, min(r, 1000)),
                   "max_results_per_call": min(r, 1000), "max_docs_read": DOCS} for t, r in RPC.items()}
    cp = d / "budget_caps.json"
    cp.write_text(json.dumps({"schema": "budget_caps/1.0", "caps": caps, "effective": {"cap": eff_cap},
                              "raw_sources_md5": "0" * 32}, indent=1))
    tasks = {t: {"K_cap": k, "arm": "a", "P_by_arm": {"a": k},
                 "effective": {"max_search_calls": CALLS, "max_ranked_output_K": k, "max_document_opens": max(60, k),
                               "max_results_per_call": k, "max_docs_read": DOCS}} for t, k in KCAP_OF.items()}
    kp = d / "kcap.json"
    kp.write_text(json.dumps({"schema": "kcap/1.0", "service_k_max": 1000, "native_opens": 60,
                              "inputs": {"caps_sha256": _sha(cp)}, "rule": "synthetic", "tasks": tasks}, indent=1))
    return d, cp, _sha(cp), kp, _sha(kp)


@pytest.fixture(scope="module")
def cfgs(files):
    _, cp, cs, kp, ks = files
    env = {"SCRIBE_BUDGET_CAPS": str(cp), "SCRIBE_BUDGET_CAPS_SHA256": cs,
           "SCRIBE_KCAP": str(kp), "SCRIBE_KCAP_SHA256": ks}
    return PB.budget_config(env), PBK.load_kcap(env)


def test_pins_are_enforced(files):
    _, cp, cs, kp, ks = files
    with pytest.raises(PB.BudgetConfigError, match="sha256"):
        PB.budget_config({"SCRIBE_BUDGET_CAPS": str(cp), "SCRIBE_BUDGET_CAPS_SHA256": "0" * 64})
    with pytest.raises(PBK.KcapConfigError, match="sha256"):
        PBK.load_kcap({"SCRIBE_KCAP": str(kp), "SCRIBE_KCAP_SHA256": "0" * 64})
    with pytest.raises(PBK.KcapConfigError, match="pinned"):
        PBK.load_kcap({"SCRIBE_KCAP": str(kp)})


def test_rows_per_call_is_min_cap_rpc_1000_on_every_task(cfgs):
    cfg, kc = cfgs
    tasks = sorted(kc["tasks"])
    assert set(tasks) == set(cfg["caps"]) == set(RPC)
    for t in tasks:
        eff, part = PBK.kcap_effective(t, _spec_budget(t), cfg, kc)
        c, ent = cfg["caps"][t], kc["tasks"][t]
        rpc = int(c["results_per_call_max"]["value"])
        assert eff["max_results_per_call"] == min(rpc, 1000)
        assert eff["max_ranked_output_K"] == int(ent["K_cap"])
        assert eff["max_document_opens"] == max(60, int(ent["K_cap"]))
        assert eff["max_search_calls"] == int(c["search_calls"]["value"])
        assert eff["max_docs_read"] == int(c["docs_read"]["value"])
        assert part["rows_rule"] == "min(CAP.rpc, 1000)" and part["cap_rule"] == PBK.CAP_RULE
        assert part["caps"]["results_per_call"] == {"value": min(rpc, 1000), "measured": rpc,
                                                    "arm": c["results_per_call_max"]["arm"], "rule": "min(CAP.rpc, 1000)"}


def test_caps_file_cap_block_cross_check_refuses_a_mismatch(cfgs):
    cfg, kc = cfgs
    t = sorted(kc["tasks"])[0]
    bad = dict(cfg, effective_file={"cap": {t: dict(cfg["effective_file"]["cap"][t], max_results_per_call=7)}})
    with pytest.raises(PBK.KcapConfigError, match="rows per call"):
        PBK.kcap_effective(t, _spec_budget(t), bad, kc)


def test_budget_for_kcap_record(cfgs):
    cfg, kc = cfgs
    t = sorted(kc["tasks"])[1]
    eff, rec = PBK.budget_for_kcap_factory(kc)(t, _spec_budget(t), cfg)
    assert rec["cap_rule"] == PBK.CAP_RULE and rec["rows_rule"] == PBK.ROWS_RULE and rec["effective"] == eff
    assert "mode" not in rec and "cap_switch" not in rec


def test_audit_expected_rows(cfgs):
    cfg, kc = cfgs
    t = sorted(kc["tasks"])[2]
    assert KA.expected_rows({"caps": {}})[0] is None
    _, part = PBK.kcap_effective(t, _spec_budget(t), cfg, kc)
    assert KA.expected_rows({"rows_rule": part["rows_rule"], "caps": part["caps"]}) == (1000, [])
    rb2 = {"rows_rule": part["rows_rule"], "caps": {"results_per_call": {"measured": 900}}}
    assert KA.expected_rows(rb2) == (900, [])
    assert KA.expected_rows({"rows_rule": "other"})[0] is None
    assert KA.expected_rows({"rows_rule": part["rows_rule"], "caps": {}})[0] is None


def _heredoc_after(text, marker):
    i = text.index(marker)
    j = text.index("<<'PYEOF'", i) + len("<<'PYEOF'")
    j = text.index("\n", j) + 1
    k = text.index("\nPYEOF\n", j)
    return text[j:k + 1]


def test_acquisition_launcher_budget_records_show_the_capped_rows(files, tmp_path):
    import subprocess
    _, cp, cs, kp, ks = files
    s = (ROOT / "scribe" / "launchers" / "run_same_pool_acquisition.slurm").read_text()
    py = _heredoc_after(s, '$RUN_PY - "$BUDGET_CAPS" "$BUDGET_CAPS_SHA256" "$KCAP"')
    tasks = ",".join(sorted(RPC))
    r1 = subprocess.run([sys.executable, "-", str(cp), cs, str(kp), ks, tasks, str(tmp_path)], input=py,
                        capture_output=True, text=True)
    assert r1.returncode == 0, r1.stdout + r1.stderr
    rbi = json.loads((tmp_path / "retrieval_budget_inputs.json").read_text())
    assert {t: v["max_results_per_call"] for t, v in rbi["effective"].items()} == {t: min(r, 1000) for t, r in RPC.items()}
    assert {t: v["max_ranked_output_K"] for t, v in rbi["effective"].items()} == KCAP_OF
    assert rbi["problems"] == [] and "superseded_by" not in rbi and "caps_effective_not_used" not in rbi
    assert json.loads((tmp_path / "kcap_inputs.json").read_text())["K_cap"] == KCAP_OF
    assert "K_cap: file sha256" in r1.stdout, r1.stdout
    r2 = subprocess.run([sys.executable, "-", str(cp), cs, str(kp), "0" * 64, tasks, str(tmp_path)], input=py,
                        capture_output=True, text=True)
    assert r2.returncode != 0 and "recorded at job start" in r2.stdout
    r3 = subprocess.run([sys.executable, "-", str(cp), "0" * 64, str(kp), ks, tasks, str(tmp_path)], input=py,
                        capture_output=True, text=True)
    assert r3.returncode != 0 and "recorded at job start" in r3.stdout
