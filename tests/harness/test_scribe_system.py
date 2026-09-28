"""Tests that the harness exposes one system, SCRIBE, refuses other names and lever sets, and passes
the synthesis claims to planning."""

from __future__ import annotations

import json

import pytest

from harness_env import R, W, LV, LEVERS_FILE, TASK, MockClient


def test_one_system():
    assert R.SYSTEMS == ("SCRIBE",)


@pytest.mark.parametrize("system", ["other", "scribe"])
def test_other_systems_refused(system, levers_installed, task_runs):
    with pytest.raises(ValueError):
        R.run_generation(TASK, system, 0, MockClient(), level=7, mode="bundle_entry", entry_mode="bundle_entry")
    assert not (task_runs / "level7").exists(), "a refused system must write nothing"


def test_levers_are_mandatory(task_runs, monkeypatch):
    monkeypatch.delattr(W, "writing", raising=False)
    with pytest.raises(RuntimeError, match="lever writer"):
        R.run_generation(TASK, "SCRIBE", 0, MockClient(), level=7, mode="bundle_entry", entry_mode="bundle_entry")
    assert not (task_runs / "level7").exists()


def test_other_lever_sets_refused(tmp_path):
    with pytest.raises(ValueError):
        LV.install(W, R, {"paragraphs": True})
    other = tmp_path / "levers.json"
    other.write_text(json.dumps({"schema": LV.SCHEMA, "levers": {"paragraphs": True}}))
    with pytest.raises(ValueError):
        LV.load(str(other))


def test_release_levers_constant_is_the_file():
    assert LV.RELEASE_LEVERS == json.loads(LEVERS_FILE.read_text())["levers"] == LV.load(str(LEVERS_FILE))


def test_scribe_synthesis_builds_a_claim_graph_and_planning_sees_its_claims(levers_installed, task_runs):
    seen = {}

    class Spy(MockClient):
        def chat(self, messages, **k):
            u = messages[-1]["content"]
            if self.kind(u) == "planning":
                seen["planning_prompt"] = u
            return super().chat(messages, **k)

    c = Spy()
    out, mans = R.run_generation(TASK, "SCRIBE", 0, c, level=7, mode="bundle_entry", entry_mode="bundle_entry")
    g = out["synthesis_graph"]
    assert c.calls["extract"] == 2 and c.calls["relations"] >= 1 and c.calls["cross"] >= 1
    assert g["relations"], "SCRIBE's graph carries relations"
    assert "relations_not_attempted" not in g["validation"]
    cross = [x for x in g["claims"] if x["type"] != "study_finding"]
    assert cross, "SYN_CROSS claims are in the graph"
    listed = seen["planning_prompt"]
    for x in g["claims"]:
        assert f"  {x['claim_id']} [{x['type']}]:" in listed, x["claim_id"]
    assert [m["window"] for m in mans] == ["synthesis", "planning", "writing"]
    assert all(m["status"] == "ok" and m["system"] == "SCRIBE" for m in mans)
