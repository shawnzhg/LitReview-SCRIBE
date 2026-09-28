"""Tests that planning fits the cross-batch claims of a synthesis graph before the chunk claims when
the claim window is full, keeping the order of both groups."""

from __future__ import annotations

import copy
import re
from pathlib import Path

from harness_env import R, W, TASK, MockClient
SPEC = {"task_id": "t_synthetic", "question": "Q", "audience": "A", "output_spec": {"target_words": 4000}}
CROSS_TYPES = ("cross_study_synthesis", "conflict", "limitation", "gap")


class Capture(MockClient):

    def chat(self, messages, **k):
        self.prompt = messages[-1]["content"]
        return super().chat(messages, **k)


def _graph(n_chunk=700, n_cross=12):
    claims = [{"claim_id": f"c{i}", "text": f"Finding {i} " + "x" * 300, "type": "study_finding",
               "polarity": "unknown", "evidence_ids": [], "confidence": None} for i in range(n_chunk)]
    fed = {c["claim_id"]: i // 250 for i, c in enumerate(claims)}
    claims += [{"claim_id": f"c{n_chunk + j}", "text": f"Across studies, pattern {j}.",
                "type": CROSS_TYPES[j % 4], "polarity": "mixed", "evidence_ids": [], "confidence": 0.5}
               for j in range(n_cross)]
    return {"task_id": SPEC["task_id"], "content_hash": "sha256:synthetic", "claims": claims,
            "validation": {"cross_batch_of": fed}}


def _plan(graph):
    from calllog import CallLog
    c = Capture()
    plan, _ = W.planning(SPEC, graph, c, CallLog(Path("/dev/null"), "test", SPEC["task_id"]))
    return plan, re.findall(r"^\s+(c\d+) \[", c.prompt, re.M)


def test_cross_claims_are_listed_before_the_chunk_claims():
    g = _graph()
    before = copy.deepcopy(g)
    fed = g["validation"]["cross_batch_of"]
    cross = [c["claim_id"] for c in g["claims"] if c["claim_id"] not in fed]
    chunk = [c["claim_id"] for c in g["claims"] if c["claim_id"] in fed]
    plan, listed = _plan(g)
    assert g == before
    assert listed[:len(cross)] == cross and listed[len(cross):] == chunk[:len(listed) - len(cross)]
    assert plan["validation"]["claims_shown"] == len(listed)
    assert plan["validation"]["claims_dropped_for_context"] == len(g["claims"]) - len(listed) > 0
    assert plan["coverage_map"]["n_claims"] == len(g["claims"])


def test_order_is_unchanged_without_cross_batch_of_or_within_the_window():
    g = _graph(n_chunk=20, n_cross=3)
    assert [c["claim_id"] for c in W.cross_claims_first(g)] == [f"c{i}" for i in range(20, 23)] + \
        [f"c{i}" for i in range(20)]
    del g["validation"]["cross_batch_of"]
    assert W.cross_claims_first(g) == g["claims"]
    assert W.cross_claims_first({"claims": g["claims"]}) == g["claims"]


def test_scribe_run_marks_cross_claims_by_cross_batch_of_and_lists_them_first(levers_installed, task_runs):
    seen = {}

    class Spy(MockClient):
        def chat(self, messages, **k):
            if self.kind(messages[-1]["content"]) == "planning":
                seen["prompt"] = messages[-1]["content"]
            return super().chat(messages, **k)

    out, _ = R.run_generation(TASK, "SCRIBE", 0, Spy(), level=7, mode="bundle_entry", entry_mode="bundle_entry")
    g = out["synthesis_graph"]
    fed = g["validation"]["cross_batch_of"]
    cross = [c["claim_id"] for c in g["claims"] if c["claim_id"] not in fed]
    assert cross and cross == [c["claim_id"] for c in g["claims"] if c["type"] != "study_finding"]
    assert cross == [c["claim_id"] for c in g["claims"]][-len(cross):]
    listed = re.findall(r"^\s+(c\d+) \[", seen["prompt"], re.M)
    assert listed == cross + [c["claim_id"] for c in g["claims"] if c["claim_id"] in fed]
