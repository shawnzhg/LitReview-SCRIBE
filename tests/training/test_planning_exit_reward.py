"""Tests the planning-exit reward: the z-normalisation against the scorer's, the weighted tanh
formula, the band-peer exclusion, and the reward end to end on a synthetic base run."""

import hashlib
import json
import math
import random
import types

import numpy as np
import pandas as pd
import pytest

TASK = "pmcid_T1"
TITLES = ["Introduction", "Immune response to infection", "Vaccine development", "Conclusions"]


def _pe():
    import planning_exit_readouts as PE
    return PE


class FakeEmbed:

    def __init__(self, dim=64):
        self.dim, self.memo = dim, {}

    def _vec(self, t):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def seed(self, texts, vecs):
        self.memo[tuple(texts)] = np.asarray(vecs, dtype=np.float32)

    def _raw(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode(self, texts):
        k = tuple(texts)
        if k not in self.memo:
            self.memo[k] = self._raw(list(texts))
        return self.memo[k]

    @staticmethod
    def cosine_matrix(a, b):
        return a @ b.T if a.size and b.size else np.zeros((a.shape[0], b.shape[0]), np.float32)


def test_z_value_equals_ccbench_normalise():
    PE = _pe()
    from ccbench.rankability import normalise as NRM
    rng = random.Random(0)
    rows, bands = [], []
    for i in range(400):
        d = rng.choice(["quality", "style"])
        own = rng.choice([1.0, 0.8, 0.0, float("nan"), 0.9995])
        qlo = rng.choice([0.0, 0.1, 0.5, float("nan"), 1.0])
        qhi = rng.choice([0.2, 0.6, 1.0, float("nan")])
        s = rng.choice([0.0, 0.05, 0.3, 0.7, 1.0, 1.0001, float("nan")])
        rows.append({"task": f"t{i}", "readout": "r", "value": s, "direction": d})
        bands.append({"task": f"t{i}", "readout": "r", "own": own, "q_lo": qlo, "q_hi": qhi})
    ref = NRM.normalise(pd.DataFrame(rows), pd.DataFrame(bands))
    for row, band, (_, r) in zip(rows, bands, ref.iterrows()):
        z, note = PE.z_value(row["value"], band, row["direction"], "r")
        if pd.isna(r.z):
            assert z is None, (row, band, z)
        else:
            assert math.isclose(z, r.z, abs_tol=1e-12), (row, band, z, r.z)
        assert note == r.z_note


def test_reward_on_synthetic_z(targets):
    import readout_reward as RR
    T = RR.load_targets(targets["targets"])
    pe = T["window_readouts"]["planning_exit"]
    base = {"outline_title_f1_lex": 0.05, "outline_title_f1_emb": 0.50, "org_size_fit": 0.89}
    assert RR.compute(base, base, T, "planning_exit")["reward01"] == 0.5
    cand = {"outline_title_f1_lex": 0.15, "outline_title_f1_emb": 0.45, "org_size_fit": 0.95}
    out = RR.compute(cand, base, T, "planning_exit")
    num = den = 0.0
    for r in pe:
        g = math.tanh((cand[r] - base[r]) / pe[r]["s"])
        g = min(0.0, g) if pe[r]["mode"] == "guard" else g
        num, den = num + pe[r]["w"] * g, den + pe[r]["w"]
    assert math.isclose(out["R"], num / den, abs_tol=1e-12)
    assert math.isclose(out["reward01"], 0.5 * (1 + num / den), abs_tol=1e-12)
    o2 = RR.compute(dict(cand, outline_title_f1_emb=None), base, T, "planning_exit")
    assert o2["n_scored"] == 2 and o2["parts"]["outline_title_f1_emb"] == {"skipped": "missing_on_one_side"}
    assert RR.compute({k: None for k in cand}, base, T, "planning_exit") is None


def test_band_peers_drop_evaluation_ids():
    import precompute_train_bands as PB
    calls = []

    def peers(task, split=None):
        calls.append((task, split))
        return ["p1", "p2", "p3", "p4"]
    got = PB.band_peers("t", types.SimpleNamespace(peers=peers), {"p2", "p4", "e1"})
    assert got == ["p1", "p3"] and calls == [("t", "train")]


def _plan(titles):
    return {"sections": [{"section_id": f"s{i + 1}", "title": t, "claim_ids": [f"c{i + 1}"],
                          "word_budget": 500} for i, t in enumerate(titles)]}


class _NoopValidator:

    def __init__(self, schema):
        self.schema = schema

    def iter_errors(self, obj):
        return iter(())


@pytest.fixture()
def base_run(tmp_path, monkeypatch):
    import importlib.util
    import sys
    if importlib.util.find_spec("jsonschema") is None:
        monkeypatch.setitem(sys.modules, "jsonschema", types.SimpleNamespace(Draft202012Validator=_NoopValidator))
    import planning_task as LP
    claims = [{"claim_id": f"c{i}", "type": "finding", "text": f"claim {i} about infection"}
              for i in range(1, 7)]
    graph = {"claims": claims, "content_hash": "sha256:" + "0" * 64}
    spec = {"task_id": TASK, "question": "Infection and immunity", "audience": "researchers",
            "output_spec": {"target_words": 3000}}
    d = tmp_path / "base" / TASK / "seed0"
    d.mkdir(parents=True)
    (d / "synthesis_graph.json").write_text(json.dumps(graph))
    base_plan = LP.parse_candidate_plan(spec, graph, json.dumps(_plan(TITLES[:3])))
    (d / "outline_plan.json").write_text(json.dumps(base_plan))
    specs = tmp_path / "taskspecs" / "train"
    specs.mkdir(parents=True)
    (specs / f"{TASK}.json").write_text(json.dumps(spec))
    monkeypatch.setattr(LP, "SPEC_ROOT", specs)
    return tmp_path / "base"


def test_planning_exit_reward_end_to_end(base_run, targets):
    PE = _pe()
    import planning_task as LP
    import readout_reward as RR
    fe = FakeEmbed()
    cc = PE.make_cc(fe)
    hum = [cc["cco"]._norm(t) for t in TITLES]
    band = {r: {"own": 1.0, "q_lo": 0.1, "q_hi": 0.9} for r in PE.READOUTS}
    B = {"tasks": {TASK: {"hum_top": hum, "n_hum_top": len(hum), "bands": band, "complete": True}},
         "_path": "<test>", "_sha256_16": "test"}
    T = RR.load_targets(targets["targets"])
    rw = LP.PlanningExitReward(str(base_run), embed=fe, bands=B, targets=T)
    rw.check_tasks([TASK])
    it = lambda text: {"item": {"task": TASK}, "text": text}
    base_plan = rw.base(TASK)["outline_plan"]
    same = rw._one(it(json.dumps(_plan([s["title"] for s in base_plan["sections"]]))))
    assert same.status == "ok" and same.reward == 0.5, (same.status, same.components.get("error"))
    better = rw._one(it(json.dumps(_plan(TITLES))))
    c = better.components
    assert better.status == "ok" and better.reward != 0.5 and 0.0 <= better.reward <= 1.0, c
    assert better.reward == RR.compute(c["pe_z"], c["pe_base_z"], T, "planning_exit")["reward01"]
    bad = rw._one(it("I cannot write an outline."))
    assert bad.status == "parse_failed" and LP.classify_failure({"status": bad.status}) == "candidate"
    assert LP.classify_failure({"status": "failed"}) == "infra" and LP.classify_failure({"status": "ok"}) is None
    out = rw([it(json.dumps(_plan(TITLES))), it("no outline")])
    assert [r.status for r in out] == ["ok", "parse_failed"] and rw.stats["n_parse_failed"] == 1
