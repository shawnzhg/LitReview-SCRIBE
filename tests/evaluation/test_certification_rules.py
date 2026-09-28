"""Tests the certification rules on synthetic data: pair family and radius, the equal-axis
comparison, the refinement family, dropped failures, board-view declarations, calibration, the
rebuild assertions and their count, and peer-band normalisation."""

import importlib.util
import itertools
import json
import types

import numpy as np
import pandas as pd
import pytest

from eval_env import EVAL

from windowbench import axes as A
from windowbench import config as C
from windowbench import decide as DEC
from windowbench import merged_board as MB
from windowbench import roster as R

OURS = "X.bundle_entry"


@pytest.fixture(autouse=True)
def roster(tmp_path):
    p = tmp_path / "roster_extra.json"
    p.write_text(json.dumps({"version": 1, "families": [{"tag": "ours", "keys": [OURS], "seeds": [1, 2]}]}))
    R.reload(str(p))
    yield R
    R.reload()


GROUP = {"synthesis": "claim_coverage", "reasoning": "reasoning", "planning": "organisation",
         "writing": "citation_fidelity", "form": "form"}
FIXED = ["autosurvey.ref", "surveyg.ref", "llmxmr.ref", "sgi.ref", "lira"]
TASKS = [f"t{i:02d}" for i in range(50)]


class FakeData:
    def __init__(self, Z, groups, bot=None):
        self.Z, self.groups, self.tasks = Z, groups, list(TASKS)
        self.clusters = {t: f"c{i % 12}" for i, t in enumerate(TASKS)}
        self.bot = bot or {}
        self.radii, self.eps_fair = None, {}

    def readouts_at(self, obs):
        return list(self.Z[obs].columns) if obs in self.Z else []

    def group(self, r):
        return self.groups[r]

    def flags(self, r, obs):
        return []

    def z_matrix(self, obs, keys):
        Z = self.Z.get(obs)
        return pd.DataFrame() if Z is None else Z[Z.index.get_level_values(0).isin(keys)]


def fake(systems, reads_per_axis, value, obs_list=("system_trunc",), bot=None, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    cols = [f"{ax}_{j}" for ax, n in reads_per_axis.items() for j in range(n)]
    groups = {c: GROUP[c.rsplit("_", 1)[0]] for c in cols}
    Z = {}
    for obs in obs_list:
        idx = pd.MultiIndex.from_tuples([(s, t) for s in systems for t in TASKS], names=["system", "task"])
        M = pd.DataFrame(index=idx, columns=cols, dtype=float)
        for s in systems:
            for c in cols:
                M.loc[pd.IndexSlice[s, :], c] = value(s, c.rsplit("_", 1)[0], obs) + rng.normal(0, noise, len(TASKS))
        Z[obs] = M
    return FakeData(Z, groups, bot)


def test_axis_family_is_all_pairs_of_the_table_and_the_radius_is_not_inflated():
    D = fake(FIXED, {"synthesis": 2, "form": 1}, lambda s, ax, obs: 0.3 + 0.1 * FIXED.index(s))
    conf = A.axis_confirmatory(D, FIXED, "system_trunc")
    assert set(conf.n_pairs_in_family) == {10} and set(conf.n_axes_in_family) == {2}
    assert np.allclose(conf.d_eff, 0.05 / 20)
    lira = conf[(conf.a == "lira") | (conf.b == "lira")]
    assert len(lira) == 8 and (lira.regime == "unfair").all() and not lira.certified.any()
    r = conf[(conf.a == "autosurvey.ref") & (conf.b == "surveyg.ref") & (conf.axis == "form")].iloc[0]
    V = A.axis_values(D, FIXED, "form", "system_trunc", ["form_0"])
    d = (V.loc["autosurvey.ref"] - V.loc["surveyg.ref"]).values
    q, _se, _K = DEC.cluster_t_radius(d, TASKS, D.clusters, 0.05 / 20)
    assert r.q == pytest.approx(q, rel=1e-12) and r.certified


def test_primary_weights_axes_equally():
    val = lambda s, ax, obs: (1.0 if ax == "reasoning" else 0.0) if s == "autosurvey.ref" else 0.5
    D = fake(["autosurvey.ref", "surveyg.ref"], {"reasoning": 13, "form": 1}, val, noise=0.0)
    z = A.axis_z_rows(D, ["autosurvey.ref", "surveyg.ref"], "system_trunc")
    U, axes = A.composite(D, z, "system_trunc", ["autosurvey.ref", "surveyg.ref"])
    assert axes == ["form", "reasoning"]
    assert U.loc["autosurvey.ref"].mean() == pytest.approx(0.5) and U.loc["surveyg.ref"].mean() == pytest.approx(0.5)
    prim = A.primary_confirmatory(D, ["autosurvey.ref", "surveyg.ref"], "system_trunc")
    assert len(prim) == 1 and prim.iloc[0].n_axes == 2 and prim.iloc[0].n_readouts == 14
    assert prim.iloc[0]["diff"] == pytest.approx(0.0) and not prim.iloc[0].certified


def test_a_failed_task_is_dropped_from_every_pair_and_from_the_board_export():
    bot = {("autosurvey.ref", "system_trunc"): {"t00", "t01"}}
    D = fake(FIXED[:3], {"form": 1}, lambda s, ax, obs: 0.5, bot=bot)
    conf = A.axis_confirmatory(D, FIXED[:3], "system_trunc")
    n = {(r.a, r.b): r.n for r in conf.itertuples()}
    assert n[("autosurvey.ref", "surveyg.ref")] == 48 and n[("surveyg.ref", "llmxmr.ref")] == 50
    ex = MB.export_per_task_axis_z(D, FIXED[:3], ["system_trunc"], D.clusters)
    mine = ex[ex.system == "autosurvey.ref"]
    assert len(mine) == 48 and not set(mine.task) & {"t00", "t01"}


def test_refinement_family_is_pipelines_with_a_refinement_stage_times_the_draft_exit_axes():
    systems = ["autosurvey.ref", "surveyg.ref", OURS]
    reads = {"synthesis": 2, "reasoning": 1, "planning": 2, "writing": 1, "form": 1}
    val = lambda s, ax, obs: 0.5 + (0.1 if (obs == "system_trunc" and ax == "form") else 0.0)
    D = fake(systems, reads, val, obs_list=("system_trunc", "draft_trunc"))
    eff = A.refinement_effect(D, systems)
    assert set(eff.axis) == {"synthesis", "reasoning", "writing", "form"}
    assert set(eff.n_in_family) == {2 * 4}
    ours = eff[eff.system == OURS]
    assert (ours.status == "no_refinement_stage").all()
    res = A.big_table(D, systems, ["system_trunc", "draft_trunc"])
    assert "planning" not in set(res["conf"][res["conf"].obs == "draft_trunc"].axis)
    lad = A.draft_ladder(res["conf"])
    assert "planning" not in set(lad.axis)


def _board_module(rel):
    spec = importlib.util.spec_from_file_location("board_" + rel.replace("/", "_").replace(".py", ""), EVAL / "board" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_board_view_builder_declares_external_agents_like_the_baselines(tmp_path):
    vb = _board_module("view_builder.py")
    rv = _board_module("run_view.py")
    keys = sorted(vb.EXT_KEYS)
    assert keys == sorted(["openai_luna_mcp.ref", "openai_luna_mcp", "claude_sonnet_mcp.ref", "claude_sonnet_mcp",
                           "claude_science_mcp.ref", "claude_science_mcp", "gemini_web_dr.ref", "gemini_web_dr", "elicit_sr.ref"])
    assert all(vb.EXT_KEYS[k]["dataset"] == ("fixed_input" if k.endswith(".ref") else "same_pool") for k in keys)
    spec = {"systems": {k: {f: vb.EXT_KEYS[k][f] for f in vb.INJECT_FIELDS if f in vb.EXT_KEYS[k]} for k in keys}}
    p = tmp_path / "roster_inject.json"
    p.write_text(json.dumps(spec))
    fake = types.SimpleNamespace(SYSTEMS={}, _GPT_REF=R._GPT_REF, _GPT_SELF=R._GPT_SELF, _arm=R._arm)
    assert sorted(rv.inject(fake, str(p))["declared"]) == keys
    for k in keys:
        s, base = fake.SYSTEMS[k], (R._GPT_REF if k.endswith(".ref") else R._GPT_SELF)
        assert (s["role"], s["provenance"]) == ("baseline", base["provenance"])
        assert s["model_group"] == vb.EXT_KEYS[k].get("model_group", "gpt56")
        assert s["entries"]["system"] == base["system"] and s["entries"]["planning"] is None and s["draft_is_final"]
    assert fake.SYSTEMS["elicit_sr.ref"]["model_group"] == "elicit"
    with pytest.raises(SystemExit):
        rv.inject(fake, str(p))


def _board_root(tmp_path, gaps, noise=0.02):
    rng = np.random.default_rng(1)
    clusters = {t: f"c{i % 12}" for i, t in enumerate(TASKS)}
    rows = [{"obs": "system_trunc", "axis": ax, "system": s, "task": t, "z": 0.4 + g + float(rng.normal(0, noise)),
             "cluster": clusters[t], "n_readouts": 2}
            for s, g in gaps.items() for t in TASKS for ax in ("form", "planning", "reasoning", "synthesis", "writing")]
    (tmp_path / "inputs").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "inputs" / "per_task_axis_z.csv", index=False)
    pd.DataFrame(sorted(clusters.items()), columns=["task", "cluster"]).to_csv(tmp_path / "inputs" / "task_clusters.csv",
                                                                               index=False)
    return tmp_path


def test_calibration_centres_differences_and_computes_c_star_on_the_leaderboard_family(tmp_path):
    from windowbench import aa_test as AA
    root = _board_root(tmp_path, {"autosurvey.ref": 0.0, "llmxmr.ref": 0.1, "surveyg.ref": 0.2, "sgi.ref": 0.3, "lira": 0.4})
    res = AA.run(root, ["system_trunc"], reps=200)
    assert len(res) == 1
    r = res[0]
    assert r["tier"] == "reference_fed" and r["pairs"] == 10 and r["certified"] == 10
    assert r["familywise_false_certification"] < 0.3
    assert 0.0 < r["c_star"] < 5.0 and r["relations_removed_at_c_star"] == []
    lo = r["leave_one_subfield_out"]
    assert lo["folds"] == 12 and lo["winner_flips"] == 0
    merge = MB.import_merge_windowbench()
    z, clusters = merge.load(root)
    U, _ = merge.composite(z, "system_trunc", ["autosurvey.ref", "llmxmr.ref"])
    one = AA.calibrate(merge, U, clusters, reps=400, seed=3)
    assert one["pairs"] == 1 and abs(one["familywise_false_certification"] - 0.05) < 0.05


def test_replicate_null_uses_the_table_family_and_dropped_failures():
    from windowbench import validate as V
    seeds = R.replicate_keys(OURS)
    D = fake(seeds, {"form": 2, "writing": 2}, lambda s, ax, obs: 0.5, seed=4)
    Z = D.z_matrix("system_trunc", seeds)
    reads = list(Z.columns)
    n_cells, n_cert = V.replicate_null(Z, seeds, reads, TASKS, D.clusters, {}, 0.05)
    assert (n_cells, n_cert) == (12, 0)
    Z2 = Z.copy()
    Z2.loc[pd.IndexSlice[seeds[2], :], :] += 0.2
    assert V.replicate_null(Z2, seeds, reads, TASKS, D.clusters, {}, 0.05)[1] == 8
    failed = {seeds[0]: set(TASKS[:45])}
    assert V.replicate_null(Z, seeds, reads, TASKS, D.clusters, failed, 0.05)[0] == 4


def test_entry_byte_equality_over_the_fixed_input_arms(tmp_path):
    from windowbench import validate as V
    canon, allow = tmp_path / "canonical", tmp_path / "allow"
    canon.mkdir(), allow.mkdir()
    b = {"papers": [{"paper_id": "1"}, {"paper_id": "2"}], "validation": {"review_pmid_excluded": "9"}, "content_hash": "sha256:x"}
    (canon / "t0.json").write_text(json.dumps(b))
    (allow / "t0.json").write_text(json.dumps(["1", "2", "9"]))
    raw = json.dumps(b, indent=1).encode()
    entries = {}
    for arm in ("a/seed0", "b/seed0"):
        p = tmp_path / arm.replace("/", "_")
        p.write_bytes(raw)
        entries[arm] = {"t0": p}
    assert V.entry_byte_equality(["t0"], entries, canon, allow) == {"t0": []}
    (tmp_path / "b_seed0").write_bytes(json.dumps(b).encode())
    assert any("differ" in x for x in V.entry_byte_equality(["t0"], entries, canon, allow)["t0"])
    (tmp_path / "b_seed0").write_bytes(raw)
    (allow / "t0.json").write_text(json.dumps(["1", "2", "3"]))
    assert any("allowlist" in x for x in V.entry_byte_equality(["t0"], entries, canon, allow)["t0"])


def _cells():
    rows = [{"window": "system_trunc", "ours_family": "a+b", "opponent": o, "readout": r, "kind": "graph",
             "ours": 0.5 + 0.01 * i, "opp": 0.4, "diff": 0.1 + 0.01 * i, "q": 0.05, "certified": i % 2 == 0}
            for i, (o, r) in enumerate(itertools.product(["x", "y"], ["r1", "r2", "r3"]))]
    return pd.DataFrame(rows)


def test_the_rebuilt_table_must_reproduce_the_previous_analysis_cell_by_cell(tmp_path):
    from windowbench import validate as V
    prev = _cells()
    now = prev.copy()
    now[V.CELL_VALUES] += 1e-12
    r = V.reproduce_cells(prev, now)
    assert r["ok"] and r["cells"] == 6 and 0 < r["max_abs"] < 1e-9
    prev.to_csv(tmp_path / "cells.csv", index=False)
    assert V.reproduce_cells(pd.read_csv(tmp_path / "cells.csv"), now)["ok"]
    moved = now.copy()
    moved.loc[2, "q"] += 1e-6
    assert not V.reproduce_cells(prev, moved)["ok"]
    flipped = now.copy()
    flipped.loc[0, "certified"] = False
    assert V.reproduce_cells(prev, flipped)["flips"] == 1 and not V.reproduce_cells(prev, flipped)["ok"]
    gap = now.copy()
    gap.loc[3, "diff"] = np.nan
    assert V.reproduce_cells(prev, gap)["nan_mismatch"] == 1 and not V.reproduce_cells(prev, gap)["ok"]
    short = V.reproduce_cells(prev, now.iloc[1:])
    assert short["only_prev"] == 1 and not short["ok"]
    assert not V.reproduce_cells(prev, pd.concat([now, now.iloc[:1]]))["ok"]


def test_the_rebuild_uses_the_families_opponents_and_windows_of_the_previous_analysis(monkeypatch):
    from windowbench import run as RUN
    from windowbench import validate as V
    calls = []
    monkeypatch.setattr(RUN, "build_cells", lambda D, ours, opps, windows: calls.append((ours, opps, sorted(windows))) or _cells())
    prev = pd.concat([_cells(), _cells().assign(window="system")])
    V.rebuild_cells(object(), prev)
    assert calls == [(["a", "b"], ["x", "y"], ["system", "system_trunc"])]


class _AdmissionData:
    def __init__(self, scores):
        self.scores = scores

    def readouts_at(self, obs):
        return sorted(self.scores[self.scores.window == obs].readout.unique())

    def flags(self, r, obs):
        return ["guardrail"] if r == "completion" else []

    def group(self, r):
        return "completion" if r == "completion" else "form"


def test_independent_rescoring_may_differ_only_on_readouts_the_admission_rule_excludes(tmp_path):
    from windowbench import validate as V
    rows = [{"system": s, "task": t, "window": "system_trunc", "readout": r, "value": 0.5}
            for s in ("autosurvey.ref", "sgi.ref") for t in TASKS[:10] for r in ("completion", "form_a")]
    table = pd.DataFrame(rows).assign(source="E13")
    other = pd.concat([pd.DataFrame(rows), pd.DataFrame([{**rows[0], "system": OURS}])], ignore_index=True)
    other.loc[(other.readout == "completion") & (other.task == TASKS[0]), "value"] = 0.9
    n, diff, unique = V.independent_rescoring(table, other)
    assert unique and n == len(table) and len(diff) == 2 and set(diff.readout) == {"completion"}
    D = _AdmissionData(table)
    V.FAILS.clear()
    other.to_parquet(tmp_path / "ok.parquet", index=False)
    V.rescoring_differs_only_on_excluded_readouts(D, str(tmp_path / "ok.parquet"))
    assert V.FAILS == []
    bad = other.copy()
    bad.loc[(bad.readout == "form_a") & (bad.task == TASKS[1]), "value"] = np.nan
    bad.to_csv(tmp_path / "bad.csv", index=False)
    V.rescoring_differs_only_on_excluded_readouts(D, str(tmp_path / "bad.csv"))
    assert len(V.FAILS) == 1
    V.rescoring_differs_only_on_excluded_readouts(D, str(tmp_path / "missing.parquet"))
    V.table_reproduces_previous_analysis(D, None)
    assert len(V.FAILS) == 3
    V.FAILS.clear()
    assert not V.independent_rescoring(table, pd.concat([other, other.iloc[:1]]))[2]


def test_the_rebuild_assertions_need_the_previous_analysis_and_the_independent_scores():
    from windowbench import validate as V
    with pytest.raises(SystemExit):
        V.main(["--previous", "cells.csv"])
    with pytest.raises(SystemExit):
        V.main(["--independent", "window_scores.parquet"])


def test_allocation_scores_use_the_ccbench_normaliser():
    from ccbench.rankability.normalise import calibration, normalise
    from windowbench import allocation_pairs as AP
    H = pd.DataFrame([{"task": "t0", "source_review": s, "is_self": s == "t0", "alloc_ari": v}
                      for s, v in (("t0", 0.8), ("p1", 0.2), ("p2", 0.3), ("p3", 0.4), ("p4", 0.5))]
                     + [{"task": "t1", "source_review": "t1", "is_self": True, "alloc_ari": 0.5}])
    S = pd.DataFrame([{"system": "x", "task": "t0", "alloc_ari": 0.6, "n_sys_sections": 4, "alloc_spread": 2},
                      {"system": "x", "task": "t1", "alloc_ari": 0.25, "n_sys_sections": 4, "alloc_spread": 4}])
    Z = AP.calibrate(S, H, ("alloc_ari",)).set_index("task")
    q10 = float(pd.Series([0.2, 0.3, 0.4, 0.5]).quantile(0.10))
    assert Z.loc["t0", "z"] == pytest.approx((0.6 - q10) / (0.8 - q10))
    assert Z.loc["t1", "z"] == pytest.approx(0.5) and Z.loc["t1", "z_note"] == "no_peers"
    assert Z.loc["t0", "collapse"] == pytest.approx(0.5)


def test_entry_conditioned_synthesis_is_normalised_with_the_peer_band():
    from windowbench import conditioned_synthesis as CS
    H = pd.DataFrame([{"system": f"human:{s}", "task": "t0", "readout": "synP_all_cov", "value": v, "source_review": s}
                      for s, v in (("t0", 0.8), ("p1", 0.2), ("p2", 0.2), ("p3", 0.2))])
    df = pd.DataFrame([{"system": "x", "task": "t0", "readout": "synP_all_cov", "value": 0.6, "window": "system"}])
    Z = CS.calibrate(df, H)
    assert Z.z.iloc[0] == pytest.approx((0.6 - 0.2) / (0.8 - 0.2))


def test_backbone_labels_lead_with_the_backbone_of_the_paper_tables():
    assert R.SYSTEMS[OURS]["backbone"].split(" ")[0] == "Qwen3.8-27B"
    assert R.SYSTEMS["autosurvey.ref"]["backbone"].split(" ")[0] == "gpt-5.6-luna"


def test_a_full_validation_run_makes_twenty_nine_assertions(tmp_path, monkeypatch):
    from windowbench import validate as V

    class Stub:
        tasks, clusters, bot, bot_dropped, radii, eps_fair = list(TASKS), {t: t for t in TASKS}, {}, [], None, {}
        systems_present, draft_systems, outline_axis_loaded = [], [], False
        scores = pd.DataFrame(columns=["system", "task", "window", "readout", "value", "source"])
        human = pd.DataFrame(columns=["system", "task", "window", "readout", "value", "source_review"])
        outline_axis_human = pd.DataFrame()

        def readouts_at(self, obs):
            return []

        def z_matrix(self, obs, keys):
            return pd.DataFrame()

        def flags(self, r, obs):
            return []

        def group(self, r):
            return "form"

    monkeypatch.setattr(V, "Data", Stub)
    monkeypatch.setattr(C, "SOURCES", {"bundle_entry_root": tmp_path, "canonical_bundles": tmp_path, "allowlists": tmp_path})
    monkeypatch.setattr(C, "OUT_DEFAULT", tmp_path)
    V.FAILS.clear()
    V.RUN.clear()
    with pytest.raises(SystemExit):
        V.main(["--previous", str(tmp_path / "cells.csv"), "--independent", str(tmp_path / "s.parquet"), "--replicates", OURS])
    assert len(V.RUN) == 29 and len(set(V.RUN)) == 29
    V.FAILS.clear()
    V.RUN.clear()
