"""Unit tests on synthetic data: the cluster-robust radius, certification, the common-task and
failure rules, fairness regimes, the roster and axis rank intervals."""

import json

import numpy as np
import pandas as pd
import pytest

from windowbench import decide as DEC
from windowbench import fairness as F
from windowbench import roster as R


def _series(vals, tasks):
    return pd.Series(vals, index=tasks, dtype=float)


def test_cluster_t_radius_matches_plain_t_when_singleton_clusters():
    rng = np.random.default_rng(0)
    d = rng.normal(0.1, 0.2, 40)
    tasks = [f"t{i}" for i in range(40)]
    q, se, K = DEC.cluster_t_radius(d, tasks, {t: t for t in tasks}, 0.05)
    assert K == 40
    from scipy import stats
    se_plain = d.std(ddof=1) / np.sqrt(40)
    assert abs(se - se_plain) / se_plain < 1e-9
    assert abs(q - stats.t.ppf(0.975, 39) * se_plain) < 1e-9


def test_contrast_certifies_large_effect_and_not_null():
    rng = np.random.default_rng(1)
    tasks = [f"t{i}" for i in range(50)]
    clusters = {t: f"c{i % 10}" for i, t in enumerate(tasks)}
    a = _series(rng.normal(0.8, 0.05, 50), tasks)
    b = _series(rng.normal(0.5, 0.05, 50), tasks)
    r = DEC.contrast(a, b, clusters, 0.05 / 4, eps=0.0)
    assert r["certified"] and r["winner"] == "a" and r["n"] == 50 and r["K"] == 10
    r0 = DEC.contrast(a, a + rng.normal(0, 0.01, 50), clusters, 0.05 / 4, eps=0.0)
    assert not r0["certified"]


def test_contrast_eps_nan_is_descriptive_only():
    tasks = [f"t{i}" for i in range(20)]
    a = _series(np.linspace(0.9, 1.0, 20), tasks)
    b = _series(np.linspace(0.1, 0.2, 20), tasks)
    r = DEC.contrast(a, b, {t: t for t in tasks}, 0.05, eps=float("nan"))
    assert r["sampling_decided"] and not r["certified"] and r["status"] == "descriptive"


def test_failed_task_is_dropped_from_the_pair_and_ten_common_tasks_are_needed():
    tasks = [f"t{i}" for i in range(12)]
    a = _series([0.8] * 12, tasks)
    b = _series([0.5] * 10 + [np.nan, np.nan], tasks)
    r = DEC.contrast(DEC.drop_failed(a, {"t0"}), b, {t: t for t in tasks}, 0.05, eps=0.0)
    assert r["n"] == 9 and r["status"] == "too_few_tasks" and not r["certified"]
    r = DEC.contrast(a, b, {t: t for t in tasks}, 0.05, eps=0.0)
    assert r["n"] == 10 and r["status"] != "too_few_tasks" and r["opp"] == 0.5
    assert np.isnan(DEC.drop_failed(b, {"t3"})["t3"])
    from windowbench import config as C
    assert C.REGISTERED["min_tasks"] == 10


@pytest.fixture
def roster(tmp_path):
    p = tmp_path / "roster_extra.json"
    p.write_text(json.dumps({"version": 1, "families": [
        {"tag": "ours", "keys": ["X.bundle_entry"], "seeds": [1, 2]},
        {"tag": "ours_pool", "keys": ["Xn"], "template": "native"}]}))
    R.reload(str(p))
    yield R
    R.reload()


def test_regimes_from_roster_only(roster):
    r = F.regime("X.bundle_entry", "X.bundle_entry.seed1", "system")
    assert r["regime"] == "exact" and r["eps_outside"] == 0.0
    r = F.regime("X.bundle_entry", "autosurvey.ref", "system")
    assert r["regime"] == "exact" and r["cross_model"] is True and r["eps_outside"] == 0.0
    r = F.regime("X.bundle_entry", "autosurvey.ref", "system", scaffold=True)
    assert r["regime"] == "unfair" and r["reason"] == "backbone_outside_scaffold_window"
    r = F.regime("X.bundle_entry", "autosurvey", "system")
    assert r["regime"] == "unfair" and r["reason"] == "no_admissible_tolerance"
    r = F.regime("autosurvey", "autosurvey.ref", "writing")
    assert r["entry_a"] == r["entry_b"] and r["regime"] == "unfair"
    assert F.regime("autosurvey.ref", "surveyg.ref", "writing")["regime"] == "exact"
    r = F.regime("X.bundle_entry", "Xn", "synthesis")
    assert r["regime"] != "exact"
    r = F.regime("lira", "autosurvey", "retrieval")
    assert r["regime"] == "not_observable"
    with pytest.raises(ValueError):
        F.family_regime(["X.bundle_entry", "Xn"], "autosurvey", "system", None, None, None)


def test_roster_families_datasets_and_replicate_keys(roster):
    assert R.resolve_arms("ours") == ["X.bundle_entry"]
    assert R.resolve_arms("fixed_input") == R.DATASETS["fixed_input"] and len(R.DATASETS["fixed_input"]) == 5
    assert R.replicate_keys("X.bundle_entry") == ["X.bundle_entry", "X.bundle_entry.seed1", "X.bundle_entry.seed2"]
    assert all(k in R.SYSTEMS for k in R.replicate_keys("X.bundle_entry"))
    assert R.SYSTEMS["Xn"]["entries"]["retrieval"] == "pool_full"
    with pytest.raises(KeyError):
        R.resolve_arms("nonexistent.arm")


def test_the_roster_extra_is_read_on_first_use(tmp_path, monkeypatch):
    p = tmp_path / "roster_extra.json"
    p.write_text(json.dumps({"version": 1, "families": [{"tag": "late", "keys": ["L.bundle_entry"], "model_group": "gpt56"}]}))
    R.reload()
    assert "L.bundle_entry" not in R.SYSTEMS
    R.reset()
    monkeypatch.setenv("WINDOWBENCH_ROSTER_EXTRA", str(p))
    assert R.SYSTEMS["L.bundle_entry"]["backbone"] == R.BACKBONE["gpt56"]
    monkeypatch.delenv("WINDOWBENCH_ROSTER_EXTRA")
    R.reload()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"version": 1, "families": [{"tag": "b", "keys": ["autosurvey"]}]}))
    with pytest.raises(KeyError):
        R.reload(str(bad))
    R.reload()


def test_admissibility_test_compares_the_nuisance_radius_with_the_tolerance():
    radii = pd.DataFrame([{"window": "system", "a": "lira", "b": "autosurvey", "readout": r, "status": "ok",
                           "eps_outside": e, "L_M": 0.89} for r, e in (("r1", 0.72), ("r2", 0.10))])
    tol = {"r1": 0.21, "r2": 0.21, "r3": 0.21, "r4": 0.7}
    r = F.regime("lira", "autosurvey", "system", "r1", radii, tol)
    assert r["regime"] == "unfair" and r["reason"] == "eps_out_exceeds_eps_fair" and r["eps_outside"] == 0.72
    r = F.regime("lira", "autosurvey", "system", "r2", radii, tol)
    assert r["regime"] == "tolerance" and r["eps_outside"] == 0.10
    r = F.regime("lira", "autosurvey", "system", "r3", radii, tol)
    assert r["regime"] == "unfair" and r["reason"] == "no_finite_radius"
    radii4 = pd.concat([radii, radii.assign(readout="r4", eps_outside=0.1)], ignore_index=True)
    r = F.regime("lira", "autosurvey", "system", "r4", radii4, tol)
    assert r["regime"] == "unfair" and r["reason"] == "no_admissible_tolerance"
    assert F.admissible(0.0, 0.2) == (True, "") and F.admissible(0.3, 0.2)[0] is False


def test_axes_rank_intervals_use_the_transitive_closure():
    from windowbench import axes as AX
    rows = [("A", "B", True, "a"), ("B", "C", True, "a"), ("A", "C", False, None)]
    conf = pd.DataFrame([{"obs": "system_trunc", "axis": "form", "a": a, "b": b, "regime": "exact", "certified": c,
                          "winner": w, "diff": 0.1, "ours": 0.5, "opp": 0.4} for a, b, c, w in rows])
    ri = AX.rank_intervals(conf, ["A", "B", "C"]).set_index("system")
    assert (ri.loc["A", "lo"], ri.loc["A", "hi"]) == (1, 1)
    assert (ri.loc["B", "lo"], ri.loc["B", "hi"]) == (2, 2)
    assert (ri.loc["C", "lo"], ri.loc["C", "hi"]) == (3, 3) and ri.loc["C", "certified_position"]
    assert ri.loc["C", "losses"] == 1


def test_axes_rank_intervals_and_pair_matrix_on_synthetic_conf():
    from windowbench import axes as AX
    conf = pd.DataFrame([
        {"obs": "system_trunc", "axis": "synthesis", "a": "x", "b": "y", "regime": "exact", "certified": True, "winner": "a", "diff": 0.2, "ours": 0.8, "opp": 0.6},
        {"obs": "system_trunc", "axis": "synthesis", "a": "x", "b": "z", "regime": "exact", "certified": True, "winner": "a", "diff": 0.3, "ours": 0.8, "opp": 0.5},
        {"obs": "system_trunc", "axis": "synthesis", "a": "y", "b": "z", "regime": "exact", "certified": False, "winner": None, "diff": 0.1, "ours": 0.6, "opp": 0.5},
    ])
    ri = AX.rank_intervals(conf, ["x", "y", "z"]).set_index("system")
    assert (ri.loc["x", "lo"], ri.loc["x", "hi"]) == (1, 1) and ri.loc["x", "certified_position"]
    assert (ri.loc["y", "lo"], ri.loc["y", "hi"]) == (2, 3)
    assert (ri.loc["z", "lo"], ri.loc["z", "hi"]) == (2, 3)
    assert abs(ri.loc["x", "mean_z"] - 0.8) < 1e-12


def test_axes_members_and_labels(roster):
    from windowbench import axes as AX
    assert AX.members("ours") == ["X.bundle_entry"]
    assert AX.members("autosurvey.ref") == ["autosurvey.ref"]
    assert AX.label("autosurvey.ref") == "AutoSurvey.ref"
    with pytest.raises(KeyError):
        AX.members("no.such.system")
