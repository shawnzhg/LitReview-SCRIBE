"""Tests the planning-exit reward targets on a synthetic cells table: the gap to the strongest
pipeline, the half-width scale over admissible cells, the weight rule and the file's integrity."""

import json

import pytest

from training_env import CELLS, EXPECTED, INITIAL, OURS, synthetic_cells


def test_weights_follow_the_gap_rule(targets):
    body = json.loads(targets["targets"].read_text())
    assert body["initial_agent"] == INITIAL and body["baselines"] == ["a", "b", "c"]
    assert body["window_sets"] == {"planning_exit": sorted(CELLS)}
    pe = body["window_readouts"]["planning_exit"]
    for r, (mode, w, best, s) in EXPECTED.items():
        e = pe[r]
        assert (e["mode"], e["w"], e["best_system"]) == (mode, w, best), (r, e)
        assert e["s"] == pytest.approx(s)
        assert e["gap"] == pytest.approx(CELLS[r][best][0] - OURS)
        assert "c" not in e["q_opponents"]


def test_rule_function():
    import reward_weights as RW
    assert RW.weight(0.3, 0.2) == ("gain", 2.5)
    assert RW.weight(10.0, 0.2) == ("gain", 3.0)
    assert RW.weight(-0.05, 0.2) == ("gain", 1.0)
    assert RW.weight(-0.4, 0.2) == ("guard", 0.5)


def test_builder_is_deterministic(targets, tmp_path):
    import reward_weights as RW
    out = tmp_path / "again.json"
    RW.main(["--cells", str(targets["cells"]), "--out", str(out)])
    assert out.read_text() == targets["targets"].read_text()


def test_two_families_or_no_admissible_cell_are_refused(tmp_path):
    import reward_weights as RW
    cells = tmp_path / "cells.csv"
    cells.write_text(synthetic_cells() + synthetic_cells("other.fixed").split("\n", 1)[1])
    with pytest.raises(SystemExit, match="initial agent only"):
        RW.build(str(cells))
    text = synthetic_cells().replace("tolerance", "unfair").replace("exact", "unfair")
    cells.write_text(text)
    with pytest.raises(SystemExit, match="no admissible cell"):
        RW.build(str(cells))


def test_load_targets_checks_the_hash(targets, tmp_path):
    import readout_reward as RR
    t = RR.load_targets(targets["targets"])
    assert "planning_exit" in t["window_sets"]
    body = json.loads(targets["targets"].read_text())
    body["window_readouts"]["planning_exit"]["org_size_fit"]["w"] = 9.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="sha256_16"):
        RR.load_targets(bad)
