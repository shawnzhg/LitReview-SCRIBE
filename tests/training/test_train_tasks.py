"""Tests the training task lists: the candidate builder's exclusion of the evaluation tasks, their
peers and the training reviews they cite, its order, the refusal of evaluation ids, and a
deterministic rebuild on the site data."""

import hashlib
import json

import pytest

from training_env import EVAL_TASKS, RUNS_ROOT, TRAIN_MANIFEST, need


def _gold(tmp_path, pmids):
    g = tmp_path / "gold"
    g.mkdir()
    for t, p in pmids.items():
        (g / f"{t}.json").write_text(json.dumps({"review_pmid": p}))
    return g


def test_builder_excludes_eval_tasks_peers_and_cited_reviews(tmp_path):
    import build_train_tasks as BT
    train = [f"t{i}" for i in range(20)]
    gold = _gold(tmp_path, {t: str(1000 + i) for i, t in enumerate(train)})
    r = BT.build(train, eval_tasks={"t0", "e1"}, peer_set={"t1", "t2", "e1"},
                 ref_pmids={"1003", "1004", "99"}, train_gold=gold, n=10)
    assert r["excluded"] == ["t1", "t2", "t3", "t4"]
    assert not set(r["candidates"]) & {"t0", "t1", "t2", "t3", "t4"}
    assert r["candidates"] == sorted(r["candidates"], key=lambda t: hashlib.sha256(t.encode()).hexdigest())
    assert len(r["candidates"]) == 10 and r["counts"]["pool"] == 15
    with pytest.raises(SystemExit, match="fewer than"):
        BT.build(train, {"t0"}, set(), set(), gold, n=20)


def test_builder_needs_every_review_pmid(tmp_path):
    import build_train_tasks as BT
    gold = _gold(tmp_path, {"t1": "1"})
    with pytest.raises(SystemExit, match="no review_pmid"):
        BT.build(["t1", "t2"], set(), set(), set(), gold, n=1)


def test_refuse_eval_refuses_eval_tasks_and_excluded_ids(tmp_path, monkeypatch):
    import planning_task as LP
    ev = tmp_path / "eval.json"
    ev.write_text(json.dumps({"tasks": ["e1", "e2"]}))
    ex = tmp_path / "eval_excluded.txt"
    ex.write_text("p1\np2\n")
    monkeypatch.setattr(LP, "EVAL_TASKS_PATH", str(ev))
    monkeypatch.setattr(LP, "EVAL_EXCLUDED_PATH", str(ex))
    LP.refuse_eval(["t1", "t2"], "test")
    with pytest.raises(ValueError, match="evaluation tasks"):
        LP.refuse_eval(["t1", "e2"], "test")
    with pytest.raises(ValueError, match="peers or references"):
        LP.refuse_eval(["t1", "p2"], "test")
    monkeypatch.setattr(LP, "EVAL_EXCLUDED_PATH", str(tmp_path / "missing.txt"))
    with pytest.raises(ValueError, match="could not be read"):
        LP.refuse_eval(["t1"], "test")


def test_site_rebuild_is_deterministic_and_disjoint(tmp_path):
    import build_train_tasks as BT
    need(TRAIN_MANIFEST, EVAL_TASKS, RUNS_ROOT / "gold" / "train")
    outs = []
    for k in ("a", "b"):
        d = tmp_path / k
        assert BT.main(["--out_dir", str(d)]) == 0
        outs.append(((d / "train_candidates.txt").read_text(), (d / "eval_excluded.txt").read_text()))
    assert outs[0] == outs[1]
    cand = outs[0][0].split()
    excluded = set(outs[0][1].split())
    evals = set(json.loads(EVAL_TASKS.read_text())["tasks"])
    train = {json.loads(l)["task_id"] for l in open(TRAIN_MANIFEST) if l.strip()}
    assert len(cand) == BT.DEFAULT_N == len(set(cand))
    assert set(cand) <= train and not set(cand) & (evals | excluded)
    assert BT.main(["--out_dir", str(tmp_path / "a"), "--check"]) == 0
