"""Tests that the scorer refuses dehydrated release files and graphs without their companion files, on a synthetic data root."""

import importlib.util
import json
from pathlib import Path

import pytest

from eval_env import EVAL

TASK = "pmcid_PMC7000001"
WRAPPER = {"format": "biolit-dehydrated/1", "kind": "x", "task_id": TASK, "path": "x", "data": {}}
FILES = {"graph": f"data_2000/dataset_2000/graphs/{TASK}/graph_enhanced.json",
         "refs": f"data_2000/dataset_2000/refs_enriched/{TASK}.json",
         "grounding": f"data/reference_corpus/train_2000/grounding/{TASK}.json",
         "rs": f"biolit-bench/results_2000/benchmark_claims/{TASK}/review_structure_llm.json"}


def objects():
    return {"graph": {"claims": [{"claim_id": "c1", "sentence": "Synthetic claim about topic A.", "reference_ids": [1],
                                  "groundedness": 0.9, "max_ref_weight": 1, "section_id": "s1", "paragraph_id": "p1"}],
                      "review_level_edges": [], "hard_negatives": [], "reference_weights": {"1": 1}},
            "refs": {"1": {"pmid": "11", "doi": None, "title": "Synthetic paper"}},
            "grounding": {"c1": {"contradict": 0.1}},
            "rs": {"sections": [{"section_id": "s1", "title": "Synthetic heading", "heading_level": 1,
                                 "section_type": "body"}],
                   "paragraphs": [{"paragraph_id": "p1", "section_id": "s1", "text": "Synthetic claim about topic A.",
                                   "central_sentence": ""}]}}


@pytest.fixture
def root(tmp_path, monkeypatch):
    from ccbench import paths
    from ccbench.gt import graphs, units
    for name, obj in objects().items():
        p = tmp_path / FILES[name]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj))
    spec = tmp_path / "runs" / "taskspecs" / "dev" / f"{TASK}.json"
    spec.parent.mkdir(parents=True)
    spec.write_text(json.dumps({"task_id": TASK, "question": "Synthetic topic A", "publication_cutoff": 2020}))
    monkeypatch.setattr(paths, "ICLR", tmp_path)
    monkeypatch.setattr(paths, "PARENT", tmp_path / "no_parent")
    monkeypatch.setattr(paths, "EVAL_PREFIX", "")
    cached = (paths.split_of, graphs.load, units.build)
    for f in cached:
        f.cache_clear()
    yield tmp_path
    for f in cached:
        f.cache_clear()


def test_complete_data_loads(root):
    from ccbench.gt import graphs, units
    g = graphs.load(TASK)
    assert [c.text for c in g.claims] == ["Synthetic claim about topic A."] and g.pmids == {"11"}
    assert [s.title for s in g.sections] == ["Synthetic heading"] and g.cutoff == 2020
    assert g.meta["has_contradict"] and g.meta["has_sections"]
    assert list(units._paragraphs(TASK)[0]) == ["p1"]


@pytest.mark.parametrize("name", list(FILES))
def test_a_dehydrated_file_is_refused(root, name):
    from ccbench import paths
    from ccbench.gt import graphs, units
    (root / FILES[name]).write_text(json.dumps(WRAPPER, separators=(",", ":")))
    with pytest.raises(paths.DehydratedDataError, match="rehydrate"):
        graphs.load(TASK)
    if name == "rs":
        with pytest.raises(paths.DehydratedDataError):
            units._paragraphs(TASK)
        with pytest.raises(paths.DehydratedDataError):
            graphs._load_sections(TASK)


@pytest.mark.parametrize("names", [["rs"], ["refs"], ["grounding"], ["rs", "refs", "grounding"]])
def test_a_graph_without_its_companions_is_refused(root, names):
    from ccbench import paths
    from ccbench.gt import graphs, units
    for name in names:
        (root / FILES[name]).unlink()
    with pytest.raises(paths.MissingCompanionError) as e:
        graphs.load(TASK)
    assert isinstance(e.value, FileNotFoundError)
    assert sorted(Path(m).name for m in e.value.missing) == sorted(Path(FILES[n]).name for n in names)
    if "rs" in names:
        with pytest.raises(paths.MissingCompanionError):
            units._paragraphs(TASK)
        (root / FILES["graph"]).unlink()
        assert units._paragraphs(TASK) == ({}, {}) and graphs._load_sections(TASK) == []


def test_the_human_adapter_refuses_a_dehydrated_or_missing_structure(root, monkeypatch):
    from ccbench import paths
    from ccbench.adapters import human
    from ccbench.gt import graphs
    from ccbench.ingest import gold
    from ccbench.model import Context
    g = graphs.load(TASK)
    monkeypatch.setattr(graphs, "load", lambda t: g)
    monkeypatch.setattr(gold, "context_for", lambda t: Context(task=t, topic="x", cutoff=2020, target_words=100))
    assert human.adapt(TASK).meta["has_body"]
    (root / FILES["rs"]).write_text(json.dumps(WRAPPER))
    with pytest.raises(paths.DehydratedDataError):
        human.adapt(TASK)
    (root / FILES["rs"]).unlink()
    with pytest.raises(paths.MissingCompanionError):
        human.adapt(TASK)


@pytest.mark.parametrize("exc,raises", [("dehydrated", True), ("missing", True), ("other", False)])
def test_null_tau_draws_stop_on_unrehydrated_peers_only(root, monkeypatch, exc, raises):
    from ccbench import paths
    from ccbench.adapters import human
    from ccbench.config import prereg
    from ccbench.gt import graphs
    from ccbench.gt import peers as gt_peers
    from ccbench.readouts import channel
    g = graphs.load(TASK)
    err = {"dehydrated": paths.DehydratedDataError("peer.json"), "missing": paths.MissingCompanionError("peer", ["rs"]),
           "other": ValueError("unrelated")}[exc]
    monkeypatch.setattr(gt_peers, "topic_of", lambda t: "topic A")
    monkeypatch.setattr(gt_peers, "per_paper", lambda: {"peer": {"topic": "topic B", "has_graph": True}})

    def adapt(t, target_task=None):
        raise err

    monkeypatch.setattr(human, "adapt", adapt)
    if raises:
        with pytest.raises(type(err)):
            channel._compute_null_tau(TASK, g, 2, 0.95)
    else:
        assert channel._compute_null_tau(TASK, g, 2, 0.95) == float(prereg()["channel"]["claim_match_cosine"])


def test_the_elicit_ingest_refuses_dehydrated_gold_references(tmp_path, monkeypatch):
    from ccbench import paths
    spec = importlib.util.spec_from_file_location("board_elicit_guard", EVAL / "board" / "ingest" / "elicit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    refs = tmp_path / "refs.json"
    refs.write_text(json.dumps({"refs": [{"pmid": "11", "doi": "10.1/x", "title": "Synthetic paper"}]}))
    monkeypatch.setattr(mod, "_gold_refs", lambda task: refs)
    assert mod.gold_index(TASK, ["11"]) == ({"10.1/x": "11"}, {"syntheticpaper": "11"})
    refs.write_text(json.dumps(WRAPPER))
    with pytest.raises(paths.DehydratedDataError):
        mod.gold_index(TASK, ["11"])
