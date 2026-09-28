"""Tests the per-task budget builders on synthetic retrieval logs and rollouts."""

import importlib.util
import json
from pathlib import Path

import pytest

BUDGET = Path(__file__).resolve().parents[1] / "biolitbench" / "pool" / "budget"
ARMS = ["autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr", "drtulu"]


def load(name):
    spec = importlib.util.spec_from_file_location(f"budget_{name}", BUDGET / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CAPS = load("build_caps")
KCAP = load("build_kcap")


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))


def fake_runs(root, tasks):
    for t in tasks:
        n = int(t[-1])
        write_jsonl(root / "autosurvey" / t / "retrieval_tap.jsonl", [
            {"method": "get_ids_from_query", "ids": [str(i) for i in range(10 * n)], "cls": "db", "query": "q"},
            {"method": "retrieve_id", "ids": ["1", "2"], "cls": "db", "query": "q2"},
            {"method": "get_titles_from_citations", "ids": ["900", "901"], "cls": "db"},
            "{not json"])
        write_jsonl(root / "surveyforge" / t / "retrieval_tap.jsonl", [
            {"method": "retrieve_id4citation", "ids": [f"202301.{i}" for i in range(1500)], "cls": "db"}])
        for arm in ("surveyg", "sgi", "llmxmr", "drtulu"):
            rows = [{"route": "plain_search", "params": {"q": "x"}, "returned": [{"pmid": i} for i in range(5)]}
                    for _ in range(3 + n)]
            rows.append({"route": "s2_paper_citations", "returned": [{"pmid": 5000 + i} for i in range(40 * n)]})
            write_jsonl(root / arm / t / "search_calls.jsonl", rows)


def test_modules_import():
    assert CAPS.SCHEMA == "budget_caps/1.0" and KCAP.CAPS_SCHEMA == CAPS.SCHEMA
    assert CAPS.BASELINES == ARMS == KCAP.BASELINES


def test_caps_are_the_largest_value_any_baseline_spent(tmp_path):
    tasks = ["t1", "t2"]
    fake_runs(tmp_path / "runs", tasks)
    doc = CAPS.build(CAPS.arm_dirs_from(tmp_path / "runs", ARMS, []), tasks)
    assert doc["n_raw_sources"] == 12
    c = doc["caps"]["t2"]
    assert c["search_calls"] == {"value": 5, "arm": "surveyg"}
    assert c["results_per_call_max"] == {"value": 1500, "arm": "surveyforge"}
    assert c["docs_read"] == {"value": 1500, "arm": "surveyforge"}
    assert doc["caps"]["t1"]["search_calls"] == {"value": 4, "arm": "surveyg"}
    e = doc["effective"]["cap"]["t2"]
    assert e == {"max_search_calls": 5, "max_ranked_output_K": 1000, "max_document_opens": 1000,
                 "max_results_per_call": 1000, "max_docs_read": 1500, "service_clamped_rpc": True}
    assert set(doc["effective"]) == {"cap"}


def test_tap_counts_searches_rows_and_distinct_docs(tmp_path):
    fake_runs(tmp_path / "runs", ["t3"])
    m = CAPS.measure("autosurvey", tmp_path / "runs" / "autosurvey" / "t3")
    assert (m["search_calls"], m["results_per_call_max"], m["docs_read"]) == (2, 30, 32)
    m = CAPS.measure("surveyforge", tmp_path / "runs" / "surveyforge" / "t3")
    assert m["docs_read"] == 1500 and m["results_per_call_max"] == 1500
    m = CAPS.measure("sgi", tmp_path / "runs" / "sgi" / "t3")
    assert (m["search_calls"], m["results_per_call_max"], m["docs_read"]) == (6, 5, 125)


def test_missing_log_is_refused(tmp_path):
    fake_runs(tmp_path / "runs", ["t1"])
    (tmp_path / "runs" / "llmxmr" / "t1" / "search_calls.jsonl").unlink()
    with pytest.raises(SystemExit):
        CAPS.build(CAPS.arm_dirs_from(tmp_path / "runs", ARMS, []), ["t1"])


def test_arm_dir_override_and_task_files(tmp_path):
    fake_runs(tmp_path / "runs", ["t1"])
    (tmp_path / "native").mkdir()
    (tmp_path / "runs" / "drtulu").rename(tmp_path / "native" / "drtulu")
    dirs = CAPS.arm_dirs_from(tmp_path / "runs", ARMS, [f"drtulu={tmp_path / 'native' / 'drtulu'}"])
    assert CAPS.build(dirs, ["t1"])["n_raw_sources"] == 6
    with pytest.raises(SystemExit):
        CAPS.arm_dirs_from(tmp_path / "runs", ARMS, ["gemini=/x"])
    (tmp_path / "a.json").write_text(json.dumps({"tasks": ["b", "a"]}))
    (tmp_path / "a.txt").write_text("b\n# c\na\n")
    assert CAPS.read_tasks(tmp_path / "a.json") == CAPS.read_tasks(tmp_path / "a.txt") == ["a", "b"]


def fake_rollouts(root, sizes):
    for arm, per in sizes.items():
        for t, n in per.items():
            p = root / arm / f"{t}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"papers": [str(i) for i in range(n)]}))


def test_kcap_is_the_largest_baseline_bundle(tmp_path):
    tasks = ["t1", "t2"]
    fake_runs(tmp_path / "runs", tasks)
    caps = tmp_path / "budget_caps.json"
    (tmp_path / "t.txt").write_text("t1\nt2\n")
    assert CAPS.main(["--runs", str(tmp_path / "runs"), "--tasks", str(tmp_path / "t.txt"), "--out", str(caps)]) == 0
    sizes = {a: {"t1": 40, "t2": 30} for a in ARMS}
    sizes["sgi"]["t1"] = 70
    sizes["autosurvey"]["t1"] = 70
    sizes["surveyg"]["t2"] = 1200
    fake_rollouts(tmp_path / "rollouts", sizes)
    doc = KCAP.build(caps, tmp_path / "rollouts", ARMS)
    t1, t2 = doc["tasks"]["t1"], doc["tasks"]["t2"]
    assert (t1["K_cap"], t1["arm"], t1["tied_arms"]) == (70, "autosurvey", ["autosurvey", "sgi"])
    assert t1["effective"]["max_document_opens"] == 70 and t1["effective"]["max_ranked_output_K"] == 70
    assert (t2["K_cap"], t2["P_max"], t2["clamped_to_1000"]) == (1000, 1200, True)
    C = json.loads(caps.read_text())
    for t in tasks:
        rows = doc["tasks"][t]["effective"]["max_results_per_call"]
        assert rows == min(C["caps"][t]["results_per_call_max"]["value"], 1000) == 1000
        assert rows == C["effective"]["cap"][t]["max_results_per_call"]
    assert t1["effective"]["max_results_per_call"] != min(t1["K_cap"], 1000)
    assert doc["inputs"]["caps_sha256"] == KCAP.sha256(caps)
    assert doc["summary"]["n_clamped"] == 1 and doc["summary"]["argmax_arm_counts"]["surveyg"] == 1
    with pytest.raises(SystemExit):
        CAPS.main(["--runs", str(tmp_path / "runs"), "--tasks", str(tmp_path / "t.txt"), "--out", str(caps)])


def test_kcap_refuses_a_missing_rollout(tmp_path):
    fake_runs(tmp_path / "runs", ["t1"])
    caps = tmp_path / "c.json"
    caps.write_text(json.dumps(CAPS.build(CAPS.arm_dirs_from(tmp_path / "runs", ARMS, []), ["t1"])))
    fake_rollouts(tmp_path / "rollouts", {a: {"t1": 5} for a in ARMS if a != "drtulu"})
    with pytest.raises(SystemExit):
        KCAP.build(caps, tmp_path / "rollouts", ARMS)


def test_tap_counts_multi_query_calls_once_at_the_outermost_call(tmp_path):
    d = tmp_path / "autosurvey" / "t1"
    write_jsonl(d / "retrieval_tap.jsonl", [
        {"method": "batch_search", "depth": 1, "ids": [str(i) for i in range(40)], "n_queries": 2},
        {"method": "get_ids_from_queries", "depth": 0, "ids": [str(i) for i in range(40)], "n_queries": 2},
        {"method": "batch_search", "depth": 0, "ids": ["70", "71", "72"], "n_queries": 3},
        {"method": "get_ids_from_query", "ids": ["5", "6"]}])
    m = CAPS.measure("autosurvey", d)
    assert (m["search_calls"], m["results_per_call_max"], m["docs_read"]) == (3, 40, 43)
    assert {"get_ids_from_queries", "batch_search"} <= CAPS.TAP_SEARCH

