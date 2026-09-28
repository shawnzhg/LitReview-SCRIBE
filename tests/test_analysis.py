"""Tests the analysis scripts and the paper-table checker on synthetic inputs."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
ANALYSIS = EVAL / "analysis"


def _load(name, where=ANALYSIS):
    spec = importlib.util.spec_from_file_location(f"analysis_{name}", where / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RF = _load("refusals")
ME = _load("mechanism")
CR = _load("citation_route")
CPT = _load("check_paper_tables", EVAL / "board")


def test_scripts_use_the_repo_scorer():
    import ccbench
    assert Path(ccbench.__file__).resolve().parent.parent == EVAL.resolve()
    for m in (RF, ME, CR):
        assert callable(m.main)


def _radius_dir(tmp_path):
    tasks = [f"t{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    rows = []
    for i, t in enumerate(tasks):
        for s, x in (("autosurvey", 0.9 + 0.01 * i), ("sgi", 0.8 + 0.01 * i), ("lira", 0.01),
                     ("autosurvey.ref", 0.1), ("sgi.ref", 0.2), ("drtulu", 0.95)):
            rows.append({"task": t, "panel": "A", "window": "writing", "kind": "entry", "a": s, "b": None, "d": x,
                         "note": "ref_d1"})
        for s, x in (("autosurvey", 0.9 + 0.01 * i), ("sgi", 0.8 + 0.01 * i)):
            rows.append({"task": t, "panel": "A", "window": "writing", "kind": "exit", "a": s, "b": s + ".ref",
                         "d": x * (0.6 + 0.04 * i + rng.uniform(0, 0.01)), "note": ""})
    dist = pd.DataFrame(rows)
    dist.to_parquet(tmp_path / "distances.parquet", index=False)
    sc = []
    for t in tasks:
        for s in ("autosurvey", "autosurvey.ref", "sgi", "sgi.ref", "lira"):
            for r in ("r1", "r2", "r3"):
                sc.append({"system": s, "task": t, "panel": "A", "window": "writing", "readout": r,
                           "value": float(rng.uniform(0, 1))})
    scores = pd.DataFrame(sc)
    scores.to_parquet(tmp_path / "window_scores.parquet", index=False)
    pd.DataFrame([{"panel": "A", "window": "writing", "readout": r, "L_s": 1.0, "L_M": 0.5, "xi": 0.0}
                  for r in ("r1", "r2", "r3")]).to_parquet(tmp_path / "radii.parquet", index=False)
    pd.DataFrame([{"window": "writing", "readout": "r1", "design": "human_median_gap", "eps_fair": 0.2},
                  {"window": "writing", "readout": "r2", "design": "human_median_gap", "eps_fair": 0.3},
                  {"window": "writing", "readout": "r3", "design": "human_median_gap", "eps_fair": 0.6},
                  {"window": "writing", "readout": "r1", "design": "human_q25_gap", "eps_fair": 0.1},
                  {"window": "system", "readout": "r1", "design": "human_median_gap", "eps_fair": 0.1}]
                 ).to_csv(tmp_path / "tolerances.csv", index=False)
    (tmp_path / "eps_design_selected.json").write_text(json.dumps(
        {"L_M": "ratio_q95", "L_s": "q95", "ucb": "cluster_bootstrap", "eps_in": "mean"}))
    return dist, scores


def test_refusals_on_synthetic_radius_outputs(tmp_path):
    dist, scores = _radius_dir(tmp_path)
    res = RF.compute(tmp_path)
    assert res["pipelines"] == ["autosurvey", "sgi"]
    ent = dist[dist.kind == "entry"].groupby("a").d.mean()
    ex = dist[dist.kind == "exit"]
    x = np.array([ent_d for ent_d in dist[(dist.kind == "entry") & dist.a.isin(["autosurvey", "sgi"])]
                  .set_index(["a", "task"]).loc[list(zip(ex.a, ex.task))].d])
    L = float(np.quantile(ex.d.values / x, 0.95))
    assert res["L_M"] == pytest.approx(L)
    assert res["n_tolerance_readouts"] == 3 and res["n_admissible"] == 2
    val = scores.set_index(["system", "task", "readout"]).value
    Ls = {}
    for r in ("r1", "r2"):
        ratios = [abs(val[(e.a, e.task, r)] - val[(e.b, e.task, r)]) / e.d for e in ex.itertuples()]
        Ls[r] = float(np.quantile(ratios, 0.95))
    eps_fair = {"r1": 0.2, "r2": 0.3}
    t = res["table"].set_index("pipeline")
    for p in ("autosurvey", "sgi"):
        e = ent[p] + ent["lira"]
        eo = {r: Ls[r] * L * e for r in Ls}
        assert t.loc[p, "eps_in_P"] == pytest.approx(ent[p])
        assert t.loc[p, "median_eps_out"] == pytest.approx(np.median(list(eo.values())))
        assert t.loc[p, "refused"] == sum(eo[r] > eps_fair[r] for r in eo)
        assert t.loc[p, "admit_threshold_L_M"] == pytest.approx(max(eps_fair[r] / (Ls[r] * e) for r in Ls))
    assert res["median_L_s"] == pytest.approx(np.median(list(Ls.values())))
    assert set(res["loso"]) == {"autosurvey", "sgi"}
    assert "drtulu" not in res["pipelines"]
    out = tmp_path / "out.json"
    assert RF.main(["--radius-dir", str(tmp_path), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["n_admissible"] == 2


def test_refusals_refuse_without_paired_runs(tmp_path):
    dist, _ = _radius_dir(tmp_path)
    dist[dist.kind == "entry"].to_parquet(tmp_path / "distances.parquet", index=False)
    with pytest.raises(SystemExit):
        RF.compute(tmp_path)


def _ccbench_out(tmp_path):
    rows = []
    for s, papers, words in (("A", [10, 20, 30], [100, 200, 300]), ("A1", [12, 22, 32], [110, 210, 310]),
                             ("B", [5, 6, 7], [1000, 2000, 3000])):
        for i, t in enumerate(("t0", "t1", "t2")):
            rows.append({"system": s, "task": t, "status": "ok" if not (s == "A1" and t == "t2") else "bot",
                         "n_papers": papers[i], "words": words[i]})
            if s != "B":
                p = tmp_path / "rollouts" / s / f"{t}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"graph": {"nodes": list(range(papers[i] + 1)), "edges": []}}))
    (tmp_path / "E1").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "E1" / "conformance.csv", index=False)


def test_mechanism_medians(tmp_path):
    _ccbench_out(tmp_path)
    res = ME.compute(tmp_path, [ME.parse_row("fam=A,A1"), ME.parse_row("b=B")])
    fam, b = res
    assert fam["n_tasks"] == 2
    assert fam["median_papers"] == pytest.approx(np.median([11, 21]))
    assert fam["median_claims"] == pytest.approx(np.median([12, 22]))
    assert fam["by_member"]["A"]["words"] == pytest.approx(150)
    assert b["n_tasks"] == 3 and b["median_words"] == 2000 and np.isnan(b["median_claims"])
    with pytest.raises(SystemExit):
        ME.parse_row("no-members")
    assert ME.main(["--ccbench-out", str(tmp_path), "--row", "b=B"]) == 0


def test_citation_route_counts_and_margins(tmp_path):
    assert CR.pmid_of("PMID:123") == "123" and CR.pmid_of("123") == "123"
    assert CR.pmid_of("DOI:10.1/x") is None and CR.pmid_of(None) is None
    tasks = [f"t{i}" for i in range(12)]
    (tmp_path / "tasks.json").write_text(json.dumps({"tasks": tasks}))
    gold = tmp_path / "gold"
    gold.mkdir()
    for i, t in enumerate(tasks):
        (gold / f"{t}.json").write_text(json.dumps({"review_pmid": 1000 + i}))
        d = tmp_path / "logs" / t
        d.mkdir(parents=True)
        calls = [{"route": "s2_search", "params": {"query": "x"}, "n_returned": 5}]
        if i < 5:
            calls.append({"route": "s2_references", "params": {"paper_id": f"PMID:{1000 + i}"},
                          "n_returned": 3 if i else 0})
        if i == 6:
            calls.append({"route": "s2_citations", "params": {"paper_id": f"PMID:{1000 + i}"}, "n_returned": 3})
        if i == 7:
            calls.append({"route": "s2_references", "params": {"paper_id": "PMID:1"}, "n_returned": 3})
        (d / "search_calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    rt = CR.route_table(tmp_path / "logs", gold, CR.read_tasks(tmp_path / "tasks.json"))
    assert (rt.calls > 0).sum() == 5 and (rt.calls_nonempty > 0).sum() == 4
    board = tmp_path / "board"
    (board / "inputs").mkdir(parents=True)
    z = []
    for i, t in enumerate(tasks):
        for s, base in (("sgi", 0.5), ("x", 0.4), ("y", 0.45)):
            for ax in ("retrieval", "writing"):
                z.append({"obs": "system_trunc", "axis": ax, "system": s, "task": t,
                          "z": base + (0.1 if (s == "sgi" and i < 5) else 0.0), "cluster": f"c{i % 3}",
                          "n_readouts": 1})
    pd.DataFrame(z).to_csv(board / "inputs" / "per_task_axis_z.csv", index=False)
    pd.DataFrame({"task": tasks, "cluster": [f"c{i % 3}" for i in range(12)]}).to_csv(
        board / "inputs" / "task_clusters.csv", index=False)
    pd.DataFrame([{"obs": "system_trunc", "tier": "self_retrieving", "system": s} for s in ("sgi", "x", "y")]
                 ).to_csv(board / "merged_leaderboard.csv", index=False)
    without = sorted(set(tasks) - set(rt[rt.calls > 0].task))
    M = CR.margins(board, "sgi", "self_retrieving", "system_trunc", without).set_index("opponent")
    assert M.loc["x", "n_all"] == 12 and M.loc["x", "n_without_route"] == 7
    assert M.loc["x", "gap_all"] == pytest.approx(0.1 + 0.1 * 5 / 12)
    assert M.loc["x", "gap_without_route"] == pytest.approx(0.1)
    out = tmp_path / "o.json"
    assert CR.main(["--pool-logs", str(tmp_path / "logs"), "--gold", str(gold), "--tasks",
                    str(tmp_path / "tasks.json"), "--board", str(board), "--obs", "system_trunc",
                    "--json", str(out)]) == 0
    assert len(json.loads(out.read_text())["route_tasks"]) == 5


def test_refusals_charge_the_slack_only_for_an_entry_that_differs():
    Ls, fair = pd.Series({"r1": 1.0}), pd.Series({"r1": 0.5})
    one = RF.refusal_table({"p": 0.4, "ref": 0.0}, "ref", ["p"], 1.0, 0.1, Ls, fair).iloc[0]
    two = RF.refusal_table({"p": 0.4, "ref": 0.2}, "ref", ["p"], 1.0, 0.1, Ls, fair).iloc[0]
    assert one.median_eps_out == pytest.approx(0.4 + 0.1) and two.median_eps_out == pytest.approx(0.6 + 0.2)
    assert one.admit_threshold_L_M == pytest.approx((0.5 - 0.1) / 0.4)


TEX = r"""
% BEGIN tab:leaderboard (a)
Backbone & System & Composite & Rank & \rot{Synthesis} \\
\multicolumn{3}{c}{x}\\
\rowcolor{oursrow} \cellcolor{white}Qwen3.8-27B & SCRIBE & \textbf{0.600} & $1$--$2$ & \underline{0.500} \\
\rowcolor{oursrow}  & SCRIBE (untrained) & \underline{0.550} & $1$--$3$ & 0.400 \\
\rowcolor{oursrow} \cellcolor{white}gpt-5.6-luna & SCRIBE-Luna & 0.500 & $3$ & \textbf{0.700} \\
 & SurveyG & 0.400 & $4$ & 0.300 \\
% END tab:leaderboard (a)
"""


def _tables(tmp_path, tex):
    (tmp_path / "paper.tex").write_text(tex)
    rows = [("scr", "ours", "Qwen3.8-27B frozen", "0.600", "1-2", "0.500"), ("scr0", "ours", "Qwen3.8-27B frozen", "0.550", "1-3", "0.400"),
            ("luna", "ours", "gpt-5.6-luna (API)", "0.500", "3", "0.700"), ("surveyg.ref", "baseline", "gpt-5.6-luna (API)", "0.400", "4", "0.300"),
            ("gemini_web_dr.ref", "baseline", "gemini-3.8-flash", "0.450", "3-4", "0.350")]
    with open(tmp_path / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "obs", "system", "role", "backbone", "composite", "cert_interval", "z_synthesis"])
        for r in rows:
            w.writerow(["reference_fed", "system_trunc", *r])
    t = CPT.rows(tex, "tab:leaderboard (a)")
    B = CPT.board(tmp_path, "reference_fed", "system_trunc")
    names = {**CPT.BASE_A, "SCRIBE": "scr", "SCRIBE (untrained)": "scr0", "SCRIBE-Luna": "luna"}
    return t, B, names


def test_paper_table_checker_passes_a_matching_table_and_flags_each_kind_of_mismatch(tmp_path):
    t, B, names = _tables(tmp_path, TEX)
    assert [r["backbone"] for r in t] == ["Qwen3.8-27B", "Qwen3.8-27B", "gpt-5.6-luna", "gpt-5.6-luna"]
    bad, _ = CPT.check_table("a", t, B, names, ["synthesis"], ["gemini_web_dr.ref"])
    assert bad == []
    assert any("neither in the table" in b for b in CPT.check_table("a", t, B, names, ["synthesis"], [])[0])
    for old, new, what in (("& SurveyG & 0.400", "& SurveyG & 0.410", "composite"), ("$1$--$3$", "$1$--$4$", "rank"),
                           ("gpt-5.6-luna & SCRIBE-Luna", "claude-sonnet-5 & SCRIBE-Luna", "backbone"),
                           ("\\underline{0.550}", "0.550", "composite: marked"), ("\\rowcolor{oursrow}  &", " &", "shading")):
        tex = TEX.replace(old, new, 1)
        assert tex != TEX, old
        bad, _ = CPT.check_table("a", *(_tables(tmp_path, tex)), ["synthesis"], ["gemini_web_dr.ref"])
        assert any(what in b for b in bad), (what, bad)
    del names["SCRIBE"]
    assert any("UNMAPPED" in b for b in CPT.check_table("a", t, B, names, ["synthesis"], ["gemini_web_dr.ref"])[0])
