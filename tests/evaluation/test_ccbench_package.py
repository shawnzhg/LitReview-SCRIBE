"""Tests the scorer and certification package on synthetic data: modules load from the repo, the
rule ladder, the composite, rank intervals, truncation, the peer channel, heading matching, the
radius, the controllability operator and seed keys."""

import inspect
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eval_env import EVAL

CCB = EVAL / "ccbench"
GPT = ["autosurvey", "surveyforge", "surveyg", "sgi", "llmxmr"]


def test_modules_resolve_inside_the_repo():
    import ccbench
    from ccbench import contrast_certified, merge_diagnostics, merge_windowbench
    from ccbench.fair import composite, decide, filter, radius, units
    for m in (ccbench, contrast_certified, merge_diagnostics, merge_windowbench, composite, decide, filter, radius, units):
        assert CCB in Path(m.__file__).resolve().parents, m.__file__


def _synthetic_root(tmp_path, gaps):
    rng = np.random.default_rng(0)
    tasks = [f"t{i:02d}" for i in range(50)]
    clusters = {t: f"c{i % 12}" for i, t in enumerate(tasks)}
    rows = []
    for s, g in gaps.items():
        for t in tasks:
            for ax in ("form", "planning", "reasoning", "synthesis", "writing"):
                rows.append({"obs": "system_trunc", "axis": ax, "system": s, "task": t,
                             "z": 0.5 + g + float(rng.normal(0, 0.02)), "cluster": clusters[t], "n_readouts": 3})
    (tmp_path / "inputs").mkdir()
    pd.DataFrame(rows).to_csv(tmp_path / "inputs" / "per_task_axis_z.csv", index=False)
    pd.DataFrame(sorted(clusters.items()), columns=["task", "cluster"]).to_csv(tmp_path / "inputs" / "task_clusters.csv",
                                                                               index=False)
    return tmp_path


def _radius(tmp_path, refused, eps_out=0.8):
    table = [{"pipeline": p, "refused": refused, "readouts": 15, "max_eps_out": eps_out} for p in GPT]
    f = tmp_path / "refusals.json"
    f.write_text(json.dumps({"reference": "lira", "table": table}))
    from ccbench import contrast_certified as CC
    return CC.load_radius(f)


def test_refusals_follow_the_nuisance_radius(tmp_path):
    from ccbench import contrast_certified as CC
    assert "drtulu" in CC.PANEL and len(CC.PANEL) == 7
    pairs = list(itertools.combinations(CC.PANEL, 2))
    drt = [p for p in pairs if "drtulu" in p]
    lira = [p for p in pairs if "lira" in p and "drtulu" not in p]
    assert len(drt) == 6 and len(lira) == 5
    measured = _radius(tmp_path, refused=15)
    for radius in (None, measured):
        assert all(CC.nuisance(a, b, radius) == (math.inf, CC.BACKBONE_CHANGE) for a, b in drt)
        assert all(CC.nuisance(a, b, radius) == (0.0, "") for a, b in pairs if "lira" not in (a, b) and "drtulu" not in (a, b))
        assert sum(CC.admissible(a, b, radius)[0] for a, b in pairs) == 10
    assert all(CC.nuisance(a, b) == (math.inf, CC.ENTRY_UNMEASURED) for a, b in lira)
    assert all(CC.nuisance(a, b, measured) == (0.8, CC.ENTRY_EXCEEDS) for a, b in lira)
    within = _radius(tmp_path, refused=0, eps_out=0.1)
    assert all(CC.nuisance(a, b, within) == (0.1, "") for a, b in lira)


def test_ladder_refuses_eleven_pairs_before_any_score_is_read(tmp_path):
    from ccbench import contrast_certified as CC
    gaps = {s: 0.01 * i for i, s in enumerate(GPT)} | {"lira": 0.3, "drtulu": -0.3}
    root = _synthetic_root(tmp_path, gaps)
    df, score, pairs, adm = CC.contrast("system_trunc", root, _radius(tmp_path, refused=15))
    assert len(pairs) == 21 and len(adm) == 10
    d = df[df.a.eq("drtulu") | df.b.eq("drtulu")]
    assert len(d) == 6 and d.R3.all()
    assert not d.R4.any() and not d.R5.any() and (d.refusal == CC.BACKBONE_CHANGE).all()
    l = df[(df.a.eq("lira") | df.b.eq("lira")) & ~(df.a.eq("drtulu") | df.b.eq("drtulu"))]
    assert len(l) == 5 and l.R3.all() and not l.R4.any() and (l.refusal == CC.ENTRY_EXCEEDS).all()
    assert int((~df.admissible).sum()) == 11
    assert score.index[0] == "lira" and score.index[-1] == "drtulu"
    df2, _, _, adm2 = CC.contrast("system_trunc", root, _radius(tmp_path, refused=0, eps_out=0.5))
    l2 = df2[(df2.a.eq("lira") | df2.b.eq("lira")) & ~(df2.a.eq("drtulu") | df2.b.eq("drtulu"))]
    assert len(adm2) == 15 and l2.admissible.all() and l2.R4.all() and not l2.R5.any()


def test_composite_is_equal_weight_over_axes():
    from ccbench import merge_windowbench as M
    z = pd.DataFrame([{"obs": "system", "axis": ax, "system": s, "task": "t0", "z": v}
                      for s, vals in (("a", {"x": 1.0, "y": 0.0}), ("b", {"x": 0.25, "y": 0.75}))
                      for ax, v in vals.items()])
    U, axes = M.composite(z, "system", ["a", "b"])
    assert axes == ["x", "y"] and U.loc["a", "t0"] == pytest.approx(0.5) and U.loc["b", "t0"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        M.composite(z[~((z.system == "b") & (z.axis == "y"))], "system", ["a", "b"])


def test_rank_intervals_use_the_transitive_closure():
    from ccbench.fair.composite import rank_intervals
    iv, acyclic, naive_ok = rank_intervals(["A", "B", "C"], {("A", "B"), ("B", "C")})
    assert iv == {"A": (1, 1), "B": (2, 2), "C": (3, 3)} and acyclic and not naive_ok


def test_windowbench_uses_ccbench_cluster_t_radius():
    from ccbench.fair import decide as CD
    from windowbench import decide as DEC
    assert "ccbench.fair.decide" in inspect.getsource(DEC.cluster_t_radius)
    rng = np.random.default_rng(3)
    d = rng.normal(0.05, 0.1, 40)
    tasks = [f"t{i}" for i in range(40)]
    clusters = {t: f"c{i % 9}" for i, t in enumerate(tasks)}
    assert DEC.cluster_t_radius(d, tasks, clusters, 0.005) == CD.cluster_t_radius(d, tasks, clusters, 0.005)


def test_drtulu_is_a_same_pool_baseline_only():
    from windowbench import merged_board as MB
    from windowbench import roster as R
    assert "drtulu" in R.DATASETS["same_pool"] and "drtulu" not in R.DATASETS["fixed_input"]
    s = R.SYSTEMS["drtulu"]
    assert s["role"] == "baseline" and s["provenance"] == "self" and s["model_group"] == "drtulu"
    assert s["model_group"] != R.SYSTEMS["autosurvey"]["model_group"] and s["draft_is_final"]
    assert "drtulu.ref" not in R.SYSTEMS and "drtulu" in MB.EXPORT_SYSTEMS


def test_merged_board_imports_the_repo_merge_module():
    from windowbench import merged_board as MB
    merge = MB.import_merge_windowbench()
    assert Path(merge.__file__).resolve().parent == CCB
    assert set(merge.TIERS) == {"reference_fed", "self_retrieving"} and merge.DELTA == 0.05


def _rollout(words_per_sentence, n_per_section, target, panel="A"):
    from ccbench.model import Context, Report, Rollout, Sentence
    secs, sents = [], []
    for k, n in enumerate(n_per_section):
        body = []
        for j in range(n):
            text = " ".join(f"w{k}{j}{i}" for i in range(words_per_sentence)) + "."
            sents.append(Sentence(sid=f"s{k}#{j}", section=f"s{k}", text=text, cites=[f"{k}{j}"]))
            body.append(text)
        secs.append({"id": f"s{k}", "title": f"Section {k}", "level": 1, "text": "\n\n".join(body)})
    words = words_per_sentence * sum(n_per_section)
    return Rollout(system="x", task="t", panel=panel, context=Context(task="t", topic="", cutoff=2020, target_words=target),
                   report=Report(sections=secs, sentences=sents, bibliography=[], words=words))


def test_truncation_stops_at_the_target_word_count_inside_a_section():
    from ccbench.adapters.common import word_count
    from ccbench.readouts import window_views as WV
    for panel in ("A", "H"):
        ro = _rollout(10, [3, 3, 3], 45, panel)
        tv = WV.truncated_view(ro)
        assert tv.report.words == 45 == sum(word_count(s.text) for s in tv.report.sentences)
        assert [s.sid for s in tv.report.sentences] == ["s0#0", "s0#1", "s0#2", "s1#0", "s1#1"]
        assert word_count(tv.report.sentences[-1].text) == 5
        assert [sec["id"] for sec in tv.report.sections] == ["s0", "s1"]
        assert word_count(tv.report.sections[-1]["text"]) == 15
    short = _rollout(10, [2], 45)
    assert WV.truncated_view(short) is short


def test_peers_pass_through_the_system_sentence_channel(tmp_path, monkeypatch):
    from ccbench import paths
    from ccbench.adapters import common, human
    from ccbench.gt import graphs
    from ccbench.ingest import gold
    from ccbench.model import Context
    text1 = "Genes encode products ([17]). They act in pathways (). Models ([5]; [61]) link them."
    rs = {"sections": [{"section_id": "sec_1", "title": "Intro", "heading_level": 1, "section_type": "body"}],
          "paragraphs": [{"paragraph_id": "p1", "section_id": "sec_1", "text": text1}]}
    (tmp_path / "rs.json").write_text(json.dumps(rs))
    claim = graphs.Claim(id="c1", text="They act in pathways ().", section_id="sec_1", section_type="body", paragraph_id="p1",
                         ref_nos=[2], pmids=["222"], groundedness=0.9, contradict=None, weight=1, tier="required")
    g = graphs.GStar(task="t", topic="x", claims=[claim], edges=[], hard_negatives=[], sections=[], refs={17: {"pmid": "171"}, 5: {"pmid": "55"}, 61: {"pmid": "611"}, 2: {"pmid": "222"}},
                     reference_weights={}, cutoff=2020, w_req=1.0)
    monkeypatch.setattr(graphs, "exists", lambda t: True)
    monkeypatch.setattr(graphs, "load", lambda t: g)
    monkeypatch.setattr(gold, "context_for", lambda t: Context(task=t, topic="x", cutoff=2020, target_words=100))
    monkeypatch.setattr(paths, "resolve_opt", lambda rel: tmp_path / "rs.json")
    ro = human.adapt("peer", target_task="t")
    assert [s.text for s in ro.report.sentences] == [human.CITE.sub("", s).strip() for s in common.split_sentences(text1)]
    assert [s.cites for s in ro.report.sentences] == [["171"], ["222"], ["55", "611"]]
    assert ro.report.words == common.word_count(human.CITE.sub(" ", text1)) and ro.report.total_citation_marks == 3


def test_headings_match_by_fuzzy_title_and_shared_claims(monkeypatch):
    from ccbench.gt.units import Units
    from ccbench.model import Context, Report, Rollout, Sentence
    from ccbench.readouts import channel, embed
    rng = np.random.default_rng(0)
    vec = {}
    def enc(texts, prefix="", batch_size=32):
        out = []
        for t in texts:
            if t not in vec:
                v = rng.normal(size=8)
                vec[t] = v / np.linalg.norm(v)
            out.append(vec[t])
        return np.array(out).reshape(len(texts), 8)
    monkeypatch.setattr(embed, "encode", enc)
    monkeypatch.setattr(channel, "null_tau", lambda task, g: 0.99)
    from ccbench.gt import graphs
    claims = [graphs.Claim(id=f"c{i}", text=f"claim number {i} about the topic here", section_id="h1" if i < 2 else "h2", section_type="body",
                           paragraph_id=None, ref_nos=[], pmids=[], groundedness=None, contradict=None, weight=0, tier="admissible") for i in range(4)]
    g = graphs.GStar(task="t", topic="", claims=claims, edges=[], hard_negatives=[], sections=[], refs={}, reference_weights={}, cutoff=2020, w_req=1.0)
    u = Units(task="t", hubs=set(), tail=set(), recent=set(), classic=set(), communities=[], subfields={}, claims_by_tier={}, grounded_hi=[], grounded_lo=[],
              central=[], multi_source={}, reused_papers={}, edges_by_type={}, edges_by_distance={}, hard_negatives=[], motifs={},
              top_sections=[{"id": "h1", "title": "Gene regulation networks", "claims": ["c0", "c1"]}, {"id": "h2", "title": "Protein folding dynamics", "claims": ["c2", "c3"]}],
              section_of_claim={}, paragraph_of_claim={})
    secs = [{"id": "a", "title": "1. Gene regulation networks", "level": 1, "text": claims[0].text},
            {"id": "b", "title": "Protein folding dynamics", "level": 1, "text": claims[0].text},
            {"id": "c", "title": "Unrelated heading", "level": 1, "text": claims[2].text}]
    sents = [Sentence(sid="a#0", section="a", text=claims[0].text), Sentence(sid="c#0", section="c", text=claims[2].text)]
    ro = Rollout(system="x", task="t", panel="A", context=Context(task="t", topic="", cutoff=2020), report=Report(sections=secs, sentences=sents, words=20))
    ind = channel.induce(ro, g, u)
    assert ind.section_match == {"h1": "a"}


def test_task_composite_needs_every_axis_and_rules_need_ten_tasks(tmp_path):
    from ccbench import contrast_certified as CC
    from ccbench import merge_windowbench as M
    z = pd.DataFrame([{"obs": "system", "axis": ax, "system": "a", "task": t, "z": 1.0} for t in ("t0", "t1") for ax in ("x", "y")
                      if not (t == "t1" and ax == "y")])
    U = M.task_composite(z, ["x", "y"])
    assert U.loc["a", "t0"] == 1.0 and np.isnan(U.loc["a", "t1"])
    gaps = {s: 0.05 * i for i, s in enumerate(GPT)} | {"lira": 0.3, "drtulu": -0.3}
    root = _synthetic_root(tmp_path, gaps)
    zz = pd.read_csv(root / "inputs" / "per_task_axis_z.csv")
    keep = zz.task.isin([f"t{i:02d}" for i in range(9)]) | (zz.system != "llmxmr")
    zz[keep].to_csv(root / "inputs" / "per_task_axis_z.csv", index=False)
    df, _, pairs, _ = CC.contrast("system_trunc", root, None)
    assert not any("llmxmr" in (r.a, r.b) for r in df.itertuples()) and len(df) == 15


def test_radius_charges_the_slack_only_for_a_shifted_entry():
    from ccbench.fair import radius as RAD
    tasks = [f"t{i}" for i in range(12)]
    rng = np.random.default_rng(1)
    dist, scores = [], []
    for t in tasks:
        for s, x in (("autosurvey", 0.9), ("lira", 0.0), ("surveyg", 0.8)):
            dist.append({"task": t, "panel": "A", "window": "writing", "kind": "entry", "a": s, "b": "reference", "d": x, "note": "d1_canonical"})
        for s in ("autosurvey", "surveyg"):
            dist.append({"task": t, "panel": "A", "window": "writing", "kind": "exit", "a": s, "b": s + ".ref", "d": 0.5 + 0.3 * rng.random(), "note": "d4"})
        for a, b in (("autosurvey", "lira"), ("autosurvey", "surveyg"), ("lira", "surveyg")):
            dist.append({"task": t, "panel": "A", "window": "writing", "kind": "exit", "a": a, "b": b, "d": 0.5, "note": "d4"})
        for s in ("autosurvey", "lira", "surveyg"):
            scores.append({"system": s, "task": t, "panel": "A", "window": "writing", "readout": "r1", "value": rng.random()})
    dist, scores = pd.DataFrame(dist), pd.DataFrame(scores)
    cfg = {"same_model_groups": {"gpt56": ["autosurvey", "lira", "surveyg"]}, "lm_exclude": []}
    R = RAD.radii(scores, dist, {"L_M": "affine", "L_s": "q95", "eps_in": "mean"}, cfg).set_index(["a", "b"])
    r = R.loc[("autosurvey", "lira")]
    assert r.status == "ok" and r.eps_outside == pytest.approx(r.L_s * (r.L_M * (0.9 + 0.0) + r.xi))
    r2 = R.loc[("autosurvey", "surveyg")]
    assert r2.eps_outside == pytest.approx(r2.L_s * (r2.L_M * (0.9 + 0.8) + 2 * r2.xi))
    assert set(R.reset_index().panel) == {"A"} and "eps_outside_ucb" not in R.columns


def test_operator_computes_the_diagonal():
    from ccbench.distance.kernel import build_kernel
    from ccbench.distance.operator import apply_T, fixed_point, make_union
    from ccbench.model import Context, Event, Label, Rollout
    def ro(n):
        ev = [Event(seq=i, t_start=0.0, t_end=0.0, kind="search" if i % 2 else "llm", label=Label("acquisition", "search" if i % 2 else "generate", "pool_search" if i % 2 else "policy_llm", "1-16", "results_some")) for i in range(n)]
        return Rollout(system="s", task="t", panel="A", context=Context(task="t", topic="", cutoff=2020), events=ev)
    U = make_union([build_kernel("a", [ro(3), ro(4)]), build_kernel("b", [ro(2)])])
    T1 = apply_T(U, np.ones((U.n, U.n)), 0.9, 0.5)
    assert np.all(np.diag(T1) > 0.4)
    d, info = fixed_point(U, np.ones((U.n, U.n)))
    assert info["converged"] and np.abs(np.diag(d)).max() < 1e-5


def test_seed_keys_read_the_seed_directory(monkeypatch):
    from ccbench import build, paths
    assert paths.split_mode("bundle_entry.seed2") == ("bundle_entry", 2) and paths.split_mode("bundle_entry") == ("bundle_entry", 0)
    seen = []
    monkeypatch.setattr(paths, "resolve", lambda rel: seen.append(str(rel)) or Path(rel))
    paths.agent_run_dir("SCRIBE", "t1", "bundle_entry.seed1")
    assert seen[-1].endswith("runs/level7/SCRIBE/bundle_entry/t1/seed1")
    monkeypatch.setenv("CCBENCH_PANEL_B_EXTRA", "SCRIBE.bundle_entry,SCRIBE.bundle_entry.seed1,SCRIBE")
    assert build.agent_runs() == [("SCRIBE", "bundle_entry"), ("SCRIBE", "bundle_entry.seed1"), ("SCRIBE", "native_chain")]
    assert build.system_key("SCRIBE", "bundle_entry.seed1") == "SCRIBE.bundle_entry.seed1"
