"""Tests prepare_root and verify_data on a synthetic snapshot: root layout, rehydrated roots of both profiles, each failing check, drift."""

import copy
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
REL = Path(__file__).resolve().parents[1] / "biolitbench" / "release"
if str(REL) not in sys.path:
    sys.path.insert(0, str(REL))
import prepare_root
import rehydrate
import release_format as rf
import verify_data

T1, T2, P1, R1, N1 = (f"pmcid_PMC80000{i}" for i in range(1, 6))
TEST, PEERS, TRAIN, NR = [T1, T2], [P1], [P1, R1, N1], {N1}
CUTOFF = 2020
SENTS = {"601": ["Made up result one.", "Made up result two."],
         "602": ["Invented observation alpha.", "Invented observation beta."],
         "603": ["Fictional claim x.", "Fictional claim y."],
         "604": ["Toy finding p.", "Toy finding q."]}
CITED = {T1: ["601", "602"], T2: ["603"], P1: ["603", "604"], R1: ["604"], N1: ["601"]}
YEARS = {"601": 2018, "602": 2020, "603": 2019, "604": 2015}


def abstract(p):
    return " ".join(SENTS[p])


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


def tsv(path, header, rows):
    put(path, "\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))


def review(t):
    sents = [f"Synthetic statement one of {t}.", f"Synthetic statement two of {t}."]
    rs = {"paper": {"title": f"Review {t}", "authors": [], "references": []},
          "sections": [{"section_id": "sec_1", "title": f"Heading of {t}", "heading_level": 1,
                        "parent_section_id": None, "section_type": "body"}],
          "paragraphs": [{"paragraph_id": "p_1", "section_id": "sec_1", "text": " ".join(sents), "sentences": sents,
                          "central_sentence": ""}]}
    claims = [{"claim_id": f"claim_{i}", "paragraph_id": "p_1", "section_id": "sec_1", "sentence_index": i,
               "sentence": s, "citation_markers": [], "reference_ids": [1], "groundedness": 0.9,
               "max_ref_weight": 1} for i, s in enumerate(sents)]
    edge = {"source_id": "claim_0", "target_id": "claim_1", "source_text": sents[0], "target_text": sents[1]}
    graph = {"claims": claims, "review_level_edges": [dict(edge, relation_type="evidence")],
             "restatement_edges": [], "hard_negatives": [], "local_edges": [edge], "reference_weights": {"1": 1}}
    return rs, graph


def dry_rs(rs):
    d = copy.deepcopy(rs)
    for p in d["paragraphs"]:
        p["sentences"] = rf.sentences_placeholder(p["sentences"])
        p["text"] = rf.text_placeholder(p["text"])
    for s in d["sections"]:
        s["title"] = rf.text_placeholder(s["title"])
    return d


def dry_graph(g):
    d = copy.deepcopy(g)
    for c in d["claims"]:
        c["citation_markers"] = rf.marker_spans(c["sentence"], c["citation_markers"])
        c["sentence"] = None
    for name in rf.EDGE_LISTS:
        for e in d[name]:
            e["source_text"] = e["target_text"] = None
    return d


def texts_of(t):
    out = {}
    for p in CITED[t]:
        title = f"Title {p}"
        out[p] = {"title": title, "abstract": abstract(p),
                  "sentences": {f"pmid{p}_0": title, **{f"pmid{p}_{k + 1}": s for k, s in enumerate(SENTS[p])}}}
    return out


def dry_texts(texts):
    d = copy.deepcopy(texts)
    for p, e in d.items():
        a = e["abstract"]
        e["abstract"] = {"@": "pm"}
        for loc, s in e["sentences"].items():
            e["sentences"][loc] = ["t", 0, len(s)] if s == e["title"] else ["a", a.find(s), a.find(s) + len(s)]
    return d


def bundle_of(t, texts):
    ev = []
    for p, e in texts.items():
        ev.append({"evidence_id": f"e_{p}", "paper_id": p, "granularity": "abstract", "locator": f"pmid{p}",
                   "text_hash": rf.text_hash(e["abstract"])})
        ev += [{"evidence_id": f"e_{loc}", "paper_id": p, "granularity": "abstract_sentence", "locator": loc,
                "text_hash": rf.text_hash(s)} for loc, s in e["sentences"].items()]
    return {"task_id": t, "papers": [{"paper_id": p, "year": YEARS[p]} for p in texts], "evidence": ev}


def build(tmp):
    snap, work = tmp / "hf", tmp / "work"
    rows, absrows = [], {}

    def store(kind, t, prof, obj, split=None, raw=None, path=None):
        path = path or rf.contract_path(kind, t, split)
        ser = rf.SERIALIZATION.get(kind, "a")
        data = raw if raw is not None else rf.dumps(obj, "a" if ser == "bytes" else ser)
        put(snap / "store" / path, data)
        rows.append((path, kind, t, "store", "file", prof, rf.sha256_bytes(data)))

    def dry(kind, t, prof, obj, d, gate="file", source=None):
        path = rf.contract_path(kind, t)
        data = rf.dumps(obj, rf.SERIALIZATION[kind])
        put(snap / "dehydrated" / path, rf.dumps_wrapper(rf.make_wrapper(kind, t, path, rf.sha256_bytes(data), d, source)))
        rows.append((path, kind, t, "rehydrate", gate, prof, rf.sha256_bytes(data)))

    for t in TEST + TRAIN:
        rs, g = review(t)
        prof = "test,train" if t in PEERS else ("test" if t in TEST else "train")
        src = {"pmcid": t[6:], "pmcid_ver": t[6:] + ".1", "rehydratable": t not in NR}
        dry("rs", t, prof, rs, dry_rs(rs), source=src)
        dry("graph", t, prof, g, dry_graph(g))
        if t not in NR:
            put(work / "articles" / t / "review_structure_llm.json",
                json.dumps({"paragraphs": [{"paragraph_id": p["paragraph_id"], "text": p["text"]} for p in rs["paragraphs"]],
                            "sections": [{"section_id": s["section_id"], "title": s["title"]} for s in rs["sections"]]}))
        refs = {"1": {"pmid": CITED[t][0], "doi": None, "title": f"Title {CITED[t][0]}"}}
        store("refs_enriched", t, "test,train" if t in TEST + PEERS else "train", refs)
        store("grounding", t, prof, None, raw=json.dumps({"claim_0": {"contradict": 0.1}}).encode())
        split = "dev" if t in TEST else "train"
        spec = {"task_id": t, "split": split, "question": f"Synthetic question {t}", "publication_cutoff": CUTOFF}
        store("taskspec", t, prof, None, split, raw=json.dumps(spec, indent=1).encode())
        texts = texts_of(t)
        for p in texts:
            absrows.setdefault(p, set()).add("test" if t in TEST else "train")
        if t in TEST:
            store("gold", t, "test", {"task_id": t, "review_pmid": "1" + t[-1]}, "dev")
            store("references", t, "test", {"references": []})
            b = bundle_of(t, texts)
            store("bundle_test", t, "test", None, raw=json.dumps(b).encode())
            dry("texts_test", t, "test", texts, dry_texts(texts), "unit")
            oracle = {f"pmid{p}_{k}": s for p in texts for k, s in enumerate(SENTS[p])}
            ob = {"task_id": t, "papers": [], "evidence": [
                {"evidence_id": f"o_{loc}", "paper_id": loc[4:7], "granularity": "abstract_sentence", "locator": loc,
                 "text_hash": rf.text_hash(s)} for loc, s in oracle.items()]}
            store("oracle_bundle", t, "test", None, raw=json.dumps(ob).encode())
            spans = {loc: [abstract(loc[4:7]).find(s), abstract(loc[4:7]).find(s) + len(s)] for loc, s in oracle.items()}
            dry("oracle_sentences", t, "test", oracle, spans, "unit")
            base = {"topic": "x", "refs": [{"pmid": p, "doi": None, "title": f"Title {p}", "abstract": abstract(p)} for p in texts]}
            dbase = copy.deepcopy(base)
            for r in dbase["refs"]:
                r["abstract"] = {"@": "pm", **rf.text_placeholder(r["abstract"])}
            dry("baseline_refs", t, "test", base, dbase, "unit")
            store("allowlist_test", t, "test", sorted(texts))
            store("allowlist_test", t, "test", None, raw=b"\x93NUMPY synthetic", path=rf.contract_path("allowlist_test", t)[:-5] + ".npy")
        else:
            store("gold", t, "train", {"task_id": t, "review_pmid": "2" + t[-1]}, "train")
            store("references", t, "train", {"references": []})
            store("allowlist_train", t, "train", None, raw=json.dumps(sorted(texts)).encode())
            store("bundle_train", t, "train", None, raw=json.dumps(bundle_of(t, texts)).encode())
            dry("texts_train", t, "train", texts, dry_texts(texts), "unit")
    for path, obj, prof in (("data/pool_corpus/campaign50.json", {"tasks": TEST}, "test,train"),
                            ("data_2000/dataset_2000/manifests/per_paper.json", [], "test,train"),
                            ("data_2000/dataset_2000/manifests/domain_index.json", {}, "test,train"),
                            ("data_2000/dataset_2000/manifests/ref_years.json", YEARS, "test,train"),
                            ("data_2000/dataset_2000/manifests/pmid_subfield.json", {}, "test,train"),
                            ("runs/taskspecs/dev_index.jsonl", None, "test"),
                            ("runs/canonical/train/ref_years.json", YEARS, "train"),
                            ("runs/canonical/train/cutoffs.json", {t: CUTOFF for t in TRAIN}, "train")):
        data = "".join(json.dumps({"task_id": t}) + "\n" for t in TEST).encode() if obj is None else json.dumps(obj).encode()
        put(snap / "store" / path, data)
        rows.append((path, "global", "", "store", "file", prof, rf.sha256_bytes(data)))
    for rel, data in (("E10/null_tau.json", json.dumps({t: 0.5 for t in TEST})), ("E10/calibration.csv", "readout,x\n"),
                      ("readout_membership.json", json.dumps({"readouts": []}))):
        put(snap / "scoring" / "test" / rel, data)
        rows.append(("out/" + rel, "scoring", "", "scoring", "file", "test", rf.sha256_bytes(data.encode())))
    tsv(snap / "manifests" / "original_sha256.tsv", rf.ORIGINAL_COLUMNS, sorted(rows))
    tsv(snap / "manifests" / "abstract_sha256.tsv", rf.ABSTRACT_COLUMNS,
        [(p, rf.sha256_hex(abstract(p)), len(abstract(p)), ",".join(sorted(absrows[p]))) for p in sorted(absrows, key=int)])
    tsv(snap / "manifests" / "article_sources.tsv", rf.SOURCE_COLUMNS,
        [(t, t[6:], t[6:] + ".1", "false" if t in NR else "true", "test,train" if t in PEERS else
          ("test" if t in TEST else "train")) for t in TEST + TRAIN])
    put(snap / "splits" / "test.txt", "".join(t + "\n" for t in sorted(TEST)))
    put(snap / "splits" / "train.txt", "".join(t + "\n" for t in sorted(TRAIN)))
    tsv(snap / "splits" / "test_peers.tsv", ("task_id", "peer_id"), [(t, P1) for t in TEST])
    put(snap / "splits" / "train.jsonl", "".join(json.dumps({"task_id": t}) + "\n" for t in TRAIN))
    put(snap / "README.md", "synthetic card\n")
    write_manifest(snap)
    put(work / "abstracts.jsonl", "".join(json.dumps({"pmid": p, "abstract": abstract(p), "render": "glkb/1"}) + "\n"
                                          for p in sorted(SENTS)))
    return snap, work


def write_manifest(snap):
    files = sorted(str(p.relative_to(snap)) for p in snap.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    put(snap / "MANIFEST.sha256", "".join(f"{rf.file_sha256(snap / f)}  {f}\n" for f in files))


@pytest.fixture
def test_set(monkeypatch):
    monkeypatch.setattr(verify_data, "TEST_SET_SHA16", hashlib.sha256("\n".join(sorted(TEST)).encode()).hexdigest()[:16])
    monkeypatch.delenv("CCBENCH_PARENT", raising=False)


def make_root(tmp, snap, work, profile, abstracts=None):
    root = tmp / f"root_{profile}"
    assert prepare_root.main(["--snapshot", str(snap), "--profile", profile, "--root", str(root)]) == 0
    rehydrate.main(["--snapshot", str(snap), "--root", str(root), "--profile", profile, "--articles",
                    str(work / "articles"), "--abstracts", str(abstracts or work / "abstracts.jsonl")])
    return root


def failures(root):
    rep = json.loads((root / "verify_report.json").read_text())
    return {k: v["failures"] for k, v in rep["checks"].items() if v["failures"]}


def test_prepare_root_links_the_store_copies_scoring_and_writes_site_env(tmp_path, capsys):
    snap, work = build(tmp_path)
    root = tmp_path / "root"
    assert prepare_root.main(["--snapshot", str(snap), "--profile", "test", "--root", str(root)]) == 0
    first = json.loads(capsys.readouterr().out)
    spec = root / rf.contract_path("taskspec", P1, "train")
    assert spec.is_symlink() and os.readlink(spec) == str(snap / "store" / rf.contract_path("taskspec", P1, "train"))
    assert (root / "out/E10/null_tau.json").is_file() and not (root / "out/E10/null_tau.json").is_symlink()
    assert (root / "manifests").is_symlink() and (root / "splits").is_symlink()
    assert not (root / rf.contract_path("texts_test", T1)).exists() and not (root / rf.contract_path("rs", T1)).exists()
    assert not (root / rf.contract_path("gold", R1, "train")).exists() and (root / "runs/taskspecs/dev_index.jsonl").is_file()
    env = dict(line.split("=", 1) for line in (root / "site.env").read_text().splitlines() if not line.startswith("#"))
    assert env["CCBENCH_ROOT"] == str(root) and env["CCBENCH_EVAL_SUBDIR"] == "" and env["SCRIBE_RUNS_ROOT"] == str(root / "runs")
    assert env["CCBENCH_OUT"] == env["CCBENCH_SHARED_OUT"] == str(root / "scores")
    assert env["WINDOWBENCH_MEMBERSHIP"] == str(root / "scores/readout_membership.json")
    assert (root / "scores/E10/null_tau.json").read_bytes() == (root / "out/E10/null_tau.json").read_bytes()
    assert env["SCRIBE_REHYDRATE_REPORT"] == str(root / "rehydrate_report.json")
    assert env["SCRIBE_ALLOWLISTS"] == str(root / "eval/test/allowlists") and "SCRIBE_TRAIN_MANIFEST" not in env
    assert prepare_root.main(["--snapshot", str(snap), "--profile", "test", "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["linked"] == 0 and first["linked"] > 0 and first["seeded"] == first["copied"] > 0
    (root / "scores/E10/null_tau.json").write_text("{}")
    assert prepare_root.main(["--snapshot", str(snap), "--profile", "test", "--root", str(root)]) == 0
    assert (root / "scores/E10/null_tau.json").read_text() == "{}"
    capsys.readouterr()
    (root / "out/E10/null_tau.json").write_text("{}")
    assert prepare_root.main(["--snapshot", str(snap), "--profile", "test", "--root", str(root)]) == 1
    assert "not replaced" in capsys.readouterr().err
    train = tmp_path / "train_root"
    assert prepare_root.main(["--snapshot", str(snap), "--profile", "train", "--root", str(train), "--parent", str(tmp_path / "models")]) == 0
    env = dict(line.split("=", 1) for line in (train / "site.env").read_text().splitlines() if not line.startswith("#"))
    assert env["SCRIBE_TRAIN_MANIFEST"] == str(train / "splits/train.jsonl") and env["CCBENCH_PARENT"] == str(tmp_path / "models")
    assert not (train / "out/E10").exists() and not (train / "runs/taskspecs/dev").exists()
    assert (train / rf.contract_path("refs_enriched", T1)).is_symlink()
    with pytest.raises(SystemExit):
        prepare_root.main(["--snapshot", str(snap), "--profile", "test", "--root", str(snap / "inside")])


@pytest.mark.parametrize("profile", ["test", "train"])
def test_a_rehydrated_root_passes_every_check(tmp_path, test_set, profile, capsys):
    snap, work = build(tmp_path)
    root = make_root(tmp_path, snap, work, profile)
    assert verify_data.main(["--root", str(root), "--profile", profile, "--snapshot", str(snap)]) == 0
    rep = json.loads((root / "verify_report.json").read_text())
    assert rep["ok"] and all(v["ok"] for v in rep["checks"].values())
    assert rep["checks"]["closure"]["n_checked"] > 0 and rep["checks"]["units"]["n_checked"] > 0
    if profile == "train":
        assert rep["checks"]["null_tau"].get("skipped") and not (root / rf.contract_path("graph", N1)).exists()
    assert verify_data.main(["--root", str(root), "--profile", profile]) == 0


def test_verify_data_names_every_broken_check(tmp_path, test_set, monkeypatch, capsys):
    snap, work = build(tmp_path)
    root = make_root(tmp_path, snap, work, "test")
    (root / rf.contract_path("grounding", P1)).unlink()
    rs_path = root / rf.contract_path("rs", T2)
    rs_path.write_bytes((snap / "dehydrated" / rf.contract_path("rs", T2)).read_bytes())
    shutil.copytree(root / "data_2000/dataset_2000/graphs" / T1, root / "data_2000/dataset_2000/graphs/pmcid_PMC8999999")
    put(root / "runs/taskspecs/dev/pmcid_PMC8999999.json", "{}")
    bundle = root / rf.contract_path("bundle_test", T1)
    late = json.loads(bundle.read_text())
    late["papers"][0]["year"] = CUTOFF + 1
    bundle.unlink()
    put(bundle, json.dumps(late))
    texts = root / rf.contract_path("texts_test", T1)
    tx = json.loads(texts.read_text())
    tx["601"]["sentences"]["pmid601_1"] = "Made up result one, edited."
    texts.write_text(json.dumps(tx, indent=1, ensure_ascii=False))
    tau = root / "out/E10/null_tau.json"
    tau.write_text(json.dumps({T1: 0.5}))
    parent = tmp_path / "parent"
    put(parent / rf.contract_path("refs_enriched", T1), "{}")
    put(snap / "stray.txt", "x")
    assert verify_data.main(["--root", str(root), "--profile", "test", "--snapshot", str(snap), "--parent", str(parent)]) == 1
    f = failures(root)
    assert set(f) == {"snapshot", "closure", "file_gate", "units", "wrappers", "graph_set", "splits", "cutoff",
                      "null_tau", "parent"}
    assert f["snapshot"] == ["stray.txt: in the snapshot but not listed in MANIFEST.sha256"]
    assert f["closure"] == [f"{rf.contract_path('grounding', P1)}: missing"]
    assert any(rf.contract_path("rs", T2) in m for m in f["wrappers"])
    assert {m.split(":")[0] for m in f["file_gate"]} == {rf.contract_path("rs", T2), rf.contract_path("bundle_test", T1),
                                                         "out/E10/null_tau.json"}
    assert f["units"] == [f"{rf.contract_path('texts_test', T1)}: sha256 differs and rehydrate_report.json does not list the file as drift"]
    assert "pmcid_PMC8999999" in f["graph_set"][0] and "pmcid_PMC8999999" in f["splits"][0]
    assert "1 papers dated after the cutoff year" in f["cutoff"][0] and "missing 1" in f["null_tau"][0]
    assert f["parent"] == [f"{rf.contract_path('refs_enriched', T1)}: also present under CCBENCH_PARENT={parent}; the scorer may read it from there"]
    assert "FAIL closure" in capsys.readouterr().err
    rep = json.loads((root / "rehydrate_report.json").read_text())
    rep["root"] = str(tmp_path / "elsewhere")
    (root / "rehydrate_report.json").write_text(json.dumps(rep))
    assert verify_data.main(["--root", str(root), "--profile", "train"]) == 1
    f = failures(root)
    assert any("root" in m for m in f["report"]) and any("profile" in m for m in f["report"])


def test_verify_data_accepts_exactly_the_reported_drift(tmp_path, test_set):
    snap, work = build(tmp_path)
    today = work / "today.jsonl"
    lines = [json.loads(l) for l in (work / "abstracts.jsonl").read_text().splitlines()]
    for r in lines:
        if r["pmid"] == "602":
            r["abstract"] = "Invented observation alpha. Invented observation beta, revised today."
    put(today, "".join(json.dumps(r) + "\n" for r in lines))
    root = make_root(tmp_path, snap, work, "test", today)
    rep = json.loads((root / "rehydrate_report.json").read_text())
    assert {(d["kind"], d["locator"]) for d in rep["drift"]} >= {("texts_test", "pmid602"), ("texts_test", "pmid602_2")}
    assert verify_data.main(["--root", str(root), "--profile", "test"]) == 0
    texts = root / rf.contract_path("texts_test", T1)
    tx = json.loads(texts.read_text())
    tx["602"]["sentences"]["pmid602_2"] = "Invented observation beta, changed again."
    tx["601"]["sentences"]["pmid601_1"] = "Made up result one, edited."
    texts.write_text(json.dumps(tx, indent=1, ensure_ascii=False))
    assert verify_data.main(["--root", str(root), "--profile", "test"]) == 1
    msgs = failures(root)["units"]
    assert any("pmid602_2 (abstract_sentence): text does not hash to the report's written_text_hash" in m for m in msgs)
    assert any("pmid601_1 (abstract_sentence): differs from its sealed digest and the report does not list it" in m for m in msgs)
    assert "revised today" not in (root / "verify_report.json").read_text()


def test_verify_data_refuses_a_report_with_a_failed_file_or_unfetched_abstracts(tmp_path, test_set):
    snap, work = build(tmp_path)
    root = make_root(tmp_path, snap, work, "test")
    rep = json.loads((root / "rehydrate_report.json").read_text())
    rep["files"] = [{"path": rf.contract_path("rs", P1), "kind": "rs", "task_id": P1, "status": "failed",
                     "target_sha256": "0" * 64, "written_sha256": None, "reason": "digest_mismatch"}]
    (root / "rehydrate_report.json").write_text(json.dumps(rep, indent=1))
    assert verify_data.main(["--root", str(root), "--profile", "test"]) == 1
    assert failures(root) == {"report": [f"{rf.contract_path('rs', P1)}: rehydrate status failed (digest_mismatch)"]}
    rep["files"], rep["abstracts"]["not_fetched"] = [], 2
    (root / "rehydrate_report.json").write_text(json.dumps(rep, indent=1))
    assert verify_data.main(["--root", str(root), "--profile", "test"]) == 1
    assert failures(root) == {"report": ["2 abstracts were not fetched (rerun biolitbench/release/fetch_abstracts.py)"]}
    del rep["abstracts"]["not_fetched"]
    (root / "rehydrate_report.json").write_text(json.dumps(rep, indent=1))
    assert verify_data.main(["--root", str(root), "--profile", "test"]) == 1
    assert "no abstracts.not_fetched count" in failures(root)["report"][0]
    rep["schema"] = "other/1"
    (root / "rehydrate_report.json").write_text(json.dumps(rep))
    assert verify_data.main(["--root", str(root), "--profile", "test"]) == 1
    assert "schema" in failures(root)["report"][0]


def test_the_board_reads_the_shared_tables_of_a_prepared_root(tmp_path, monkeypatch):
    import importlib.util
    snap, work = build(tmp_path)
    root = tmp_path / "root"
    assert prepare_root.main(["--snapshot", str(snap), "--profile", "test", "--root", str(root)]) == 0
    env = dict(line.split("=", 1) for line in (root / "site.env").read_text().splitlines() if not line.startswith("#"))
    (root / "scores/rollouts/newkey").mkdir(parents=True)
    put(root / "scores/_embed_cache/a.npy", b"x")
    monkeypatch.setenv("CCBENCH_SHARED_OUT", env["CCBENCH_SHARED_OUT"])
    spec = importlib.util.spec_from_file_location("board_view_builder_root", REL.parents[1] / "evaluation/board/view_builder.py")
    vb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vb)
    monkeypatch.setattr(vb, "bundle_entry_root", lambda: tmp_path / "absent")
    st = vb.shared_state()
    assert vb.SHARED_OUT == root / "scores" and st["rollout_keys"] == ["newkey"] and st["embed_cache_files"] == 1


def gate_lines():
    s = (Path(__file__).resolve().parents[1] / "scribe" / "launchers" / "run_fixed_input.slurm").read_text().splitlines()
    i = next(n for n, line in enumerate(s) if line.startswith("DATA_ROOT="))
    return "\n".join(s[i:i + 3])


@pytest.mark.parametrize("profile", ["test", "train"])
def test_the_fixed_input_launcher_gates_on_verify_data(tmp_path, test_set, profile):
    import subprocess
    snap, work = build(tmp_path)
    root = make_root(tmp_path, snap, work, profile)
    py = tmp_path / "run_py"
    put(py, f"#!/bin/sh\nexec {sys.executable} -c 'import sys; sys.path.insert(0, sys.argv[1].rsplit(\"/\", 1)[0]); "
            f"import verify_data; verify_data.TEST_SET_SHA16 = \"{verify_data.TEST_SET_SHA16}\"; "
            f"sys.exit(verify_data.main(sys.argv[2:]))' \"$@\"\n")
    py.chmod(0o755)
    env = {"PATH": os.environ["PATH"], "RUNS": str(root / "runs"), "RUN_PY": str(py),
           "LITREVIEW_ROOT": str(Path(__file__).resolve().parents[1]), "PYTHONDONTWRITEBYTECODE": "1"}
    run = lambda: subprocess.run(["bash", "-c", gate_lines() + '\necho "gate passed $DATA_PROFILE"'], env=env,
                                 capture_output=True, text=True)
    r = run()
    assert r.returncode == 0 and f"gate passed {profile}" in r.stdout
    assert json.loads((root / "verify_report.json").read_text())["profile"] == profile
    for f in (root / "scores").rglob("*"):
        if f.is_file():
            f.write_text("rescored")
    assert run().returncode == 0
    (root / rf.contract_path("grounding", P1 if profile == "test" else R1)).unlink()
    r = run()
    assert r.returncode == 90 and "FAIL closure" in r.stdout and "gate passed" not in r.stdout
