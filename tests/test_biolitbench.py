"""Tests the BioLitBench builders and the pool on synthetic data: the eight relation types, the NLI
grounding rule, top-level hard negatives, the strict cutoff, the exclusion of the evaluated review,
and the fail-closed tap."""

import gzip
import importlib.util
import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BB = ROOT / "biolitbench"
POOL = BB / "pool"
TAP = POOL / "embed" / "tap"
if str(POOL) not in sys.path:
    sys.path.insert(0, str(POOL))
import pool_common


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GRAPH = load(BB / "build" / "biolitreview_bench" / "extract" / "build_enhanced_graph.py", "bb_graph")
PAPER_TYPES = ["parallel", "contrast", "elaboration", "evidence", "mechanism", "context", "synthesis",
               "condition"]


def test_relation_labels_are_the_eight_paper_types_plus_unrelated():
    assert GRAPH.RELATION_TYPES == PAPER_TYPES
    assert GRAPH.LABELS == PAPER_TYPES + ["unrelated"]
    assert GRAPH.SCHEMA["properties"]["relation_type"]["enum"] == GRAPH.LABELS
    for t in GRAPH.LABELS:
        assert f"- {t}:" in GRAPH.FEWSHOT and f"-> {t}\n" in GRAPH.FEWSHOT, t
    assert "other_relation" not in GRAPH.PROMPT + GRAPH.FEWSHOT


def write_paper(bench, name):
    d = bench / name
    d.mkdir(parents=True)
    claims = [{"claim_id": "c1", "sentence": "S1", "section_id": "s1a", "paragraph_id": "p1", "reference_ids": [1, 2]},
              {"claim_id": "c2", "sentence": "S2", "section_id": "s2", "paragraph_id": "p2", "reference_ids": [3]},
              {"claim_id": "c3", "sentence": "S3", "section_id": "s1b", "paragraph_id": "p3", "reference_ids": []},
              {"claim_id": "c4", "sentence": "S4", "section_id": "s1", "paragraph_id": "p4", "reference_ids": [4]}]
    (d / "claims_llm.jsonl").write_text("".join(json.dumps(c) + "\n" for c in claims))
    sections = [{"section_id": "s1", "title": "Intro", "parent_section_id": None},
                {"section_id": "s1a", "title": "2.1 Detail", "parent_section_id": "s1"},
                {"section_id": "s1b", "title": "3 Deeper", "parent_section_id": "s1a"},
                {"section_id": "s2", "title": "Methods", "parent_section_id": None}]
    (d / "review_structure_llm.json").write_text(json.dumps({"sections": sections}))


def test_nli_grounding_and_assembly(tmp_path, monkeypatch):
    monkeypatch.setattr(GRAPH, "BENCH", tmp_path / "bench")
    write_paper(tmp_path / "bench", "P")
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "P.json").write_text(json.dumps({"1": {"abstract": "A1"}, "2": {"abstract": "A2"},
                                             "3": {"abstract": "A3"}, "4": {"abstract": None}}))
    cls = {"A1": "neutral", "A2": "entailment", "A3": "contradiction"}
    monkeypatch.setattr(GRAPH, "nli_classes", lambda pairs: [cls[p] for p, _ in pairs])
    gdir = tmp_path / "grounding"
    GRAPH.step_verify("P", refs, gdir)
    g = json.loads((gdir / "P.json").read_text())
    assert g["c1"] == {"entail": 1.0, "entail_ref": "2", "contradict": 0.0, "contradict_ref": None, "n_refs": 2}
    assert g["c2"]["entail"] == 0.0 and g["c2"]["contradict"] == 1.0
    assert set(g) == {"c1", "c2"}

    work = tmp_path / "work"
    work.mkdir()
    (work / "weights.json").write_text(json.dumps({"1": 2, "3": 1}))
    pair = lambda a, b, rt: {"source_id": a, "target_id": b, "relation_type": rt, "cosine": 0.6}
    (work / "labeled.json").write_text(json.dumps([
        pair("c1", "c3", "unrelated"), pair("c1", "c2", "unrelated"), pair("c2", "c3", "evidence"),
        pair("c3", "c4", "other_relation"), pair("c1", "c4", None)]))
    GRAPH.step_assemble("P", work, gdir)
    graph = json.loads((work / "graph_enhanced.json").read_text())
    assert set(graph) == {"paper", "n_claims", "reference_weights", "claims", "review_level_edges", "hard_negatives"}
    assert [(e["source_id"], e["target_id"]) for e in graph["review_level_edges"]] == [("c2", "c3")]
    assert [(e["source_id"], e["target_id"]) for e in graph["hard_negatives"]] == [("c1", "c2")]
    gr = {c["claim_id"]: c["groundedness"] for c in graph["claims"]}
    assert gr == {"c1": 1.0, "c2": 0.0, "c3": None, "c4": None}


class _Ident:
    def stemWords(self, toks):
        return list(toks)


@pytest.fixture
def pool_index(tmp_path, monkeypatch):
    monkeypatch.setattr(pool_common, "make_stemmer", lambda: _Ident())
    bm25 = load(POOL / "build_bm25.py", "bb_bm25")
    monkeypatch.setattr(bm25, "make_stemmer", lambda: _Ident())
    docs = [("1", 2019, "gene regulation in yeast"), ("2", 2020, "gene regulation in yeast"),
            ("3", None, "gene regulation"), ("4", 2010, "gene regulation review"),
            ("5", 2021, "gene regulation"), ("6", 2015, "protein folding")]
    shard = tmp_path / "corpus" / "corpus_0000.jsonl.gz"
    shard.parent.mkdir()
    with gzip.open(shard, "wt") as f:
        for p, y, t in docs:
            f.write(json.dumps({"pmid": p, "title": t, "abstract": "", "year": y, "doi": ""}) + "\n")
    out = tmp_path / "bm25"
    monkeypatch.setattr(sys, "argv", ["build_bm25.py", "--shards", str(shard), "--out", str(out),
                                      "--max-df-frac", "1.0"])
    bm25.main()
    return out


def test_search_serves_only_dated_papers_before_the_cutoff(pool_index):
    ix = pool_common.PoolIndex(pool_index)
    hits, total = ix.search("gene regulation", cutoff=2020, k=10)
    assert sorted(h["pmid"] for h in hits) == ["1", "4"] and total == 2
    hits, total = ix.search("gene regulation", cutoff=2020, k=10, exclude=("4",))
    assert [h["pmid"] for h in hits] == ["1"] and total == 1
    hits, _ = ix.search("gene regulation", cutoff=2022, k=10, year_min=2019)
    assert sorted(h["pmid"] for h in hits) == ["1", "2", "5"]


def load_service(monkeypatch):
    fa = types.ModuleType("fastapi")

    class FastAPI:
        def __init__(self, **k):
            pass

        def get(self, *a, **k):
            return lambda f: f
        post = get

    class HTTPException(Exception):
        def __init__(self, status_code, detail=None):
            super().__init__(detail)
            self.status_code = status_code

    fa.FastAPI, fa.HTTPException, fa.Request = FastAPI, HTTPException, object
    fa.Query = lambda default=None, **k: default
    resp = types.ModuleType("fastapi.responses")
    resp.HTMLResponse = resp.JSONResponse = resp.Response = lambda *a, **k: (a, k)
    monkeypatch.setitem(sys.modules, "fastapi", fa)
    monkeypatch.setitem(sys.modules, "fastapi.responses", resp)
    return load(POOL / "pool_service.py", "bb_pool_service")


def write_edges(d, fwd, max_pmid=7):
    d.mkdir()
    bwd = {}
    for s, ts in fwd.items():
        for t in ts:
            bwd.setdefault(t, []).append(s)
    for name, adj in (("fwd", fwd), ("bwd", bwd)):
        indptr, val = [0], []
        for p in range(max_pmid + 1):
            val.extend(adj.get(p, []))
            indptr.append(len(val))
        np.save(d / f"{name}_indptr.npy", np.asarray(indptr, dtype=np.int64))
        np.save(d / f"{name}_val.npy", np.asarray(val, dtype=np.uint32))
    (d / "meta.json").write_text(json.dumps({"n_edges_input": sum(map(len, fwd.values()))}))


def test_pool_service_hides_the_review_and_every_paper_not_before_the_cutoff(pool_index, tmp_path, monkeypatch):
    svc = load_service(monkeypatch)
    write_edges(tmp_path / "edges", {1: [2, 4, 3, 6]})
    monkeypatch.setattr(svc, "EDGES_PATH", str(tmp_path / "edges"))
    (tmp_path / "cutoffs.json").write_text(json.dumps({"t1": 2020}))
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "t1.json").write_text(json.dumps({"review_pmid": "4"}))
    svc.ST = svc.State(pool_index, tmp_path / "cutoffs.json", gold, tmp_path / "log.jsonl", "http://x")
    hits, total = svc.ST.ranked("t1", "gene regulation", 10)
    assert [h["pmid"] for h in hits] == ["1"] and total == 1
    assert svc.visible_doc("t1", "4") is None and svc.visible_doc("t1", "2") is None
    assert svc.visible_doc("t1", "3") is None and svc.visible_doc("t1", "6")["pmid"] == "6"
    assert svc.visible_neighbours("t1", svc.ST.edges.refs(1), None).tolist() == [6]

    allow = tmp_path / "allow"
    allow.mkdir()
    (allow / "t1.json").write_text(json.dumps(["1", "3", "4", "6"]))
    monkeypatch.setattr(svc, "ALLOWLIST_DIR", str(allow))
    svc.ST = svc.State(pool_index, tmp_path / "cutoffs.json", gold, tmp_path / "log.jsonl", "http://x")
    hits, _ = svc.ST.ranked("t1", "gene regulation", 10)
    assert [h["pmid"] for h in hits] == ["1", "6"]

    (gold / "t1.json").write_text(json.dumps({"review_pmid": None}))
    with pytest.raises(SystemExit):
        svc.State(pool_index, tmp_path / "cutoffs.json", gold, tmp_path / "log.jsonl", "http://x")


def test_selector_views_exclude_the_review_and_fail_closed(tmp_path, monkeypatch):
    sel = load(TAP / "pool_selector.py", "bb_selector")
    assert sel._visible("201907.56", 2020, "55") and not sel._visible("201907.55", 2020, "55")
    assert not sel._visible("202007.56", 2020, "55") and not sel._visible("56", 2020, "55")
    view = tmp_path / "2020.t1.npy"
    np.save(view, np.asarray([1, 2], dtype="int64"))
    (tmp_path / "task_to_cutoff.json").write_text(json.dumps({"tasks": {"t1": {"cutoff": 2020, "review_pmid": "55"}}}))
    monkeypatch.setenv(sel.CUTOFF_ENV, str(view))
    assert sel._cutoff_year() == 2020 and sel._review_pmid() == "55"
    assert sel._cutoff_selector().is_member(2)
    fresh = load(TAP / "pool_selector.py", "bb_selector2")
    monkeypatch.setenv(fresh.CUTOFF_ENV, str(tmp_path / "2020.t2.npy"))
    with pytest.raises(RuntimeError):
        fresh._cutoff_selector()
    with pytest.raises(KeyError):
        fresh._review_pmid()


FAKE_DB = '''
import faiss
import numpy as np

TinyDB = object


class database:
    def __init__(self):
        self.abs_loaded_index = faiss.IndexIDMap(faiss.IndexFlatL2(2))
        self.abs_loaded_index.add_with_ids(np.eye(3, 2, dtype="float32"), np.arange(1, 4, dtype="int64"))
        self.title_loaded_index = self.abs_loaded_index
        self.index_to_id = {1: "a1", 2: "a2", 3: "a3"}

    def batch_search(self, query_vectors, top_k=1, title=False):
        return [["a%d" % i for i in range(1, top_k + 1)] for _ in query_vectors]

    def get_ids_from_queries(self, queries, num, shuffle=False):
        return self.batch_search(np.zeros((len(queries), 2), dtype="float32"), num)
'''


def run_tap(tmp_path, code, src=FAKE_DB, **env):
    app = tmp_path / "app"
    (app / "src").mkdir(parents=True, exist_ok=True)
    (app / "src" / "__init__.py").write_text("")
    (app / "src" / "database.py").write_text(src)
    full = {"PATH": "/usr/bin:/bin", "PYTHONPATH": f"{TAP}:{app}", "PYTHONDONTWRITEBYTECODE": "1", **env}
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=full, cwd=tmp_path,
                          timeout=120)


def cutoff_view(tmp_path):
    view = tmp_path / "views" / "2020.t1.npy"
    view.parent.mkdir()
    np.save(view, np.asarray([1, 3], dtype="int64"))
    (view.parent / "task_to_cutoff.json").write_text(json.dumps({"tasks": {"t1": {"cutoff": 2020, "review_pmid": "9"}}}))
    return view


def test_tap_logs_selected_searches_with_their_nesting_depth(tmp_path):
    log = tmp_path / "tap.jsonl"
    r = run_tap(tmp_path, "import src.database as d; print(d.database().get_ids_from_queries(['q1', 'q2'], 3))",
                RETRIEVAL_TAP_LOG=str(log), POOL_SELECTOR_CUTOFF_NPY=str(cutoff_view(tmp_path)))
    assert r.returncode == 0, r.stderr
    assert "[pool_docstore] patched 1 name(s) in src.database" in r.stderr
    assert "cutoff selector applied to src.database.database" in r.stderr
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert [(x["method"], x["depth"], x["n_ids"]) for x in rows] == [("batch_search", 1, 4),
                                                                     ("get_ids_from_queries", 0, 4)]
    assert set(rows[1]["ids"]) == {"a1", "a3"}


@pytest.mark.parametrize("view", [None, "missing.npy"])
def test_tap_ends_the_process_when_the_selector_cannot_engage(tmp_path, view):
    env = {} if view is None else {"POOL_SELECTOR_CUTOFF_NPY": str(tmp_path / view)}
    r = run_tap(tmp_path, "import src.database", RETRIEVAL_TAP_LOG=str(tmp_path / "tap.jsonl"), **env)
    assert r.returncode == 86 and "[retrieval_tap] FATAL" in r.stderr


def test_docstore_refuses_a_paper_db_without_its_sqlite_store(tmp_path):
    (tmp_path / "db").mkdir()
    r = run_tap(tmp_path, f"import src.database as d; d.TinyDB({str(tmp_path / 'db' / 'arxiv_paper_db.json')!r})",
                src="TinyDB = object\n")
    assert r.returncode != 0 and "docstore_meta.json" in r.stderr


def test_canonical_bundle_excludes_the_review_and_papers_not_before_the_cutoff(tmp_path, monkeypatch):
    B = load(BB / "tasks" / "build_canonical_bundles.py", "bb_canonical")
    for d in ("allow", "specs", "gold", "refs", "sents"):
        (tmp_path / d).mkdir()
    (tmp_path / "allow" / "t1.json").write_text(json.dumps(["10", "11", "12", "13", "14"]))
    (tmp_path / "specs" / "t1.json").write_text(json.dumps({"publication_cutoff": 2020,
                                                            "content_hash": "sha256:" + "0" * 64}))
    (tmp_path / "gold" / "t1.json").write_text(json.dumps({"review_pmid": "11"}))
    ry = tmp_path / "ref_years.json"
    ry.write_text(json.dumps({"10": 2018}))
    monkeypatch.setattr(B, "REF_YEARS", ry)
    B.ref_years.cache_clear()
    db = sqlite3.connect(":memory:")
    db.execute("create table meta (pmid integer, title text, year integer, abstract text)")
    db.executemany("insert into meta values (?,?,?,?)", [(10, "T10", 2018, "A10"), (11, "T11", 2019, "A11"),
                                                         (12, "T12", 2020, "A12"), (14, "T14", 2019, "A14")])
    ctx = B.SplitCtx("dev", allowlist_dir=tmp_path / "allow", taskspecs_dir=tmp_path / "specs",
                     gold_dir=tmp_path / "gold", refs_dir=tmp_path / "refs", sentences_dir=tmp_path / "sents",
                     out=tmp_path / "out")
    bundle, texts, row = B.build_task("t1", db, ctx)
    assert [(p["paper_id"], p["year"]) for p in bundle["papers"]] == [("10", 2018), ("14", 2019)]
    v = bundle["validation"]
    assert v["review_pmid_excluded"] == "11" and v["post_cutoff_excluded_pmids"] == ["12", "13"]
    assert v["n_allowlisted_file"] - v["n_review_pmid_excluded"] - v["n_post_cutoff_excluded"] == len(bundle["papers"])
    ry.write_text(json.dumps({"10": 2020}))
    B.ref_years.cache_clear()
    with pytest.raises(ValueError):
        B.build_task("t1", db, ctx)


def test_taskspec_follows_the_schema(tmp_path):
    M = load(BB / "tasks" / "make_taskspecs.py", "bb_taskspecs")
    mini = load(ROOT / "tests" / "harness" / "_mini_jsonschema.py", "bb_mini_jsonschema")
    rec = {"task_id": "pmcid_PMC1", "title": "Q", "topic": "T", "review_pmid": 7, "pub_year_min": 2020,
           "body_words": 5000}
    spec = M.build(rec, "pool-bm25-0123456789abcdef")
    schema = json.loads((BB / "schemas" / "task_spec.schema.json").read_text())
    assert list(mini.Draft202012Validator(schema).iter_errors(spec)) == []
    assert spec["publication_cutoff"] == 2020 and spec["budget"] == {"max_document_opens": 60}
    assert spec["scope"]["exclusion"] == ["the review under evaluation itself (PMID 7)"]
    (tmp_path / "meta.json").write_text("{}")
    assert M.snapshot_id(tmp_path).startswith("pool-bm25-") and len(M.snapshot_id(tmp_path)) == 26


def test_splits_are_paper_disjoint_and_cover_the_pool(tmp_path, monkeypatch):
    S = load(BB / "build" / "make_splits.py", "bb_splits")
    ds = tmp_path / "ds"
    for p in ("a", "b", "c", "d"):
        (ds / "graphs" / p).mkdir(parents=True)
    (ds / "writing_usable.txt").write_text("a b c\n")
    (ds / "retrieval_usable.txt").write_text("a\n")
    (ds / "manifests").mkdir()
    (ds / "manifests" / "domain_index.json").write_text(json.dumps({"x": {"papers": ["a", "b"]}, "y": {"papers": ["c"]}}))
    monkeypatch.setattr(S, "HELD_OUT", 2)
    monkeypatch.setattr(sys, "argv", ["make_splits.py", "--dataset", str(ds), "--out", str(tmp_path / "m")])
    S.main()
    rd = lambda n: [json.loads(l)["task_id"] for l in (tmp_path / "m" / n).read_text().splitlines()]
    dev, train = rd("biolit_dev.jsonl"), rd("biolit_train.jsonl")
    assert len(dev) == 2 and not set(dev) & set(train) and set(dev) | set(train) == {"a", "b", "c", "d"}
    assert "c" in dev and "d" in train


def test_embedding_layout_feeds_assemble_db_and_the_cutoff_views(tmp_path):
    sys.path.insert(0, str(POOL / "embed"))
    try:
        C = load(POOL / "embed" / "embed_common.py", "embed_common")
        sys.modules["embed_common"] = C
        A = load(POOL / "embed" / "assemble_db.py", "bb_assemble")
        V = load(POOL / "embed" / "cutoff_views.py", "bb_views")
    finally:
        sys.path.remove(str(POOL / "embed"))
    recs = [{"pmid": "11", "title": "t11", "abstract": "a11", "year": 2019, "doi": ""},
            {"pmid": "22", "title": "t22", "abstract": "a22", "year": 2021, "doi": ""},
            {"pmid": "33", "title": "t33", "abstract": "a33", "year": 2015, "doi": ""}]
    shard = tmp_path / "corpus" / "corpus_0000.jsonl.gz"
    shard.parent.mkdir()
    with gzip.open(shard, "wt") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    meta = tmp_path / "meta"
    meta.mkdir()
    with gzip.open(meta / "meta_000.jsonl.gz", "wt") as f:
        f.writelines(json.dumps({"pmid": p, "n_citation": 3}) + "\n" for p in ("11", "22", "33"))
    emb = tmp_path / "emb"
    for model in ("nomic", "gte"):
        for field in C.FIELDS:
            d = emb / f"{model}_{field}"
            d.mkdir(parents=True)
            np.save(d / C.emb_name(field, model, 0), np.eye(3, C.SPECS[model]["dim"], dtype="float32"))
            (d / C.pmids_name(0)).write_text(json.dumps({"pmids": [r["pmid"] for r in recs]}))
    out = tmp_path / "db"
    args = types.SimpleNamespace(system="both", emb_root=str(emb), shards=str(shard), out=str(out),
                                 meta_dir=str(meta), survey_ids=None)
    assert A.build(args) == 0
    assert np.load(out / "id_pmid.npy").tolist() == [11, 22, 33]
    assert np.load(out / "id_year.npy").tolist() == [2019, 2021, 2015]
    idmap = json.loads((out / "surveyforge_db" / C.ID_MAP_NAME).read_text())
    assert idmap == {"201907.11": 1, "202107.22": 2, "201507.33": 3}
    (tmp_path / "cut.json").write_text(json.dumps({"t1": 2021}))
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "t1.json").write_text(json.dumps({"review_pmid": "33"}))
    V.build(str(tmp_path / "cut.json"), str(gold), str(out), str(tmp_path / "views"))
    assert np.load(tmp_path / "views" / "2021.t1.npy").tolist() == [1]
