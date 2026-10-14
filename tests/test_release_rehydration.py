"""Tests the release parser and rehydration on a synthetic snapshot: round trip, gate refusal, drift record."""

import copy
import gzip
import json
import multiprocessing
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
REL = Path(__file__).resolve().parents[1] / "biolitbench" / "release"
if str(REL) not in sys.path:
    sys.path.insert(0, str(REL))
import fetch_abstracts
import fetch_articles
import parse_jats
import rehydrate
import release_format as rf

T, P, NR = "pmcid_PMC9000001", "pmcid_PMC9000002", "pmcid_PMC9000003"
HEAD = (b'<?xml version="1.0"  ?><!DOCTYPE pmc-articleset PUBLIC "-//NLM//DTD ARTICLE SET 2.0//EN" '
        b'"https://dtd.nlm.nih.gov/ncbi/pmc/articleset/nlm-articleset-2.0.dtd"><pmc-articleset>')
TAIL = b"</pmc-articleset>"


def article(pmcid, ver, word):
    return (f'<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="review-article"><front><article-meta>'
            f'<article-id pub-id-type="pmcid">{pmcid}</article-id>'
            f'<article-id pub-id-type="pmcid-ver">{pmcid}.{ver}</article-id>'
            f'<title-group><article-title>Synthetic review {word}</article-title></title-group>'
            f'<contrib-group><contrib contrib-type="author"><name><surname>Alpha</surname>'
            f'<given-names>A</given-names></name></contrib></contrib-group>'
            f'<abstract><p>Alpha {word} gamma delta. Epsilon zeta eta.</p></abstract></article-meta></front><body>'
            f'<sec><title>Introduction {word}</title><p>Token one {word} is linked to token two '
            f'[<xref ref-type="bibr" rid="r1">1</xref>]. Token three follows from token four '
            f'[<xref ref-type="bibr" rid="r1 r2">1,2</xref>].</p>'
            f'<fig id="f1"><caption><p>Figure caption {word}.</p></caption></fig></sec>'
            f'<sec><title>Discussion</title><p>Token five explains token six '
            f'[<xref ref-type="bibr" rid="r2">2</xref>]. Plain closing words here.</p></sec></body>'
            f'<back><ref-list><ref id="r1"><mixed-citation><article-title>Ref title one</article-title>'
            f'<pub-id pub-id-type="pmid">111</pub-id></mixed-citation></ref><ref id="r2"><mixed-citation>'
            f'<source>Ref source two</source><pub-id pub-id-type="doi">10.1/x</pub-id></mixed-citation></ref>'
            f'</ref-list></back></article>').encode()


def abstract_record(pmid, parts):
    texts = "".join(f'<AbstractText{" Label=" + chr(34) + lab + chr(34) if lab else ""}>{txt}</AbstractText>'
                    for lab, txt in parts)
    abstract = f"<Abstract>{texts}</Abstract>" if parts else ""
    return (f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><ArticleTitle>Record {pmid}"
            f"</ArticleTitle>{abstract}</Article></MedlineCitation></PubmedArticle>")


RECORDS = {"101": [(None, "Kappa lambda mu. Nu xi omicron. Pi rho sigma.")],
           "102": [("BACKGROUND", "Tau upsilon phi."), ("RESULTS", "Chi psi; omega &quot;q&quot;.")],
           "103": [(None, "Alef bet gimel. Dalet he vav.")]}


def abstract_set_xml(records):
    return ("<?xml version=\"1.0\" ?><PubmedArticleSet>" + "".join(abstract_record(p, v) for p, v in records.items())
            + "</PubmedArticleSet>").encode()


def canonical(pmid):
    return fetch_abstracts.render_rule([(lab, txt.replace("&quot;", '"')) for lab, txt in RECORDS[pmid]])


def texts_and_oracle():
    a101, a102, a103 = canonical("101"), canonical("102"), canonical("103")
    oracle = {"pmid101_0": "Kappa lambda mu.", "pmid101_1": "Nu xi omicron.", "pmid101_2": "Pi rho sigma.",
              "pmid103_0": "Title three", "pmid103_1": "Alef bet gimel.", "pmid103_2": "Dalet he vav."}
    join103 = " ".join(oracle[f"pmid103_{k}"] for k in range(3))
    texts = {"101": {"title": "Title one.", "abstract": a101,
                     "sentences": {"pmid101_0": "Kappa lambda mu.", "pmid101_1": "Nu xi omicron.",
                                   "pmid101_2": "Pi rho sigma."}},
             "102": {"title": "Title two", "abstract": a102,
                     "sentences": {"pmid102_0": "Title two", "pmid102_1": "BACKGROUND Tau upsilon phi."}},
             "103": {"title": "Title three", "abstract": join103,
                     "sentences": {"pmid103_1": "Alef bet gimel.", "pmid103_2": "Dalet he vav."}},
             "105": {"title": "No abstract", "abstract": None, "sentences": {}}}
    baseline = {"topic": "Synthetic review one", "n_refs": 3, "task_id": T,
                "refs": [{"pmid": "101", "doi": None, "title": "Title one.", "abstract": a101},
                         {"pmid": "103", "doi": None, "title": "Title three", "abstract": join103},
                         {"pmid": "105", "doi": None, "title": "No abstract", "abstract": None}]}
    return texts, oracle, baseline, {"101": a101, "102": a102, "103": a103}, join103


def bundle(units):
    return {"schema_version": "x", "task_id": T,
            "evidence": [{"evidence_id": f"e{i}", "paper_id": pid, "granularity": g, "locator": loc,
                          "text_hash": rf.text_hash(txt), "condition": None}
                         for i, (pid, g, loc, txt) in enumerate(units)]}


def dehydrate_rs(rs):
    d = copy.deepcopy(rs)
    for p in d["paragraphs"]:
        sents = p["sentences"]
        p["text"] = rf.text_placeholder(p["text"])
        p["sentences"] = rf.sentences_placeholder(sents)
    for s in d["sections"]:
        s["title"] = rf.text_placeholder(s["title"])
    return d


def dehydrate_graph(g):
    d = copy.deepcopy(g)
    for c in d["claims"]:
        c["citation_markers"] = rf.marker_spans(c["sentence"], c["citation_markers"])
        c["sentence"] = None
    for name in rf.EDGE_LISTS:
        for e in d[name]:
            e["source_text"] = e["target_text"] = None
    return d


def dehydrate_abstract_kinds(texts, oracle, baseline, abstracts):
    o = {}
    for loc, s in oracle.items():
        a = abstracts[rf.oracle_locator(loc)[0]]
        i = a.find(s)
        o[loc] = [i, i + len(s)] if i >= 0 else s
    tx = copy.deepcopy(texts)
    for pid, e in tx.items():
        src = e["abstract"]
        if src is None:
            continue
        e["abstract"] = ({"@": "pm"} if src == abstracts.get(pid)
                         else {"@": "oracle_join", **rf.text_placeholder(src)})
        for loc, s in e["sentences"].items():
            i = src.find(s)
            e["sentences"][loc] = ["a", i, i + len(s)] if i >= 0 else ["t", e["title"].find(s),
                                                                         e["title"].find(s) + len(s)]
    bl = copy.deepcopy(baseline)
    for r in bl["refs"]:
        if r["abstract"] is not None:
            tag = "pm" if r["abstract"] == abstracts.get(r["pmid"]) else "oracle_join"
            r["abstract"] = {"@": tag, **rf.text_placeholder(r["abstract"])}
    return tx, o, bl


def graph_of(rs, claims):
    cl = [{"claim_id": c["claim_id"], "paragraph_id": c["paragraph_id"], "section_id": c["section_id"],
           "sentence_index": c["sentence_index"], "sentence": c["sentence"],
           "citation_markers": c["citation_markers"], "reference_ids": c["reference_ids"]} for c in claims]
    s = {c["claim_id"]: c["sentence"] for c in cl}
    edge = lambda a, b, **kw: {"source_id": a, "target_id": b, "source_text": s[a], "target_text": s[b], **kw}
    return {"paper_id": "x", "claims": cl,
            "review_level_edges": [edge("claim_0000", "claim_0002", relation_type="evidence")],
            "restatement_edges": [], "hard_negatives": [edge("claim_0001", "claim_0002")],
            "local_edges": [edge("claim_0000", "claim_0001")]}


def put(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def tsv(path, header, rows):
    put(path, ("\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows)).encode())


def make_snapshot(tmp):
    snap, cache, pcache = tmp / "hf", tmp / "cache_articles", tmp / "cache_abstracts"
    arts = {T: article("PMC9000001", 2, "one"), P: article("PMC9000002", 1, "two"), NR: article("PMC9000003", 1, "nr")}
    put(cache / "batch_000.xml.gz", gzip.compress(HEAD + arts[T] + arts[P] + TAIL))
    put(cache / "batch_000.ids", b"PMC9000001\nPMC9000002\n")
    put(pcache / "batch_000.xml.gz", gzip.compress(abstract_set_xml(RECORDS)))
    put(pcache / "batch_000.ids", b"101\n102\n103\n")
    expected, originals = {}, []
    for t, prof in ((T, "test"), (P, "test,train"), (NR, "train")):
        x = tmp / "src" / (t + ".xml")
        put(x, HEAD + arts[t] + TAIL)
        rs, claims, _ = parse_jats.parse(str(x))
        for r in rs["paper"]["references"]:
            del r["raw"]
        g = graph_of(rs, claims)
        src = {"pmcid": t[6:], "pmcid_ver": "", "rehydratable": t != NR}
        for kind, obj, d, src in (("rs", rs, dehydrate_rs(rs), src), ("graph", g, dehydrate_graph(g), None)):
            path = rf.contract_path(kind, t)
            data = rf.dumps(obj, rf.SERIALIZATION[kind])
            expected[path] = data
            originals.append((path, kind, t, "rehydrate", "file", prof, rf.sha256_bytes(data)))
            wrapper = rf.make_wrapper(kind, t, path, rf.sha256_bytes(data), d, src)
            put(snap / "dehydrated" / path, rf.dumps_wrapper(wrapper))
    texts, oracle, baseline, abstracts, join103 = texts_and_oracle()
    tx, o, bl = dehydrate_abstract_kinds(texts, oracle, baseline, abstracts)
    tunits = [(p, "abstract", f"pmid{p}", e["abstract"]) for p, e in texts.items() if e["abstract"] is not None]
    tunits += [(p, "abstract_sentence", loc, s) for p, e in texts.items() for loc, s in e["sentences"].items()]
    sealed = {"bundle_test": bundle(tunits),
              "oracle_bundle": bundle([(rf.oracle_locator(k)[0], "abstract_sentence", k, v)
                                       for k, v in oracle.items()])}
    for kind, obj in sealed.items():
        path = rf.contract_path(kind, T)
        put(snap / "store" / path, json.dumps(obj).encode())
        originals.append((path, kind, T, "store", "file", "test", rf.sha256_bytes(json.dumps(obj).encode())))
    for kind, obj, d in (("oracle_sentences", oracle, o), ("texts_test", texts, tx), ("baseline_refs", baseline, bl)):
        path = rf.contract_path(kind, T)
        data = rf.dumps(obj, rf.SERIALIZATION[kind])
        expected[path] = data
        originals.append((path, kind, T, "rehydrate", "unit", "test", rf.sha256_bytes(data)))
        put(snap / "dehydrated" / path, rf.dumps_wrapper(rf.make_wrapper(kind, T, path, rf.sha256_bytes(data), d)))
    tsv(snap / "manifests" / "original_sha256.tsv", rf.ORIGINAL_COLUMNS, sorted(originals))
    tsv(snap / "manifests" / "abstract_sha256.tsv", rf.ABSTRACT_COLUMNS,
        [(p, rf.sha256_hex(a), len(a), "test") for p, a in sorted(abstracts.items(), key=lambda kv: int(kv[0]))])
    tsv(snap / "manifests" / "article_sources.tsv", rf.SOURCE_COLUMNS,
        [(T, "PMC9000001", "PMC9000001.2", "true", "test"), (P, "PMC9000002", "PMC9000002.1", "true", "test,train"),
         (NR, "PMC9000003", "PMC9000003.1", "false", "train")])
    return snap, cache, pcache, expected


def run_user_side(tmp, snap, cache, pcache, profile="test"):
    work, root = tmp / "work", tmp / "root"
    assert fetch_articles.main(["--snapshot", str(snap), "--profile", profile, "--work", str(work),
                                "--cache-dir", str(cache), "--offline"]) == 0
    assert parse_jats.main(["--xml", str(work / "xml"), "--out", str(work / "articles")]) == 0
    fetch_abstracts.main(["--snapshot", str(snap), "--profile", profile, "--work", str(work),
                          "--cache-dir", str(pcache), "--offline"])
    code = rehydrate.main(["--snapshot", str(snap), "--root", str(root), "--profile", profile,
                           "--articles", str(work / "articles"), "--abstracts", str(work / "abstracts.jsonl")])
    return code, work, root, json.loads((root / "rehydrate_report.json").read_text())


def test_parser_output_is_deterministic_across_containers_and_runs(tmp_path):
    a = article("PMC9000001", 2, "one")
    put(tmp_path / "single.xml", HEAD + a + TAIL)
    put(tmp_path / "single.xml.gz", gzip.compress(HEAD + a + TAIL))
    put(tmp_path / "bare.xml", b'<?xml version="1.0"?>' + a)
    outs = [parse_jats.serialize(*parse_jats.parse(str(tmp_path / n))) for n in ("single.xml", "single.xml.gz",
                                                                                 "bare.xml", "single.xml")]
    assert all(o == outs[0] for o in outs)
    rs = json.loads(outs[0]["review_structure_llm.json"])
    assert all(p["text"] == " ".join(p["sentences"]) for p in rs["paragraphs"])
    assert rs["paragraphs"][1]["text"].endswith("[[1,2]].") and rs["paragraphs"][2]["role"] == "caption"
    assert [s["title"] for s in rs["sections"]] == ["Abstract", "Introduction one", "Discussion"]
    assert [c["reference_ids"] for c in map(json.loads, outs[0]["claims_llm.jsonl"].splitlines())] == [[1], [1, 2], [2]]
    assert outs[0]["relations_llm.jsonl"] == ""
    assert parse_jats.main(["--xml", str(tmp_path / "single.xml"), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "single" / "review_structure_llm.json").read_text() == outs[0]["review_structure_llm.json"]


def test_rehydration_restores_every_kind_byte_exactly(tmp_path):
    snap, cache, pcache, expected = make_snapshot(tmp_path)
    code, work, root, rep = run_user_side(tmp_path, snap, cache, pcache)
    assert code == 0
    for path, data in expected.items():
        if NR in path:
            assert not (root / path).exists()
        else:
            assert (root / path).read_bytes() == data, path
    assert rep["schema"] == rf.REPORT_SCHEMA and rep["profile"] == "test" and rep["files"] == [] and rep["drift"] == []
    assert rep["files_by_kind"]["rs"]["ok"] == 2 and rep["files_by_kind"]["texts_test"]["ok"] == 1
    assert rep["abstracts"] == {"expected": 3, "verified": 3, "drifted": 0, "missing": 0, "not_fetched": 0}
    assert rep["manifest_sha256"] == rf.file_sha256(snap / "manifests" / "original_sha256.tsv")
    assert rep["parser_sha256"] == rf.file_sha256(REL / "parse_jats.py")
    for wrapper in (snap / "dehydrated").rglob("*.json"):
        w = json.loads(wrapper.read_text())
        assert "Kappa" not in wrapper.read_text() and "Token one" not in wrapper.read_text()
        assert rf.is_wrapper(w) and not (snap / "store" / w["path"]).exists()
    assert "Kappa" not in json.dumps(rep) and "Token" not in json.dumps(rep)


def test_parallel_workers_under_spawn_give_the_same_files_and_report(tmp_path, monkeypatch):
    snap, cache, pcache, expected = make_snapshot(tmp_path)
    code, work, root, rep = run_user_side(tmp_path, snap, cache, pcache)
    monkeypatch.setattr(rehydrate, "Pool", multiprocessing.get_context("spawn").Pool)
    root2 = tmp_path / "root2"
    assert rehydrate.main(["--snapshot", str(snap), "--root", str(root2), "--profile", "test", "--workers", "2",
                           "--articles", str(work / "articles"), "--abstracts", str(work / "abstracts.jsonl")]) == 0
    rep2 = json.loads((root2 / "rehydrate_report.json").read_text())
    for key in ("files_by_kind", "abstracts", "units", "files", "drift"):
        assert rep2[key] == rep[key]
    for path, data in expected.items():
        if NR not in path:
            assert (root2 / path).read_bytes() == data


def test_train_profile_marks_structure_only_articles_and_restores_the_rest(tmp_path):
    snap, cache, pcache, expected = make_snapshot(tmp_path)
    code, work, root, rep = run_user_side(tmp_path, snap, cache, pcache, profile="train")
    assert code == 0
    path = rf.contract_path("rs", P)
    assert (root / path).read_bytes() == expected[path]
    assert rep["files_by_kind"]["rs"] == {"ok": 1, "drift": 0, "failed": 0, "not_rehydratable": 1}
    assert [(f["task_id"], f["status"]) for f in rep["files"]] == [(NR, "not_rehydratable"), (NR, "not_rehydratable")]
    assert not (root / rf.contract_path("graph", NR)).exists()


def test_gate_refuses_a_changed_article_and_never_writes_it(tmp_path):
    snap, cache, pcache, expected = make_snapshot(tmp_path)
    work = tmp_path / "work"
    fetch_articles.main(["--snapshot", str(snap), "--profile", "test", "--work", str(work), "--cache-dir", str(cache),
                         "--offline"])
    parse_jats.main(["--xml", str(work / "xml"), "--out", str(work / "articles")])
    art = work / "articles" / T / "review_structure_llm.json"
    rs = json.loads(art.read_text())
    rs["paragraphs"][1]["text"] = rs["paragraphs"][1]["text"].replace("Token one", "Token uno")
    art.write_text(json.dumps(rs))
    w = snap / "dehydrated" / rf.contract_path("graph", P)
    wrapper = json.loads(w.read_text())
    wrapper["task_id"] = T
    w.write_bytes(rf.dumps_wrapper(wrapper))
    root = tmp_path / "root"
    (work / "none.jsonl").write_text("")
    code = rehydrate.main(["--snapshot", str(snap), "--root", str(root), "--profile", "test",
                           "--articles", str(work / "articles"), "--abstracts", str(work / "none.jsonl")])
    rep = json.loads((root / "rehydrate_report.json").read_text())
    assert code == 1
    reasons = {(f["task_id"], f["kind"]): f["reason"] for f in rep["files"]}
    assert reasons[(T, "rs")] == "paragraph_digest" and reasons[(T, "graph")] == "rs_unavailable"
    assert reasons[(P, "graph")] == "wrapper_mismatch"
    for kind, t in (("rs", T), ("graph", T), ("graph", P)):
        assert not (root / rf.contract_path(kind, t)).exists()
    assert (root / rf.contract_path("rs", P)).read_bytes() == expected[rf.contract_path("rs", P)]
    assert all(f["written_sha256"] is None for f in rep["files"] if f["status"] == "failed")


def test_a_unit_that_fails_while_its_abstract_verifies_fails_the_file(tmp_path):
    snap, cache, pcache, expected = make_snapshot(tmp_path)
    w = snap / "dehydrated" / rf.contract_path("texts_test", T)
    wrapper = json.loads(w.read_text())
    wrapper["data"]["101"]["sentences"]["pmid101_1"][1] += 1
    w.write_bytes(rf.dumps_wrapper(wrapper))
    code, work, root, rep = run_user_side(tmp_path, snap, cache, pcache)
    assert code == 1
    assert [(f["kind"], f["status"], f["reason"]) for f in rep["files"]] == [("texts_test", "failed", "unit_digest")]
    assert not (root / rf.contract_path("texts_test", T)).exists() and rep["drift"] == []


def test_abstracts_that_were_never_fetched_fail_their_files(tmp_path):
    snap, cache, pcache, expected = make_snapshot(tmp_path)
    put(pcache / "batch_000.ids", b"101\n103\n")
    put(pcache / "batch_000.xml.gz", gzip.compress(abstract_set_xml({k: v for k, v in RECORDS.items() if k != "102"})))
    code, work, root, rep = run_user_side(tmp_path, snap, cache, pcache)
    assert code == 1 and rep["abstracts"]["not_fetched"] == 1 and rep["drift"] == []
    assert [(f["kind"], f["status"], f["reason"]) for f in rep["files"]] == [("texts_test", "failed", "abstract_not_fetched")]
    assert not (root / rf.contract_path("texts_test", T)).exists()
    store = tmp_path / "keyless.jsonl"
    store.write_text('{"pmid": "102", "render": "glkb/1"}\n{"pmid": "103", "abstract": null}\n')
    assert rehydrate.load_abstract_store(str(store)) == {"103": None}


def test_rehydration_needs_the_abstract_store_of_the_profile(tmp_path):
    snap, cache, pcache, expected = make_snapshot(tmp_path)
    for extra in ([], ["--abstracts", str(tmp_path / "absent.jsonl")]):
        with pytest.raises(SystemExit) as e:
            rehydrate.main(["--snapshot", str(snap), "--root", str(tmp_path / "root"), "--profile", "test",
                            "--articles", str(tmp_path / "articles")] + extra)
        assert e.value.code == 2
    assert not (tmp_path / "root").exists()


def test_drifted_and_missing_abstracts_are_written_and_recorded(tmp_path):
    snap, cache, pcache, expected = make_snapshot(tmp_path)
    today = dict(RECORDS)
    today["101"] = [(None, "Kappa lambda mu. Nu xi omicron REVISED. Pi rho sigma.")]
    del today["102"]
    put(pcache / "batch_000.xml.gz", gzip.compress(abstract_set_xml(today)))
    code, work, root, rep = run_user_side(tmp_path, snap, cache, pcache)
    assert code == 0
    assert rep["abstracts"] == {"expected": 3, "verified": 1, "drifted": 1, "missing": 1, "not_fetched": 0}
    assert rep["files_by_kind"]["texts_test"]["drift"] == 1 and rep["files_by_kind"]["oracle_sentences"]["drift"] == 1
    assert rep["files_by_kind"]["baseline_refs"]["drift"] == 1 and rep["tasks_with_drift"] == 1
    texts = json.loads((root / rf.contract_path("texts_test", T)).read_text())
    assert texts["101"]["abstract"] == "Kappa lambda mu. Nu xi omicron REVISED. Pi rho sigma."
    assert texts["101"]["sentences"] == {"pmid101_0": "Kappa lambda mu.", "pmid101_1": "Nu xi omicron REVISED.",
                                         "pmid101_2": "Pi rho sigma."}
    assert texts["102"]["abstract"] is None and texts["102"]["sentences"] == {"pmid102_0": "Title two",
                                                                              "pmid102_1": ""}
    assert texts["103"] == json.loads(expected[rf.contract_path("texts_test", T)])["103"]
    got = {(d["kind"], d["locator"]): (d["status"], d["granularity"]) for d in rep["drift"]}
    assert got == {("texts_test", "pmid101"): ("drift", "abstract"),
                   ("texts_test", "pmid101_1"): ("drift", "abstract_sentence"),
                   ("texts_test", "pmid102"): ("missing", "abstract"),
                   ("texts_test", "pmid102_1"): ("missing", "abstract_sentence"),
                   ("oracle_sentences", "pmid101_1"): ("drift", "abstract_sentence"),
                   ("baseline_refs", "pmid101"): ("drift", "baseline_abstract")}
    assert rep["units"] == {"relocated": 4, "drift": 4, "missing": 2}
    assert rep["drift"] == sorted(rep["drift"], key=lambda d: (d["task_id"], d["path"], d["locator"]))
    for d in rep["drift"]:
        obj = json.loads((root / d["path"]).read_text())
        if d["kind"] == "texts_test":
            p = d["pmid"]
            v = obj[p]["abstract"] if d["granularity"] == "abstract" else obj[p]["sentences"][d["locator"]]
        elif d["kind"] == "oracle_sentences":
            v = obj[d["locator"]]
        else:
            v = next(r["abstract"] for r in obj["refs"] if r["pmid"] == d["pmid"])
        assert d["written_text_hash"] == (None if v is None else rf.text_hash(v))
        assert d["expected_text_hash"].startswith("sha256:") and d["expected_text_hash"] != d["written_text_hash"]
    files = {f["path"]: f for f in rep["files"]}
    for kind in ("texts_test", "oracle_sentences", "baseline_refs"):
        f = files[rf.contract_path(kind, T)]
        assert f["status"] == "drift" and f["written_sha256"] == rf.file_sha256(root / f["path"])
    assert "REVISED" not in json.dumps(rep)


@pytest.mark.parametrize("new,locator,want", [
    ("Alpha beta gamma, revised. Delta epsilon zeta. Eta theta iota.", "u1", "Alpha beta gamma, revised."),
    ("Alpha beta gamma. Delta EPSILON zeta now. Eta theta iota.", "u2", "Delta EPSILON zeta now."),
    ("Alpha beta gamma. Delta epsilon zeta. Eta theta iota kappa.", "u3", "Eta theta iota kappa.")])
def test_drift_span_rule_cuts_the_changed_sentence_exactly(new, locator, want):
    sents = ["Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota."]
    old = " ".join(sents)
    spans, a = [], 0
    for k, s in enumerate(sents, 1):
        spans.append((f"u{k}", a, a + len(s), rf.text_hash(s)))
        a += len(s) + 1
    out = rehydrate.resolve_spans(new, "drifted", len(old), spans)
    assert out[locator] == (want, "drift")
    assert sorted(st for _, st in out.values()) == ["drift", "relocated", "relocated"]
    assert rehydrate.resolve_spans(None, "missing", len(old), spans)["u2"] == ("", "missing")
    with pytest.raises(rehydrate.Refused):
        rehydrate.resolve_spans(new, "verified", len(old), spans)
