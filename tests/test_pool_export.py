"""Tests the pool corpus exporter against a fake GLKB query function: admission rule, sharding,
counters and resume."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "biolitbench" / "pool" / "export" / "export_pool_corpus.py"


def load():
    spec = importlib.util.spec_from_file_location("export_pool_corpus_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROWS = [
    {"pmid": "100", "title": "a", "abstract": "x" * 250, "year": 2020, "doi": "d1"},
    {"pmid": "101", "title": "b", "abstract": None, "year": 2020, "doi": None},
    {"pmid": "102", "title": "c", "abstract": "x" * 200, "year": 2019, "doi": None},
    {"pmid": "103", "title": "d", "abstract": "x" * 300, "year": 2026, "doi": None},
    {"pmid": "104", "title": "e", "abstract": "x" * 300, "year": "2001", "doi": None},
    {"pmid": "105", "title": "f", "abstract": "y" * 201, "year": 2025, "doi": "d2"},
    {"pmid": "106", "title": "g", "abstract": "z" * 400, "year": 1990, "doi": None},
]


def fake_q(rows, page):
    def q(statement, parameters, timeout=120):
        assert "n.pubmedid > $last" in statement and parameters["lim"] == page
        return [r for r in rows if r["pmid"] > parameters["last"]][:page]
    return q


def run(mod, out, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export_pool_corpus.py", "--out", str(out)])
    mod.main()


def read_shards(out):
    rows = []
    for p in sorted(out.glob("corpus_*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            rows += [json.loads(line) for line in f]
    return rows


def test_requires_glkb_environment(monkeypatch, tmp_path):
    mod = load()
    for k in mod.GLKB_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit):
        run(mod, tmp_path / "out", monkeypatch)
    assert "https://" not in SRC.read_text() and "password" not in SRC.read_text()


def test_admission_rule_shards_and_counters(monkeypatch, tmp_path):
    mod = load()
    for k in mod.GLKB_ENV:
        monkeypatch.setenv(k, "x")
    monkeypatch.setattr(mod, "PAGE", 3)
    monkeypatch.setattr(mod, "SHARD_ROWS", 2)
    monkeypatch.setattr(mod, "q", fake_q(ROWS, 3))
    out = tmp_path / "out"
    run(mod, out, monkeypatch)
    st = json.loads((out / "export_state.json").read_text())
    assert [r["pmid"] for r in read_shards(out)] == ["100", "105", "106"]
    assert (st["raw_seen"], st["kept"], st["excl_no_abs"], st["excl_short_abs"], st["excl_year"]) == (7, 3, 1, 1, 2)
    assert st["last_pmid"] == "106" and st["shard"] == 2
    assert not list(out.glob("*.part"))
    assert set(read_shards(out)[0]) == {"pmid", "title", "abstract", "year", "doi"}


def test_resume_continues_after_the_last_pmid(monkeypatch, tmp_path):
    mod = load()
    for k in mod.GLKB_ENV:
        monkeypatch.setenv(k, "x")
    monkeypatch.setattr(mod, "PAGE", 3)
    out = tmp_path / "out"
    monkeypatch.setattr(mod, "q", fake_q(ROWS[:3], 3))
    run(mod, out, monkeypatch)
    monkeypatch.setattr(mod, "q", fake_q(ROWS, 3))
    run(mod, out, monkeypatch)
    st = json.loads((out / "export_state.json").read_text())
    assert [r["pmid"] for r in read_shards(out)] == ["100", "105", "106"]
    assert (st["raw_seen"], st["kept"]) == (7, 3)
