"""Tests the release fetch tools offline: rate limit, off-peak rule, backoff, pinned routes, cache mode."""

import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs

import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "biolitbench" / "release"
if str(REL) not in sys.path:
    sys.path.insert(0, str(REL))
import fetch_abstracts
import fetch_articles
import ncbi_client
import parse_jats
import release_format as rf
from test_release_rehydration import HEAD, TAIL, article, abstract_set_xml, put, tsv

PEAK = datetime(2026, 10, 7, 19, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, d):
        self.t += d


class Transport:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, data, headers, timeout):
        self.calls.append((method, url, parse_qs(data.decode()) if data else {}))
        for prefix, handler in self.routes:
            if url.startswith(prefix):
                return handler(url, parse_qs(data.decode()) if data else {})
        return 404, {}, b""


def client(transport=None, **kw):
    clock = Clock()
    kw.setdefault("now", lambda: PEAK)
    return ncbi_client.NcbiClient(transport=transport, limiter=ncbi_client.TokenBucket(clock.now, clock.sleep),
                                  **kw), clock


def test_one_token_bucket_spaces_all_requests_of_the_process():
    clock = Clock()
    bucket = ncbi_client.TokenBucket(clock.now, clock.sleep)
    stamps = []
    for _ in range(12):
        bucket.acquire(3.0)
        stamps.append(clock.t)
    assert min(b - a for a, b in zip(stamps, stamps[1:])) == pytest.approx(1 / 3)
    clock.sleep(5.0)
    assert bucket.acquire(10.0) == 0.0 and bucket.acquire(10.0) == pytest.approx(0.1)
    a, b = ncbi_client.NcbiClient(), ncbi_client.NcbiClient(api_key="k")
    assert a.limiter is b.limiter is ncbi_client.LIMITER
    assert (a.rate, b.rate) == (3.0, 10.0)
    t = Transport([("https://", lambda u, d: (200, {}, b"</PubmedArticleSet>"))])
    shared = ncbi_client.TokenBucket(clock.now, clock.sleep)
    c1 = ncbi_client.NcbiClient(transport=t, limiter=shared)
    c2 = ncbi_client.NcbiClient(transport=t, limiter=shared)
    start = clock.t
    for _ in range(3):
        c1.get("https://pmc-oa-opendata.s3.amazonaws.com/x")
        c2.eutils("efetch.fcgi", {"db": "pubmed", "id": "1"})
    assert clock.t - start == pytest.approx(5 / 3, abs=1e-6)


@pytest.mark.parametrize("when,ok", [
    (datetime(2026, 10, 7, 19, 0, tzinfo=timezone.utc), False),
    (datetime(2026, 10, 7, 9, 0, tzinfo=timezone.utc), False),
    (datetime(2026, 10, 7, 8, 59, tzinfo=timezone.utc), True),
    (datetime(2026, 10, 8, 1, 0, tzinfo=timezone.utc), True),
    (datetime(2026, 10, 10, 19, 0, tzinfo=timezone.utc), True),
    (datetime(2026, 12, 2, 1, 59, tzinfo=timezone.utc), False),
    (datetime(2026, 12, 2, 2, 0, tzinfo=timezone.utc), True)])
def test_a_series_of_more_than_100_requests_runs_only_off_peak(when, ok):
    c, _ = client(now=lambda: when)
    c.plan(100)
    if ok:
        c.plan(1)
    else:
        with pytest.raises(ncbi_client.PeakHoursError):
            c.plan(1)
    c, _ = client(now=lambda: when, allow_peak_hours=True)
    c.plan(10000)


def test_the_request_count_also_stops_a_long_series_at_peak_hours():
    c, _ = client(Transport([("https://", lambda u, d: (200, {}, b"</PubmedArticleSet>"))]))
    for _ in range(100):
        c.eutils("efetch.fcgi", {"db": "pubmed", "id": "1"})
    with pytest.raises(ncbi_client.PeakHoursError):
        c.eutils("efetch.fcgi", {"db": "pubmed", "id": "1"})


def test_requests_carry_tool_and_only_a_given_email_and_back_off_on_429_and_5xx():
    replies = iter([(429, {"Retry-After": "2"}, b""), (503, {}, b""), (200, {}, b"<x/>"), (200, {}, b"</ok>")])
    t = Transport([("https://", lambda u, d: next(replies))])
    sleeps = []
    c, _ = client(t, sleep=sleeps.append, backoff=1.0)
    assert c.eutils("efetch.fcgi", {"db": "pubmed", "id": "1,2"}, check=lambda b: b.endswith(b"</ok>")) == b"</ok>"
    assert sleeps == [2.0, 2.0, 4.0]
    method, url, body = t.calls[0]
    assert method == "POST" and url == ncbi_client.EUTILS + "efetch.fcgi"
    assert body["tool"] == ["biolitbench"] and "email" not in body and "api_key" not in body
    c, _ = client(t, email="user@example.org", api_key="k")
    replies = iter([(200, {}, b"<x/>")])
    c.eutils("efetch.fcgi", {"db": "pubmed", "id": "1"})
    assert t.calls[-1][2]["email"] == ["user@example.org"] and t.calls[-1][2]["api_key"] == ["k"]
    with pytest.raises(ValueError):
        c.eutils("efetch.fcgi", {"db": "pubmed", "id": ",".join(map(str, range(151)))})
    bad = Transport([("https://", lambda u, d: (400, {}, b""))])
    c, _ = client(bad, sleep=sleeps.append)
    with pytest.raises(ncbi_client.NcbiError):
        c.eutils("efetch.fcgi", {"db": "pubmed", "id": "1"})
    assert len(bad.calls) == 1


def sources(snap, rows):
    tsv(snap / "manifests" / "article_sources.tsv", rf.SOURCE_COLUMNS, rows)


def test_cache_mode_takes_only_the_pinned_version_and_skips_structure_only_articles(tmp_path):
    snap, cache, work = tmp_path / "hf", tmp_path / "cache", tmp_path / "work"
    sources(snap, [("pmcid_PMC9000001", "PMC9000001", "PMC9000001.2", "true", "test"),
                   ("pmcid_PMC9000002", "PMC9000002", "PMC9000002.2", "true", "test"),
                   ("pmcid_PMC9000003", "PMC9000003", "PMC9000003.1", "false", "test"),
                   ("pmcid_PMC9000004", "PMC9000004", "PMC9000004.1", "true", "test")])
    one = article("PMC9000001", 2, "one")
    put(cache / "batch_000.xml.gz", gzip.compress(HEAD + one + article("PMC9000002", 1, "old") + TAIL))
    put(cache / "batch_000.ids", b"PMC9000001\nPMC9000002\n")
    put(cache / "single" / "pmcid_PMC9000004.xml", HEAD + article("PMC9000004", 1, "four") + TAIL)
    args = ["--snapshot", str(snap), "--profile", "test", "--work", str(work), "--cache-dir", str(cache), "--offline"]
    assert fetch_articles.main(args) == 1
    rep = json.loads((work / "fetch_articles_report.json").read_text())
    assert rep["by_route"] == {"cache": 2} and rep["n_tasks"] == 3
    assert rep["failed"] == [{"task_id": "pmcid_PMC9000002", "pmcid_ver": "PMC9000002.2",
                              "reason": "offline_not_in_cache"}]
    assert (work / "xml" / "pmcid_PMC9000001.xml").read_bytes() == HEAD + one + TAIL
    assert not (work / "xml" / "pmcid_PMC9000003.xml").exists()
    put(tmp_path / "ref.xml", HEAD + one + TAIL)
    assert parse_jats.serialize(*parse_jats.parse(str(work / "xml" / "pmcid_PMC9000001.xml"))) == \
        parse_jats.serialize(*parse_jats.parse(str(tmp_path / "ref.xml")))
    assert fetch_articles.main(args) == 1
    assert json.loads((work / "fetch_articles_report.json").read_text())["by_route"] == {"present": 2}


def cloud_routes(xml_by_pin, md5_override=None):
    def meta(url, _):
        pin = url.rsplit("/", 1)[1][:-5]
        if pin not in xml_by_pin:
            return 404, {}, b""
        md5 = md5_override or hashlib.md5(xml_by_pin[pin]).hexdigest()
        return 200, {}, json.dumps({"xml_url": f"s3://pmc-oa-opendata/{pin}/{pin}.xml?md5={md5}"}).encode()

    def xml(url, _):
        pin = url.rsplit("/", 1)[1][:-4]
        return (200, {}, xml_by_pin[pin]) if pin in xml_by_pin else (404, {}, b"")
    return [(fetch_articles.CLOUD + "metadata/", meta), (fetch_articles.CLOUD, xml)]


def test_cloud_route_checks_the_metadata_the_md5_and_the_pinned_version():
    jats = b'<?xml version="1.0"?><!DOCTYPE article>' + article("PMC9000001", 2, "one")
    c, _ = client(Transport(cloud_routes({"PMC9000001.2": jats})))
    assert fetch_articles.fetch_cloud(c, "PMC9000001.2") == (jats, "cloud")
    assert fetch_articles.fetch_cloud(c, "PMC9000001.3") == (None, "cloud_absent")
    c, _ = client(Transport(cloud_routes({"PMC9000001.2": jats}, md5_override="0" * 32)))
    assert fetch_articles.fetch_cloud(c, "PMC9000001.2") == (None, "cloud_md5_mismatch")
    c, _ = client(Transport(cloud_routes({"PMC9000001.3": jats})))
    assert fetch_articles.fetch_cloud(c, "PMC9000001.3") == (None, "cloud_version_mismatch")
    for body in (b"not json", b"[1]"):
        c, _ = client(Transport([(fetch_articles.CLOUD + "metadata/", lambda url, _, body=body: (200, {}, body))]))
        assert fetch_articles.fetch_cloud(c, "PMC9000001.2") == (None, "cloud_bad_metadata")


def test_online_fetch_uses_the_cloud_first_then_efetch_with_the_same_pin(tmp_path, monkeypatch):
    snap, work = tmp_path / "hf", tmp_path / "work"
    sources(snap, [("pmcid_PMC9000001", "PMC9000001", "PMC9000001.2", "true", "test"),
                   ("pmcid_PMC9000002", "PMC9000002", "PMC9000002.1", "true", "test"),
                   ("pmcid_PMC9000005", "PMC9000005", "PMC9000005.1", "true", "test")])
    one = b'<?xml version="1.0"?>' + article("PMC9000001", 2, "one")
    efetch = HEAD + article("PMC9000002", 1, "two") + article("PMC9000005", 2, "newer") + TAIL
    t = Transport(cloud_routes({"PMC9000001.2": one}) + [(ncbi_client.EUTILS, lambda u, d: (200, {}, efetch))])
    monkeypatch.setattr(ncbi_client, "urllib_transport", t)
    monkeypatch.setattr(ncbi_client, "LIMITER", ncbi_client.TokenBucket(Clock().now, lambda d: None))
    assert fetch_articles.main(["--snapshot", str(snap), "--profile", "test", "--work", str(work)]) == 1
    rep = json.loads((work / "fetch_articles_report.json").read_text())
    assert rep["by_route"] == {"cloud": 1, "eutils": 1}
    assert rep["failed"] == [{"task_id": "pmcid_PMC9000005", "pmcid_ver": "PMC9000005.1",
                              "reason": "eutils_version_mismatch"}]
    posts = [c for c in t.calls if c[0] == "POST"]
    assert len(posts) == 1 and posts[0][2]["db"] == ["pmc"] and posts[0][2]["id"] == ["9000002,9000005"]
    assert all(u.startswith((fetch_articles.CLOUD, ncbi_client.EUTILS)) for _, u, _ in t.calls)
    assert not any("oa.fcgi" in u or u.startswith("ftp") for _, u, _ in t.calls)


def digest_rows(snap, abstracts, extra=()):
    rows = [(p, rf.sha256_hex(a), len(a), "test") for p, a in abstracts.items()] + list(extra)
    tsv(snap / "manifests" / "abstract_sha256.tsv", rf.ABSTRACT_COLUMNS, rows)


def test_abstract_rendering_rule_and_hash_verified_variants():
    parts = [("BACKGROUND", " Tau; upsilon \"phi\". "), ("UNLABELLED", "Chi psi.")]
    assert fetch_abstracts.render_rule(parts) == "BACKGROUND Tau\\; upsilon ''phi''.   UNLABELLED Chi psi."
    assert fetch_abstracts.render_rule([(None, " Alpha beta. ")]) == "Alpha beta."
    want = "Background Tau\\; upsilon ''phi''.   Chi psi."
    text, name = fetch_abstracts.render_abstract(parts, (rf.sha256_hex(want), len(want)))
    assert (text, name) == (want, "glkb/1:cap-drop-sep3")
    text, name = fetch_abstracts.render_abstract(parts, ("0" * 64, 3))
    assert (text, name) == (fetch_abstracts.render_rule(parts), "glkb/1")
    assert fetch_abstracts.render_abstract([], ("0" * 64, 3)) == (None, "glkb/1")


def test_abstract_cache_mode_and_batched_online_fetch(tmp_path, monkeypatch):
    snap, work, cache = tmp_path / "hf", tmp_path / "work", tmp_path / "cache"
    records = {str(p): [(None, f"Synthetic abstract number {p}.")] for p in range(1000, 1160)}
    abstracts = {p: fetch_abstracts.render_rule(v) for p, v in records.items()}
    digest_rows(snap, abstracts, [("7", "0" * 64, 5, "test"), ("8", "0" * 64, 5, "train")])
    put(cache / "batch_a.xml.gz", gzip.compress(abstract_set_xml({"1000": records["1000"]})))
    put(cache / "batch_a.ids", b"1000\n7\n")
    args = ["--snapshot", str(snap), "--profile", "test", "--work", str(work), "--cache-dir", str(cache)]
    assert fetch_abstracts.main(args + ["--offline"]) == 1
    lines = [json.loads(x) for x in (work / "abstracts.jsonl").read_text().splitlines()]
    assert lines == [{"pmid": "7", "abstract": None, "render": "glkb/1"},
                     {"pmid": "1000", "abstract": abstracts["1000"], "render": "glkb/1"}]
    served = []

    def efetch(url, data):
        ids = data["id"][0].split(",")
        served.append(len(ids))
        return 200, {}, abstract_set_xml({p: records[p] for p in ids if p in records})
    t = Transport([(ncbi_client.EUTILS, efetch)])
    monkeypatch.setattr(ncbi_client, "urllib_transport", t)
    monkeypatch.setattr(ncbi_client, "LIMITER", ncbi_client.TokenBucket(Clock().now, lambda d: None))
    monkeypatch.setattr(ncbi_client, "off_peak", lambda now=None: True)
    assert fetch_abstracts.main(args) == 0
    assert served == [150, 9] and all(c[2]["db"] == ["pubmed"] and c[2]["tool"] == ["biolitbench"] for c in t.calls)
    rep = json.loads((work / "fetch_abstracts_report.json").read_text())
    assert rep == {"profile": "test", "expected": 161, "exact": 160, "today": 0, "no_abstract": 1,
                   "not_fetched": 0, "failed_requests_pmids": 0}
    assert fetch_abstracts.main(args + ["--offline"]) == 0 and len(t.calls) == 2
    assert "Synthetic" not in json.dumps(rep)


def test_abstract_fetch_refuses_a_long_series_at_peak_hours_but_keeps_the_store(tmp_path, monkeypatch):
    snap, work = tmp_path / "hf", tmp_path / "work"
    digest_rows(snap, {str(p): "x" for p in range(1, 103)})
    t = Transport([(ncbi_client.EUTILS, lambda u, d: (200, {}, b"<PubmedArticleSet></PubmedArticleSet>"))])
    monkeypatch.setattr(ncbi_client, "urllib_transport", t)
    monkeypatch.setattr(ncbi_client, "off_peak", lambda now=None: False)
    code = fetch_abstracts.main(["--snapshot", str(snap), "--profile", "test", "--work", str(work), "--batch", "1"])
    assert code == 2 and t.calls == [] and (work / "abstracts.jsonl").exists()
