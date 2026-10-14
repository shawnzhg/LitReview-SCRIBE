#!/usr/bin/env python3
"""Fetches the abstracts of a release profile from E-utilities or a cache and renders them in the benchmark format."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncbi_client
import release_format as rf

RENDER = "glkb/1"
CASES = ("as", "cap", "title")
UNLABELLED = ("UNLABELLED", "UNLABELED")


def escape(s: str) -> str:
    return s.replace('"', "''").replace(";", "\\;")


def render_rule(parts: list) -> str:
    if len(parts) > 1 or (parts and parts[0][0]):
        return escape("   ".join(((label + " ") if label else "") + text.strip() for label, text in parts))
    return escape(" ".join(text.strip() for _, text in parts))


def render_variant(parts: list, case: str, drop: bool, sep: str) -> str:
    out = []
    for label, text in parts:
        if label and drop and label.upper() in UNLABELLED:
            label = None
        if label:
            label = label if case == "as" else label.capitalize() if case == "cap" else label.title()
        out.append(((label + " ") if label else "") + text.strip())
    return escape(sep.join(out))


def renderings(parts: list):
    yield RENDER, render_rule(parts)
    for case, drop, sep in itertools.product(CASES, (False, True), ("   ", " ")):
        yield f"{RENDER}:{case}-{'drop' if drop else 'keep'}-sep{len(sep)}", render_variant(parts, case, drop, sep)


def render_abstract(parts, digest=None):
    if not parts:
        return None, RENDER
    if digest:
        want_sha, want_len = digest
        for name, text in renderings(parts):
            if len(text) == want_len and rf.sha256_hex(text) == want_sha:
                return text, name
    text = render_rule(parts)
    return (text or None), RENDER


def parse_records(data: bytes, wanted) -> dict:
    out = {}
    root = ET.fromstring(data)
    for art in root.iter("PubmedArticle"):
        pmid = (art.findtext("MedlineCitation/PMID") or "").strip()
        if pmid and pmid in wanted and pmid not in out:
            out[pmid] = [(a.get("Label"), "".join(a.itertext())) for a in art.findall(".//Abstract/AbstractText")]
    return out


def read_maybe_gz(path: str) -> bytes:
    with (gzip.open(path) if path.endswith(".gz") else open(path, "rb")) as f:
        return f.read()


def scan_cache(dirs, wanted, records: dict, absent: set) -> None:
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for name in sorted(files):
                if not name.endswith((".xml", ".xml.gz")):
                    continue
                path = os.path.join(root, name)
                found = parse_records(read_maybe_gz(path), wanted)
                for pmid, parts in found.items():
                    records.setdefault(pmid, parts)
                ids_file = path[:-7 if name.endswith(".gz") else -4] + ".ids"
                if os.path.exists(ids_file):
                    with open(ids_file) as f:
                        asked = {line.strip() for line in f if line.strip()}
                    absent.update(p for p in asked if p in wanted and p not in found)
    absent.difference_update(records)


def save_batch(cache: str, ids: list, data: bytes) -> None:
    os.makedirs(cache, exist_ok=True)
    stem = os.path.join(cache, "batch_" + hashlib.sha256(",".join(ids).encode()).hexdigest()[:16])
    for suffix, payload in ((".xml.gz", gzip.compress(data)), (".ids", ("\n".join(ids) + "\n").encode())):
        tmp = f"{stem}{suffix}.tmp-{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, stem + suffix)


def fetch(client, todo: list, batch: int, cache: str, wanted, records: dict, absent: set, failed: list) -> None:
    client.plan(math.ceil(len(todo) / batch))
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        try:
            data = client.eutils("efetch.fcgi", {"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"},
                                 check=lambda b: b.rstrip().endswith(b"</PubmedArticleSet>"))
        except ncbi_client.PeakHoursError:
            raise
        except ncbi_client.NcbiError:
            failed.extend(chunk)
            continue
        save_batch(cache, chunk, data)
        found = parse_records(data, wanted)
        records.update(found)
        absent.update(p for p in chunk if p not in found)


def write_store(path: str, pmids: list, records: dict, absent: set, digests: dict) -> Counter:
    counts = Counter()
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        for pmid in pmids:
            if pmid in records:
                text, name = render_abstract(records[pmid], digests.get(pmid))
            elif pmid in absent:
                text, name = None, RENDER
            else:
                counts["not_fetched"] += 1
                continue
            if text is None:
                counts["no_abstract"] += 1
            elif (rf.sha256_hex(text), len(text)) == tuple(digests[pmid]):
                counts["exact"] += 1
            else:
                counts["today"] += 1
            f.write(json.dumps({"pmid": pmid, "abstract": text, "render": name}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--profile", choices=rf.PROFILES, required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--cache-dir", action="append", default=[])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--batch", type=int, default=ncbi_client.MAX_IDS_PER_POST)
    ncbi_client.add_arguments(ap)
    a = ap.parse_args(argv)
    if not 1 <= a.batch <= ncbi_client.MAX_IDS_PER_POST:
        ap.error(f"--batch must be 1..{ncbi_client.MAX_IDS_PER_POST}")
    digests = rf.read_abstract_digests(a.snapshot, a.profile)
    pmids = sorted(digests, key=int)
    wanted = set(pmids)
    cache = os.path.join(a.work, "abstract_xml")
    records, absent, failed = {}, set(), []
    os.makedirs(a.work, exist_ok=True)
    scan_cache([cache] + a.cache_dir, wanted, records, absent)
    todo = [p for p in pmids if p not in records and p not in absent]
    code = 0
    try:
        if todo and not a.offline:
            fetch(ncbi_client.from_args(a), todo, a.batch, cache, wanted, records, absent, failed)
    except ncbi_client.PeakHoursError as e:
        print(str(e), file=sys.stderr)
        code = 2
    finally:
        counts = write_store(os.path.join(a.work, "abstracts.jsonl"), pmids, records, absent, digests)
        report = {"profile": a.profile, "expected": len(pmids), "exact": counts["exact"], "today": counts["today"],
                  "no_abstract": counts["no_abstract"], "not_fetched": counts["not_fetched"],
                  "failed_requests_pmids": len(failed)}
        with open(os.path.join(a.work, "fetch_abstracts_report.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(json.dumps(report))
    return code or (1 if counts["not_fetched"] else 0)


if __name__ == "__main__":
    sys.exit(main())
