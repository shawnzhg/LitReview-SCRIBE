#!/usr/bin/env python3
"""Downloads the pinned article versions from the PMC Cloud Service or E-utilities, or reads them from a cache."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ncbi_client
import release_format as rf

CLOUD = "https://pmc-oa-opendata.s3.amazonaws.com/"
ARTICLE_RE = re.compile(rb"<article[\s>].*?</article>", re.S)
VERSION_RE = re.compile(rb"<article-id[^>]*pub-id-type=\"pmcid-ver\"[^>]*>\s*([^<\s]+)\s*</article-id>")
BODY_RE = re.compile(rb"<body[\s>]")


def article_pins(snapshot, profile=None, tasks=None) -> list:
    pins = []
    for r in rf.read_article_sources(snapshot):
        if profile and profile not in rf.parse_profiles(r.get("profiles")):
            continue
        if tasks is not None and r["task_id"] not in tasks:
            continue
        if not rf.flag(r.get("rehydratable", "true")):
            continue
        pins.append((r["task_id"], r["pmcid"], r["pmcid_ver"]))
    return sorted(pins)


def split_articles(data: bytes):
    ms = list(ARTICLE_RE.finditer(data))
    if not ms:
        return b"", [], b""
    return data[:ms[0].start()], [m.group(0) for m in ms], data[ms[-1].end():]


def article_version(article: bytes):
    m = VERSION_RE.search(article)
    return m.group(1).decode("ascii", "replace") if m else None


def has_body(article: bytes) -> bool:
    return BODY_RE.search(article) is not None


def write_xml(out_dir: str, task_id: str, data: bytes) -> None:
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, task_id + ".xml")
    tmp = f"{dst}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)


def present_version(out_dir: str, task_id: str):
    p = os.path.join(out_dir, task_id + ".xml")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        _, arts, _ = split_articles(f.read())
    return article_version(arts[0]) if len(arts) == 1 else None


def read_maybe_gz(path: str) -> bytes:
    with (gzip.open(path) if path.endswith(".gz") else open(path, "rb")) as f:
        return f.read()


def from_cache(cache_dir: str, wanted: dict, out_dir: str, status: dict) -> None:
    for root, _, files in os.walk(cache_dir):
        for name in sorted(files):
            if not name.endswith((".xml", ".xml.gz")):
                continue
            prefix, arts, suffix = split_articles(read_maybe_gz(os.path.join(root, name)))
            for art in arts:
                ver = article_version(art)
                if ver in wanted and wanted[ver] not in status:
                    if not has_body(art):
                        continue
                    write_xml(out_dir, wanted[ver], prefix + art + suffix)
                    status[wanted[ver]] = "cache"


def fetch_cloud(client, pin: str):
    st, body = client.get(f"{CLOUD}metadata/{pin}.json", accept=(200, 403, 404))
    if st != 200:
        return None, "cloud_absent"
    try:
        xml_url = urlparse(json.loads(body).get("xml_url") or "")
    except (ValueError, AttributeError):
        return None, "cloud_bad_metadata"
    md5 = (parse_qs(xml_url.query).get("md5") or [None])[0]
    if not md5:
        return None, "cloud_no_md5"
    if xml_url.path.lstrip("/") != f"{pin}/{pin}.xml":
        return None, "cloud_version_mismatch"
    st, data = client.get(f"{CLOUD}{pin}/{pin}.xml", accept=(200, 403, 404))
    if st != 200:
        return None, "cloud_xml_absent"
    if hashlib.md5(data).hexdigest() != md5:
        return None, "cloud_md5_mismatch"
    _, arts, _ = split_articles(data)
    if len(arts) != 1 or article_version(arts[0]) != pin:
        return None, "cloud_version_mismatch"
    if not has_body(arts[0]):
        return None, "no_body"
    return data, "cloud"


def fetch_eutils(client, pins: list, batch: int, emit) -> None:
    for i in range(0, len(pins), batch):
        chunk = pins[i:i + batch]
        try:
            data = client.eutils("efetch.fcgi", {"db": "pmc", "id": ",".join(p.split(".")[0][3:] for p in chunk),
                                                 "retmode": "xml"},
                                 check=lambda b: b.rstrip().endswith(b"</pmc-articleset>"))
        except ncbi_client.PeakHoursError:
            raise
        except ncbi_client.NcbiError:
            for pin in chunk:
                emit(pin, None, "eutils_error")
            continue
        prefix, arts, suffix = split_articles(data)
        by_pmcid = {}
        for art in arts:
            ver = article_version(art)
            if ver:
                by_pmcid.setdefault(ver.split(".")[0], (ver, art))
        for pin in chunk:
            ver, art = by_pmcid.get(pin.split(".")[0], (None, None))
            if art is None:
                emit(pin, None, "eutils_absent")
            elif ver != pin:
                emit(pin, None, "eutils_version_mismatch")
            elif not has_body(art):
                emit(pin, None, "no_body")
            else:
                emit(pin, prefix + art + suffix, "eutils")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--profile", choices=rf.PROFILES)
    ap.add_argument("--tasks")
    ap.add_argument("--work", required=True)
    ap.add_argument("--cache-dir", action="append", default=[])
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--route", choices=("auto", "cloud", "eutils"), default="auto")
    ap.add_argument("--batch", type=int, default=100)
    ncbi_client.add_arguments(ap)
    a = ap.parse_args(argv)
    if not a.profile and not a.tasks:
        ap.error("give --profile or --tasks")
    if not 1 <= a.batch <= ncbi_client.MAX_IDS_PER_POST:
        ap.error(f"--batch must be 1..{ncbi_client.MAX_IDS_PER_POST}")
    tasks = {line.strip() for line in open(a.tasks) if line.strip()} if a.tasks else None
    pins = article_pins(a.snapshot, a.profile, tasks)
    out_dir = os.path.join(a.work, "xml")
    status, reasons = {}, {}
    for t, _, pin in pins:
        if present_version(out_dir, t) == pin:
            status[t] = "present"
    wanted = {pin: t for t, _, pin in pins if t not in status}
    for cache_dir in a.cache_dir:
        if wanted:
            from_cache(cache_dir, wanted, out_dir, status)
    todo = [(t, pin) for t, _, pin in pins if t not in status]
    code = 0
    try:
        if todo and not a.offline:
            client = ncbi_client.from_args(a)
            task_of = {pin: t for t, pin in todo}

            def emit(pin, data, why):
                if data is not None:
                    write_xml(out_dir, task_of[pin], data)
                    status[task_of[pin]] = why
                else:
                    reasons[task_of[pin]] = why

            if a.route in ("auto", "cloud"):
                for _, pin in todo:
                    try:
                        emit(pin, *fetch_cloud(client, pin))
                    except ncbi_client.NcbiError:
                        emit(pin, None, "cloud_error")
                todo = [(t, pin) for t, pin in todo if t not in status]
            if a.route in ("auto", "eutils") and todo:
                client.plan(math.ceil(len(todo) / a.batch))
                fetch_eutils(client, [pin for _, pin in todo], a.batch, emit)
    except ncbi_client.PeakHoursError as e:
        print(str(e), file=sys.stderr)
        code = 2
    finally:
        failed = [{"task_id": t, "pmcid_ver": pin,
                   "reason": reasons.get(t, "offline_not_in_cache" if a.offline else "not_fetched")}
                  for t, _, pin in pins if t not in status]
        report = {"profile": a.profile, "n_tasks": len(pins), "by_route": dict(Counter(status.values())),
                  "n_failed": len(failed), "failed": failed}
        os.makedirs(a.work, exist_ok=True)
        with open(os.path.join(a.work, "fetch_articles_report.json"), "w") as f:
            json.dump(report, f, indent=1)
        print(json.dumps({k: v for k, v in report.items() if k != "failed"}))
    return code or (1 if failed else 0)


if __name__ == "__main__":
    sys.exit(main())
