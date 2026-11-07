#!/usr/bin/env python3
"""Writes the pmid -> (title, year, abstract) SQLite table that build_canonical_bundles.py reads, for
every paper on the tasks' reference lists, in one pass over the pool corpus shards. Usage: python
build_pool_meta_sqlite.py --shards '<pool corpus>/corpus_*.jsonl.gz' [--allowlists <dir>] [--out <sqlite>]."""

from __future__ import annotations
import argparse
import glob
import gzip
import json
import os
import sqlite3
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, help="glob of the pool corpus shards")
    ap.add_argument("--allowlists", default=os.environ.get("SCRIBE_ALLOWLISTS"), help="directory of <task>.json reference lists (default $SCRIBE_ALLOWLISTS)")
    ap.add_argument("--out", default=os.environ.get("SCRIBE_POOL_META_SQLITE"), help="output SQLite (default $SCRIBE_POOL_META_SQLITE)")
    a = ap.parse_args(argv)
    need = {int(p) for f in sorted(Path(a.allowlists).glob("*.json")) for p in json.loads(f.read_text())}
    tmp = a.out + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    con = sqlite3.connect(tmp)
    con.execute("CREATE TABLE meta (pmid INTEGER PRIMARY KEY, title TEXT, year INTEGER, abstract TEXT)")
    hit = 0
    for shard in sorted(glob.glob(a.shards)):
        batch = []
        with gzip.open(shard, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                r = json.loads(line)
                p = int(r["pmid"])
                if p in need:
                    y = r.get("year")
                    batch.append((p, r.get("title") or "", int(y) if str(y).isdigit() else None, r.get("abstract") or ""))
        con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?,?,?)", batch)
        con.commit()
        hit += len(batch)
    con.close()
    os.replace(tmp, a.out)
    print(f"wrote {a.out}: {hit} rows for {len(need)} reference-list papers")


if __name__ == "__main__":
    main()
