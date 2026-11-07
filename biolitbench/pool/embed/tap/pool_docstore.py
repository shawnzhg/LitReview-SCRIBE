"""Serves the paper records of the AutoSurvey and SurveyForge databases from SQLite; a paper database
without its SQLite store is an error."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading

META_NAME = "docstore_meta.json"

FIELDS = ("id", "title", "url", "date", "abs", "cat", "authors", "citation_count", "doi")

_COLS = "faiss_id, pmid, arxiv_id, title, abs, date, url, doi, cat, authors, journal, citation_count"


def _descriptor(json_path: str):
    if os.path.basename(json_path).startswith("surveys_"):
        return None
    db_dir = os.path.dirname(os.path.abspath(json_path))
    with open(os.path.join(db_dir, META_NAME), "r", encoding="utf-8") as f:
        meta = json.load(f)
    sq = os.path.join(db_dir, meta.get("sqlite", "pool_papers.sqlite"))
    if not os.path.exists(sq):
        raise RuntimeError("pool_docstore: %s names %s, which does not exist" % (META_NAME, sq))
    meta["sqlite_path"] = sq
    return meta


def _row_to_record(row) -> dict:
    (_fid, pmid, arxiv_id, title, abs_, date, url, doi, cat, authors, _journal, cites) = row
    try:
        authors = json.loads(authors) if authors else []
    except Exception:
        authors = []
    return {
        "id": None,
        "title": title or "",
        "url": url or "",
        "date": date or "",
        "abs": abs_ or "",
        "cat": cat or "",
        "authors": authors,
        "citation_count": int(cites or 0),
        "doi": doi or "",
        "_pmid": pmid,
        "_arxiv_id": arxiv_id,
    }


class _Conn:

    def __init__(self, path):
        self.path = path
        self._local = threading.local()

    def get(self):
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect("file:%s?mode=ro" % self.path, uri=True, check_same_thread=False)
            self._local.conn = c
        return c


class SqliteTable:

    def __init__(self, conn: _Conn, id_column: str, name: str):
        self._conn = conn
        self.id_column = id_column
        self.name = name

    def _mk(self, row):
        rec = _row_to_record(row)
        rec["id"] = rec["_arxiv_id"] if self.id_column == "arxiv_id" else rec["_pmid"]
        fid = row[0]
        rec.pop("_pmid", None)
        rec.pop("_arxiv_id", None)
        ordered = {k: rec[k] for k in FIELDS}
        try:
            from tinydb.table import Document

            return Document(ordered, doc_id=fid)
        except Exception:
            return ordered

    def _fetch_ids(self, ids):
        out = []
        conn = self._conn.get()
        ids = [str(i) for i in ids]
        CH = 900
        for i in range(0, len(ids), CH):
            chunk = ids[i:i + CH]
            q = "SELECT %s FROM papers WHERE %s IN (%s)" % (
                _COLS, self.id_column, ",".join("?" * len(chunk)))
            out.extend(conn.execute(q, chunk).fetchall())
        out.sort(key=lambda r: r[0])
        return [self._mk(r) for r in out]

    def search(self, cond):
        h = getattr(cond, "_hash", None)
        if isinstance(h, tuple) and len(h) == 3 and h[0] == "one_of" and h[1] == ("id",):
            return self._fetch_ids(list(h[2]))
        return self._scan(cond)

    def _scan(self, cond):
        sys.stderr.write(
            "[pool_docstore] warning: full scan for an unsupported query %r -- correct but "
            "O(n)\n" % (getattr(cond, "_hash", cond),))
        conn = self._conn.get()
        out = []
        for row in conn.execute("SELECT %s FROM papers ORDER BY faiss_id" % _COLS):
            doc = self._mk(row)
            try:
                if cond(doc):
                    out.append(doc)
            except Exception:
                pass
        return out

    def get(self, cond=None, doc_id=None):
        if doc_id is not None:
            conn = self._conn.get()
            row = conn.execute("SELECT %s FROM papers WHERE faiss_id=?" % _COLS, (doc_id,)).fetchone()
            return self._mk(row) if row else None
        res = self.search(cond)
        return res[0] if res else None

    def all(self):
        return self._scan(lambda _d: True)

    def __len__(self):
        return self._conn.get().execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def __iter__(self):
        return iter(self.all())


class SqliteTinyDB:

    def __init__(self, path, *a, **k):
        self.path = path
        self._meta = _descriptor(path)
        self._real = None
        if self._meta is None:
            from tinydb import TinyDB as _Real

            self._real = _Real(path, *a, **k)
        else:
            self._conn = _Conn(self._meta["sqlite_path"])
            sys.stderr.write(
                "[pool_docstore] %s -> %s (id_column=%s, %s docs)\n"
                % (os.path.basename(path), os.path.basename(self._meta["sqlite_path"]),
                   self._meta.get("id_column"), self._meta.get("n_docs")))

    def table(self, name, *a, **k):
        if self._real is not None:
            return self._real.table(name, *a, **k)
        return SqliteTable(self._conn, self._meta.get("id_column", "pmid"), name)

    def close(self):
        if self._real is not None:
            self._real.close()

    def __getattr__(self, item):
        if self._real is not None:
            return getattr(self._real, item)
        raise AttributeError(item)


class LazyDocstore:

    def __init__(self, table: SqliteTable):
        self._t = table

    def search(self, _id):
        recs = self._t._fetch_ids([_id])
        if not recs:
            return "ID %s not found." % _id
        rec = dict(recs[0])
        from langchain_core.documents import Document

        content = rec.pop("abs", "")
        return Document(page_content=content, metadata=rec)

    def add(self, d):
        raise NotImplementedError("pool_docstore is read-only")


def lazy_json2doc_langchain(json_path):
    meta = _descriptor(json_path)
    if meta is None:
        from src.utils import autosurvey_db_json2doc_langchain as _real

        return _real(json_path)
    table = SqliteTable(_Conn(meta["sqlite_path"]), meta.get("id_column", "pmid"), "cs_paper_info")
    sys.stderr.write("[pool_docstore] langchain docstore -> %s (lazy)\n"
                     % os.path.basename(meta["sqlite_path"]))
    return [], LazyDocstore(table), {}


def patch_module(mod) -> int:
    n = 0
    if getattr(mod, "TinyDB", None) is not None and mod.TinyDB is not SqliteTinyDB:
        mod.TinyDB = SqliteTinyDB
        n += 1
    if getattr(mod, "autosurvey_db_json2doc_langchain", None) is not None \
            and mod.autosurvey_db_json2doc_langchain is not lazy_json2doc_langchain:
        mod.autosurvey_db_json2doc_langchain = lazy_json2doc_langchain
        n += 1
    if n:
        sys.stderr.write("[pool_docstore] patched %d name(s) in %s\n" % (n, mod.__name__))
    return n
