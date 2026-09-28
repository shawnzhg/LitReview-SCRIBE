"""Shared code for building the AutoSurvey and SurveyForge retrieval databases on the pool:
embedding specs, document text, id schemes, the metadata join and FAISS file output."""

from __future__ import annotations

import glob as _glob
import gzip
import json
import os
import re
import struct
from typing import Dict, Iterable, Iterator, List


def _site(var: str) -> str:
    return os.environ.get(var) or "/nonexistent/" + var


METRIC_INNER_PRODUCT = 0
METRIC_L2 = 1

SPECS: Dict[str, dict] = {
    "nomic": {
        "system": "autosurvey",
        "model_path": _site("SCRIBE_EMBED_MODEL_DIR"),
        "dim": 768,
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
        "st_normalize_builtin": True,
        "encode_normalize": False,
        "metric": METRIC_L2,
        "abs_text": "abstract_only",
        "index_files": {
            "title": "faiss_paper_title_embeddings.bin",
            "abs": "faiss_paper_abs_embeddings.bin",
        },
        "doc_db": "arxiv_paper_db.json",
        "db_dir": "autosurvey_db",
        "id_scheme": "pmid",
    },
    "gte": {
        "system": "surveyforge",
        "model_path": _site("SCRIBE_GTE_MODEL_DIR"),
        "dim": 1024,
        "doc_prefix": "",
        "query_prefix": "",
        "st_normalize_builtin": False,
        "encode_normalize": True,
        "metric": METRIC_INNER_PRODUCT,
        "abs_text": "title_plus_abstract_nosep",
        "index_files": {
            "title": "faiss_paper_title_embeddings_FROM_2012_0101_TO_240926.bin",
            "abs": "faiss_paper_title_abs_embeddings_FROM_2012_0101_TO_240926.bin",
        },
        "doc_db": "arxiv_paper_db_with_cc.json",
        "db_dir": "surveyforge_db",
        "id_scheme": "arxiv4",
    },
}

ID_MAP_NAME = "arxivid_to_index_abs.json"

REVERSE_MAP_NAME = "synthetic_id_to_pmid.json"

SQLITE_NAME = "pool_papers.sqlite"
DOCSTORE_META_NAME = "docstore_meta.json"

SURVEY_FILES = {
    "doc_db": "surveys_arxiv_paper_db.json",
    "id_map": "surveys_arxivid_to_index_abs.json",
    "title": "faiss_survey_title_embeddings_FROM_1501_TO_2409_gte.bin",
    "abs": "faiss_survey_title_abs_embeddings_FROM_1501_TO_2409_gte.bin",
}

FIELDS = ("title", "abs")


def spec(model: str) -> dict:
    if model not in SPECS:
        raise SystemExit("unknown model %r (want one of %s)" % (model, ", ".join(SPECS)))
    return SPECS[model]


_SHARD_RE = re.compile(r"corpus_(\d+)\.jsonl\.gz$")


def list_shards(pattern: str) -> List[str]:
    files = [f for f in _glob.glob(pattern) if not f.endswith(".part")]
    if not files:
        raise SystemExit("no corpus shards matched %r" % pattern)

    def key(f):
        m = _SHARD_RE.search(f)
        return (0, int(m.group(1))) if m else (1, os.path.basename(f))

    return sorted(files, key=key)


def shard_index(path: str) -> int:
    m = _SHARD_RE.search(path)
    if not m:
        raise SystemExit("cannot parse a shard index out of %r" % path)
    return int(m.group(1))


def iter_corpus(path: str) -> Iterator[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _s(v) -> str:
    return "" if v is None else str(v)


def doc_text(sp: dict, field: str, rec: dict) -> str:
    title = _s(rec.get("title"))
    if field == "title":
        body = title
    elif field == "abs":
        if sp["abs_text"] == "abstract_only":
            body = _s(rec.get("abstract"))
        elif sp["abs_text"] == "title_plus_abstract_nosep":
            body = title + _s(rec.get("abstract"))
        else:
            raise AssertionError("bad abs_text %r" % sp["abs_text"])
    else:
        raise SystemExit("unknown field %r (want title|abs)" % field)
    return sp["doc_prefix"] + body


ID_SCHEMES = ("pmid", "arxiv4")

SYNTHETIC_MONTH = 7

SYNTHETIC_DAY_OF_YEAR = "06-30"


def synthetic_arxiv_id(year, pmid) -> str:
    return "%04d%02d.%s" % (int(year) if year else 0, SYNTHETIC_MONTH, pmid)


def display_id(scheme: str, year, pmid) -> str:
    if scheme == "pmid":
        return str(pmid)
    if scheme == "arxiv4":
        return synthetic_arxiv_id(year, pmid)
    raise SystemExit("unknown id scheme %r (want one of %s)" % (scheme, ", ".join(ID_SCHEMES)))


def synth_date(year) -> str:
    return "%04d-%s" % (int(year) if year else 1900, SYNTHETIC_DAY_OF_YEAR)


_META_RE = re.compile(r"meta_(\d+)\.jsonl\.gz$")


def list_meta_shards(meta_dir: str) -> List[str]:
    if not meta_dir or not os.path.isdir(meta_dir):
        return []
    files = [os.path.join(meta_dir, f) for f in os.listdir(meta_dir)
             if _META_RE.search(f) and not f.endswith(".part")]

    def key(f):
        m = _META_RE.search(f)
        return int(m.group(1))

    return sorted(files, key=key)


def _meta_record(rec: dict) -> dict:
    cites = rec.get("n_citation", rec.get("citation_count"))
    authors = rec.get("authors") or []
    if isinstance(authors, str):
        authors = [authors]
    return {
        "citation_count": int(cites) if cites is not None else 0,
        "authors": list(authors),
        "journal": rec.get("journal") or "",
    }


class MetaJoiner:

    def __init__(self, meta_dir: str):
        self.files = list_meta_shards(meta_dir)
        self.n_shards = len(self.files)
        self._last_query = ""
        self.hits = 0
        self.misses = 0
        self.scanned = 0
        self._it = self._iter_all()
        self._cur = next(self._it, None)

    def _iter_all(self):
        prev = ""
        for f in self.files:
            for rec in iter_corpus(f):
                pmid = str(rec.get("pmid"))
                if pmid < prev:
                    raise SystemExit("meta supplement is not sorted by string pmid (%r after %r in %s)"
                                     % (pmid, prev, os.path.basename(f)))
                prev = pmid
                self.scanned += 1
                yield pmid, rec

    def get(self, pmid: str) -> dict:
        pmid = str(pmid)
        if pmid < self._last_query:
            raise SystemExit("corpus pmids are not ascending (%r after %r)" % (pmid, self._last_query))
        self._last_query = pmid
        while self._cur is not None and self._cur[0] < pmid:
            self._cur = next(self._it, None)
        if self._cur is not None and self._cur[0] == pmid:
            rec = self._cur[1]
            self._cur = next(self._it, None)
            self.hits += 1
            return _meta_record(rec)
        self.misses += 1
        return {}


def open_meta(meta_dir: str):
    if not list_meta_shards(meta_dir):
        raise SystemExit("no meta_*.jsonl.gz under %r" % meta_dir)
    return MetaJoiner(meta_dir)


_DUMMY = 1 << 20


def _fourcc(s: str) -> bytes:
    return s.encode("ascii")


def _index_header(d: int, ntotal: int, metric: int) -> bytes:
    return (
        struct.pack("<i", d)
        + struct.pack("<q", ntotal)
        + struct.pack("<q", _DUMMY)
        + struct.pack("<q", _DUMMY)
        + struct.pack("<B", 1)
        + struct.pack("<i", metric)
    )


def flat_fourcc(metric: int) -> str:
    return "IxFI" if metric == METRIC_INNER_PRODUCT else "IxF2"


def idmap_flat_size(ntotal: int, d: int) -> int:
    return 37 + 37 + 8 + ntotal * d * 4 + 8 + ntotal * 8


def write_idmap_flat(
    path: str,
    dim: int,
    metric: int,
    ntotal: int,
    vector_chunks: Iterable["object"],
    ids,
    bufsize: int = 1 << 24,
) -> int:
    import numpy as np

    ids = np.ascontiguousarray(np.asarray(ids, dtype="<i8"))
    if ids.shape != (ntotal,):
        raise SystemExit("ids shape %s != (%d,)" % (ids.shape, ntotal))

    tmp = path + ".partial"
    written = 0
    with open(tmp, "wb", buffering=bufsize) as f:
        f.write(_fourcc("IxMp"))
        f.write(_index_header(dim, ntotal, metric))
        f.write(_fourcc(flat_fourcc(metric)))
        f.write(_index_header(dim, ntotal, metric))
        f.write(struct.pack("<Q", ntotal * dim))
        for chunk in vector_chunks:
            a = np.ascontiguousarray(chunk, dtype="<f4")
            if a.ndim != 2 or a.shape[1] != dim:
                raise SystemExit("chunk shape %s incompatible with dim %d" % (a.shape, dim))
            f.write(a.tobytes(order="C"))
            written += a.shape[0]
        if written != ntotal:
            raise SystemExit("wrote %d rows, declared ntotal=%d" % (written, ntotal))
        f.write(struct.pack("<Q", ntotal))
        f.write(ids.tobytes(order="C"))
    os.replace(tmp, path)
    return written


def read_idmap_flat_header(path: str) -> dict:
    with open(path, "rb") as f:
        h = f.read(82)
    out = {
        "outer_fourcc": h[0:4].decode("ascii", "replace"),
        "d": struct.unpack("<i", h[4:8])[0],
        "ntotal": struct.unpack("<q", h[8:16])[0],
        "is_trained": h[32],
        "metric": struct.unpack("<i", h[33:37])[0],
        "sub_fourcc": h[37:41].decode("ascii", "replace"),
        "sub_d": struct.unpack("<i", h[41:45])[0],
        "sub_ntotal": struct.unpack("<q", h[45:53])[0],
        "sub_metric": struct.unpack("<i", h[70:74])[0],
        "n_floats": struct.unpack("<Q", h[74:82])[0],
    }
    out["file_size"] = os.path.getsize(path)
    out["expected_size"] = idmap_flat_size(out["ntotal"], out["d"])
    out["size_ok"] = out["file_size"] == out["expected_size"]
    return out


def emb_name(field: str, model: str, idx: int) -> str:
    return "emb_%s_%s_%03d.npy" % (field, model, idx)


def pmids_name(idx: int) -> str:
    return "pmids_%03d.json" % idx

