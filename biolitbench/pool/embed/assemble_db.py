#!/usr/bin/env python3
"""Assembles the per-shard pool embeddings into the retrieval databases of AutoSurvey and
SurveyForge. Usage: python assemble_db.py --system both --emb-root <dir> --shards <glob> --out <dir>
--meta-dir <dir> [--survey-ids <file>]."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed_common as C

SYSTEMS = {"autosurvey": "nomic", "surveyforge": "gte"}


def emb_dir(emb_root: str, model: str, field: str) -> str:
    return os.path.join(emb_root, "%s_%s" % (model, field))


def shard_pmids(emb_root: str, model: str, field: str, idx: int):
    p = os.path.join(emb_dir(emb_root, model, field), C.pmids_name(idx))
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["pmids"]


def collect_shards(shards_glob: str):
    files = C.list_shards(shards_glob)
    return [(C.shard_index(f), f) for f in files]


def build_id_space(shard_list, emb_root, models_fields, strict=True):
    pmids = []
    counts = {}
    for idx, _path in shard_list:
        ref = None
        for model, field in models_fields:
            got = shard_pmids(emb_root, model, field, idx)
            if ref is None:
                ref = got
            elif got != ref:
                raise SystemExit(
                    "shard %03d: pmid list from %s_%s disagrees with the first embedding dir "
                    "(%d vs %d entries) -- refusing to build an inconsistent id space"
                    % (idx, model, field, len(got), len(ref))
                )
        counts[idx] = len(ref)
        pmids.extend(ref)
    if strict:
        seen = {}
        dups = []
        for i, p in enumerate(pmids):
            if p in seen:
                dups.append((p, seen[p], i))
                if len(dups) >= 5:
                    break
            else:
                seen[p] = i
        if dups:
            raise SystemExit(
                "corpus contains duplicate pmids, e.g. %s.\n"
                "This is fatal: arxivid_to_index_abs.json is inverted to index_to_id, so a "
                "duplicated pmid makes some faiss id unreachable and database.search() raises "
                "KeyError at query time. De-duplicate the corpus export first."
                % ", ".join("%s@rows%d,%d" % d for d in dups)
            )
    return pmids, counts


def vector_chunks(emb_root, model, field, shard_list, dim, rows_per_chunk=200_000):
    import numpy as np

    for idx, _path in shard_list:
        npy = os.path.join(emb_dir(emb_root, model, field), C.emb_name(field, model, idx))
        a = np.load(npy, mmap_mode="r")
        if a.ndim != 2 or a.shape[1] != dim:
            raise SystemExit("%s has shape %s, expected (*, %d)" % (npy, a.shape, dim))
        for s in range(0, a.shape[0], rows_per_chunk):
            yield np.asarray(a[s:s + rows_per_chunk], dtype="float32")
        del a


def write_index(out_bin, sp, emb_root, model, field, shard_list, total):
    import numpy as np

    ids = np.arange(1, total + 1, dtype="int64")
    t0 = time.time()
    C.write_idmap_flat(out_bin, sp["dim"], sp["metric"], total,
                       vector_chunks(emb_root, model, field, shard_list, sp["dim"]), ids)
    hdr = C.read_idmap_flat_header(out_bin)
    if not hdr["size_ok"] or hdr["ntotal"] != total:
        raise SystemExit("index %s failed its own size check: %s" % (out_bin, hdr))
    print("  [index] %s  ntotal=%d d=%d metric=%d  %.1f GB  %.0fs"
          % (os.path.basename(out_bin), hdr["ntotal"], hdr["d"], hdr["metric"],
             hdr["file_size"] / 1e9, time.time() - t0), flush=True)
    return hdr


def write_id_map(path, pmids):
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("{")
        for i, p in enumerate(pmids):
            if i:
                f.write(",")
            f.write("\n  %s: %d" % (json.dumps(str(p)), i + 1))
        f.write("\n}\n")
    os.replace(tmp, path)


def _open_sqlite(path):
    import sqlite3

    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("""CREATE TABLE papers (
        faiss_id       INTEGER PRIMARY KEY,
        pmid           TEXT NOT NULL,
        arxiv_id       TEXT,
        title          TEXT,
        abs            TEXT,
        date           TEXT,
        url            TEXT,
        doi            TEXT,
        cat            TEXT,
        authors        TEXT,
        journal        TEXT,
        citation_count INTEGER
    )""")
    return con


def _record(rec, m, scheme):
    pmid = str(rec.get("pmid"))
    year = rec.get("year")
    return {
        "id": C.display_id(scheme, year, pmid),
        "title": rec.get("title") or "",
        "url": "https://pubmed.ncbi.nlm.nih.gov/%s/" % pmid,
        "date": C.synth_date(year),
        "abs": rec.get("abstract") or "",
        "cat": m.get("journal", "") or "",
        "authors": m.get("authors", []),
        "citation_count": int(m.get("citation_count", 0) or 0),
        "doi": rec.get("doi") or "",
    }


def write_stores(out_dir, shard_list, json_for, sqlite_path, meta, synth_scheme="arxiv4",
                 table_name="cs_paper_info"):
    import json as _json

    import numpy as np

    years = []
    handles = {}
    for scheme, path in json_for.items():
        handles[scheme] = open(path + ".partial", "w", encoding="utf-8")
        handles[scheme].write('{"%s": {' % table_name)

    con = _open_sqlite(sqlite_path + ".partial")
    cur = con.cursor()
    batch = []
    n = 0

    for idx, sp_path in shard_list:
        for rec in C.iter_corpus(sp_path):
            n += 1
            pmid = str(rec.get("pmid"))
            year = rec.get("year")
            years.append(int(year) if year else 0)
            m = meta.get(pmid)
            for scheme, fh in handles.items():
                doc = _record(rec, m, scheme)
                if n > 1:
                    fh.write(",")
                fh.write('"%d": %s' % (n, _json.dumps(doc, ensure_ascii=False)))
            batch.append((
                n, pmid, C.display_id(synth_scheme, year, pmid),
                rec.get("title") or "", rec.get("abstract") or "", C.synth_date(year),
                "https://pubmed.ncbi.nlm.nih.gov/%s/" % pmid, rec.get("doi") or "",
                m.get("journal", "") or "", _json.dumps(m.get("authors", [])),
                m.get("journal", "") or "", int(m.get("citation_count", 0) or 0),
            ))
            if len(batch) >= 20000:
                cur.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                batch = []
        if idx % 5 == 0 or idx == shard_list[-1][0]:
            print("  [stores] shard %03d done, %d records (meta hits=%d misses=%d)"
                  % (idx, n, meta.hits, meta.misses), flush=True)

    if batch:
        cur.executemany("INSERT INTO papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
    con.commit()
    print("  [stores] indexing sqlite ...", flush=True)
    cur.execute("CREATE UNIQUE INDEX ix_pmid ON papers(pmid)")
    cur.execute("CREATE INDEX ix_arxiv ON papers(arxiv_id)")
    con.commit()
    con.close()
    os.replace(sqlite_path + ".partial", sqlite_path)

    for scheme, fh in handles.items():
        fh.write("}}")
        fh.close()
        os.replace(json_for[scheme] + ".partial", json_for[scheme])

    stats = {"meta_shards": meta.n_shards, "with_meta": meta.hits,
             "without_meta": meta.misses, "meta_rows_scanned": meta.scanned}
    return n, np.asarray(years, dtype="int32"), stats


def link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


def build_survey_db(out_dir, survey_ids_file, pmids, emb_root, shard_list, sp, doc_db_path):
    import numpy as np

    with open(survey_ids_file, "r", encoding="utf-8") as f:
        want = [l.strip() for l in f if l.strip()]
    pos = {p: i for i, p in enumerate(pmids)}
    rows = [pos[p] for p in want if p in pos]
    missing = len(want) - len(rows)
    if not rows:
        raise SystemExit("--survey-ids matched no pmid in the pool")
    rows_arr = np.asarray(sorted(rows), dtype="int64")
    sel_pmids = [pmids[r] for r in rows_arr]
    n = len(rows_arr)
    print("  [survey] %d ids requested, %d matched (%d missing)" % (len(want), n, missing), flush=True)

    for field, fname in (("title", C.SURVEY_FILES["title"]), ("abs", C.SURVEY_FILES["abs"])):
        buf = np.empty((n, sp["dim"]), dtype="float32")
        base = 0
        taken = 0
        for idx, _p in shard_list:
            npy = os.path.join(emb_dir(emb_root, "gte", field), C.emb_name(field, "gte", idx))
            a = np.load(npy, mmap_mode="r")
            lo, hi = base, base + a.shape[0]
            m = (rows_arr >= lo) & (rows_arr < hi)
            if m.any():
                local = rows_arr[m] - lo
                buf[taken:taken + local.size] = np.asarray(a[local], dtype="float32")
                taken += local.size
            base = hi
            del a
        C.write_idmap_flat(os.path.join(out_dir, fname), sp["dim"], sp["metric"], n,
                           [buf], np.arange(1, n + 1, dtype="int64"))
        print("  [survey] wrote %s (%d x %d)" % (fname, n, sp["dim"]), flush=True)

    write_id_map(os.path.join(out_dir, C.SURVEY_FILES["id_map"]), sel_pmids)

    keep = {str(int(r) + 1) for r in rows_arr}
    with open(doc_db_path, "r", encoding="utf-8") as f:
        pool = json.load(f)["cs_paper_info"]
    out = {}
    for i, r in enumerate(rows_arr):
        out[str(i + 1)] = pool[str(int(r) + 1)]
    with open(os.path.join(out_dir, C.SURVEY_FILES["doc_db"]), "w", encoding="utf-8") as f:
        json.dump({"survey_paper_info": out}, f, ensure_ascii=False)
    print("  [survey] wrote %s (%d records)" % (C.SURVEY_FILES["doc_db"], len(out)), flush=True)
    return n


def build(args) -> int:
    import numpy as np

    systems = list(SYSTEMS) if args.system == "both" else [args.system]
    models_fields = [(SYSTEMS[s], f) for s in systems for f in C.FIELDS]

    shard_list = collect_shards(args.shards)
    print("[assemble] %d corpus shards: %s .. %s"
          % (len(shard_list), os.path.basename(shard_list[0][1]), os.path.basename(shard_list[-1][1])), flush=True)

    for model, field in models_fields:
        d = emb_dir(args.emb_root, model, field)
        for idx, _p in shard_list:
            for nm in (C.emb_name(field, model, idx), C.pmids_name(idx)):
                if not os.path.exists(os.path.join(d, nm)):
                    raise SystemExit("missing embedding artefact %s" % os.path.join(d, nm))
    pmids, _counts = build_id_space(shard_list, args.emb_root, models_fields)

    os.makedirs(args.out, exist_ok=True)
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "systems": {},
                "shards": [os.path.basename(p) for _i, p in shard_list]}

    schemes = {}
    for system in systems:
        schemes.setdefault(C.spec(SYSTEMS[system])["id_scheme"], []).append(system)

    sqlite_path = os.path.join(args.out, C.SQLITE_NAME)
    json_for = {scheme: os.path.join(args.out, "pool_paper_db_%s.json" % scheme) for scheme in schemes}
    meta = C.open_meta(args.meta_dir)
    n_docs, id_year, meta_stats = write_stores(args.out, shard_list, json_for, sqlite_path, meta)
    np.save(os.path.join(args.out, "id_year.npy"), id_year)
    np.save(os.path.join(args.out, "id_pmid.npy"), np.asarray([int(p) for p in pmids], dtype="int64"))
    print("[assemble] %d records; with_meta=%d, without_meta=%d"
          % (n_docs, meta_stats["with_meta"], meta_stats["without_meta"]), flush=True)
    if len(pmids) != n_docs:
        raise SystemExit(
            "embedding rows (%d) != corpus records (%d). The embeddings were produced from a "
            "different corpus snapshot; re-embed before assembling." % (len(pmids), n_docs))

    for system in systems:
        model = SYSTEMS[system]
        sp = C.spec(model)
        scheme = sp["id_scheme"]
        db_dir = os.path.join(args.out, sp["db_dir"])
        os.makedirs(db_dir, exist_ok=True)
        print("[assemble] %s -> %s  (id scheme %s)" % (system, db_dir, scheme), flush=True)

        how = link_or_copy(json_for[scheme], os.path.join(db_dir, sp["doc_db"]))
        print("  [docdb] %s (%s)" % (sp["doc_db"], how), flush=True)
        link_or_copy(sqlite_path, os.path.join(db_dir, C.SQLITE_NAME))
        id_col = "pmid" if scheme == "pmid" else "arxiv_id"
        with open(os.path.join(db_dir, C.DOCSTORE_META_NAME), "w", encoding="utf-8") as f:
            json.dump({"sqlite": C.SQLITE_NAME, "id_column": id_col, "table": "cs_paper_info",
                       "id_scheme": scheme, "n_docs": n_docs}, f, indent=2)
        print("  [store] %s + %s (id_column=%s)" % (C.SQLITE_NAME, C.DOCSTORE_META_NAME, id_col), flush=True)

        ids = [C.display_id(scheme, int(id_year[i]), pmids[i]) for i in range(n_docs)]
        write_id_map(os.path.join(db_dir, C.ID_MAP_NAME), ids)
        print("  [idmap] %s (%d entries, ids 1..%d, keys like %r)"
              % (C.ID_MAP_NAME, n_docs, n_docs, ids[0]), flush=True)

        if scheme != "pmid":
            rev = os.path.join(db_dir, C.REVERSE_MAP_NAME)
            tmp = rev + ".partial"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("{")
                for i in range(n_docs):
                    if i:
                        f.write(",")
                    f.write("\n  %s: %s" % (json.dumps(ids[i]), json.dumps(str(pmids[i]))))
                f.write("\n}\n")
            os.replace(tmp, rev)
            print("  [revmap] %s (synthetic id -> pmid, for downstream citation matching)"
                  % C.REVERSE_MAP_NAME, flush=True)

        entry = {"db_dir": db_dir, "dim": sp["dim"], "metric": sp["metric"], "id_scheme": scheme,
                 "doc_prefix": sp["doc_prefix"], "abs_text": sp["abs_text"],
                 "encode_normalize": sp["encode_normalize"], "n_docs": n_docs,
                 "id_column": id_col, "indexes": {}}

        for field in C.FIELDS:
            out_bin = os.path.join(db_dir, sp["index_files"][field])
            hdr = write_index(out_bin, sp, args.emb_root, model, field, shard_list, n_docs)
            entry["indexes"][field] = {"file": sp["index_files"][field], **{
                k: hdr[k] for k in ("ntotal", "d", "metric", "sub_fourcc", "file_size")}}

        if system == "surveyforge" and args.survey_ids:
            entry["survey_n"] = build_survey_db(db_dir, args.survey_ids, pmids, args.emb_root,
                                                shard_list, sp, json_for[scheme])
        manifest["systems"][system] = entry

    manifest["meta"] = meta_stats
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("[assemble] manifest.json written", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", choices=["autosurvey", "surveyforge", "both"], default="both")
    ap.add_argument("--emb-root", required=True, help="root holding <model>_<field>/ from embed_shard.py")
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-dir", required=True, help="meta_*.jsonl.gz: pmid -> n_citation, authors, journal")
    ap.add_argument("--survey-ids", help="file of pmids for SurveyForge's database_survey subset")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
