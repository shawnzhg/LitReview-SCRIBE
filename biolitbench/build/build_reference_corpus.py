#!/usr/bin/env python3
"""Resolves every cited reference of the benchmark reviews to a PMID and abstract through GLKB and
writes the enriched reference files; needs network access and the glkb_client module. Usage:
python build_reference_corpus.py --source <results dir> --out <dir>."""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

SHARDS = 64


def shard_key(pmid: str) -> int:
    s = str(pmid)
    return (int(s) if s.isdigit() else sum(ord(c) for c in s)) % SHARDS


class ShardedCache:

    def __init__(self, cache_dir: Path):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict[int, dict] = {}
        self._dirty: set[int] = set()

    def _shard(self, pmid: str) -> dict:
        k = shard_key(pmid)
        if k not in self._loaded:
            p = self.dir / f"pmid_{k:02d}.json"
            self._loaded[k] = json.loads(p.read_text()) if p.exists() else {}
        return self._loaded[k]

    def __contains__(self, pmid: str) -> bool:
        return str(pmid) in self._shard(str(pmid))

    def get(self, pmid: str):
        return self._shard(str(pmid)).get(str(pmid))

    def put(self, pmid: str, value):
        k = shard_key(pmid)
        self._shard(str(pmid))[str(pmid)] = value
        self._dirty.add(k)

    def flush(self):
        for k in sorted(self._dirty):
            (self.dir / f"pmid_{k:02d}.json").write_text(json.dumps(self._loaded[k]))
        self._dirty.clear()

    def size(self) -> int:
        n = 0
        for p in self.dir.glob("pmid_*.json"):
            n += len(json.loads(p.read_text()))
        return n

    def flat_abstracts(self) -> dict:
        flat = {}
        self._loaded.clear()
        for p in sorted(self.dir.glob("pmid_*.json")):
            shard = json.loads(p.read_text())
            for pmid, v in shard.items():
                if v and v.get("abstract"):
                    flat[pmid] = v["abstract"]
            del shard
        return flat


def resolve_dois(dois, cache_path: Path, batch=400):
    import glkb_client as G
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = sorted({d for d in dois if d and d not in cache})
    print(f"[doi] {len(todo)} to resolve ({len(cache)} cached)", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        rows = G.query(
            "UNWIND $dois AS d MATCH (n:Article {doi:d}) "
            "RETURN d AS doi, n.pubmedid AS pmid, n.title AS title, n.abstract AS abstract",
            {"dois": chunk})
        got = {r["doi"]: r for r in rows if r.get("doi")}
        for d in chunk:
            r = got.get(d)
            cache[d] = ({"pmid": str(r["pmid"]), "title": r.get("title"), "abstract": r.get("abstract")}
                        if r and r.get("pmid") else None)
        if i and i % 4000 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"  [doi] {i + len(chunk)}/{len(todo)}", flush=True)
        time.sleep(0.05)
    cache_path.write_text(json.dumps(cache))
    return cache


def fetch_abstracts(pmids, cache: ShardedCache, batch=500):
    import glkb_client as G
    todo = sorted({str(p) for p in pmids if p and str(p) not in cache})
    print(f"[pmid] {len(todo)} abstracts to fetch", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        got = G.by_pmids(chunk)
        for p in chunk:
            v = got.get(p)
            cache.put(p, {"title": v.get("title"), "abstract": v.get("abstract"), "doi": v.get("doi")}
                      if v else None)
        if i and i % 5000 == 0:
            cache.flush()
            print(f"  [pmid] {i + len(chunk)}/{len(todo)}", flush=True)
        time.sleep(0.05)
    cache.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="a biolit-bench results_* directory")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src, out = Path(a.source), Path(a.out)
    refs_out = out / "refs_enriched"
    refs_out.mkdir(parents=True, exist_ok=True)

    papers = sorted(d.name for d in (src / "benchmark_claims").iterdir() if d.is_dir())
    print(f"[load] {len(papers)} papers from {src}", flush=True)

    all_refs, dois_needed, pmids_existing = {}, set(), set()
    for p in papers:
        rp = src / "benchmark_claims" / p / "references.json"
        if not rp.exists():
            continue
        refs = json.loads(rp.read_text()).get("references", {})
        all_refs[p] = refs
        for v in refs.values():
            if v.get("pmid"):
                pmids_existing.add(str(v["pmid"]))
            elif v.get("doi"):
                dois_needed.add(v["doi"])
    print(f"[gather] pmids={len(pmids_existing)} dois-to-resolve={len(dois_needed)}", flush=True)

    doi_cache = resolve_dois(dois_needed, out / "doi_cache.json")
    resolved = set(pmids_existing)
    for d in dois_needed:
        c = doi_cache.get(d)
        if c and c.get("pmid"):
            resolved.add(str(c["pmid"]))

    cache = ShardedCache(out / "pmid_cache")
    fetch_abstracts(resolved, cache)
    for d, c in doi_cache.items():
        if c and c.get("pmid") and c.get("abstract") and str(c["pmid"]) not in cache:
            cache.put(str(c["pmid"]), {"title": c.get("title"), "abstract": c.get("abstract"), "doi": d})
    cache.flush()

    abstracts = cache.flat_abstracts()
    print(f"[cache] {len(abstracts)} abstracts resident", flush=True)

    report = []
    for p, refs in all_refs.items():
        enriched, before, after, nab = {}, 0, 0, 0
        for rid, v in refs.items():
            pmid = str(v["pmid"]) if v.get("pmid") else None
            if pmid:
                before += 1
            if not pmid and v.get("doi"):
                c = doi_cache.get(v["doi"])
                if c and c.get("pmid"):
                    pmid = str(c["pmid"])
            ab = v.get("abstract") or (abstracts.get(pmid) if pmid else None)
            if pmid:
                after += 1
            if ab:
                nab += 1
            enriched[rid] = {"pmid": pmid, "doi": v.get("doi"), "title": v.get("title"), "abstract": ab}
        (refs_out / f"{p}.json").write_text(json.dumps(enriched))
        n = len(refs) or 1
        report.append({"paper": p, "n_refs": len(refs),
                       "res_before": round(before / n, 3), "res_after": round(after / n, 3),
                       "with_abstract": round(nab / n, 3)})

    (out / "resolution_report.json").write_text(json.dumps(report, indent=1))
    import statistics as st
    print(f"[DONE] papers={len(report)} | resolution {st.mean(r['res_before'] for r in report):.3f}"
          f" -> {st.mean(r['res_after'] for r in report):.3f}"
          f" | abstract-coverage {st.mean(r['with_abstract'] for r in report):.3f}", flush=True)
    print("->", out)


if __name__ == "__main__":
    main()
