#!/usr/bin/env python3
"""Freezes the pool: adds the gold references missing from the export as the last corpus shard, and
writes the sorted pool PMID list, the per-task cutoffs and the pool manifest; needs the glkb_client
module. Usage: python freeze_manifest_streaming.py --pool <corpus dir> --tasks <dir of
<task>/refs.json> --tmp <dir>."""

import argparse, collections, glob, gzip, hashlib, json, os, subprocess, sys

ENV = dict(os.environ, LC_ALL="C")

def run(cmd, **kw):
    subprocess.run(cmd, shell=True, check=True, env=ENV, **kw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--tmp", required=True)
    a = ap.parse_args()
    POOL, TASKS, TMP = a.pool, a.tasks, a.tmp
    os.makedirs(TMP, exist_ok=True)
    import glkb_client as G
    st = json.load(open(os.path.join(POOL, "export_state.json")))
    assert not glob.glob(os.path.join(POOL, "*.part")), "export still running"
    shards = sorted(glob.glob(os.path.join(POOL, "corpus_*.jsonl.gz")))
    print(f"[freeze] {len(shards)} shards, state kept={st['kept']}", flush=True)

    years = collections.Counter(); n = 0
    raw = os.path.join(TMP, "corpus_pmids.raw")
    with open(raw, "w") as out:
        for sh in shards:
            for line in gzip.open(sh, "rt"):
                r = json.loads(line)
                out.write(r["pmid"] + "\n"); years[r["year"]] += 1; n += 1
            print(f"[freeze] {os.path.basename(sh)} cum={n}", flush=True)
    assert n == st["kept"], f"{n} != {st['kept']}"
    run(f"sort -S 700M -T {TMP} -u {raw} -o {TMP}/corpus_pmids.sorted")

    task_dirs = sorted(os.listdir(TASKS))
    gold, cutoffs = {}, {}
    for t in task_dirs:
        d = json.load(open(os.path.join(TASKS, t, "refs.json")))
        cutoffs[t] = d["cutoff_year"]
        for r in d["refs"]:
            gold.setdefault(str(r["pmid"]), r)
    with open(f"{TMP}/gold_pmids.sorted", "w") as f:
        f.write("\n".join(sorted(gold)) + "\n")
    missing = subprocess.run(f"comm -23 {TMP}/gold_pmids.sorted {TMP}/corpus_pmids.sorted",
                             shell=True, check=True, env=ENV, capture_output=True, text=True
                             ).stdout.split()
    print(f"[freeze] gold={len(gold)} missing={len(missing)}", flush=True)

    fetched = G.by_pmids(missing)
    undated = 0
    last = max(int(os.path.basename(sh)[len("corpus_"):-len(".jsonl.gz")]) for sh in shards)
    supplement = os.path.join(POOL, "corpus_%04d.jsonl.gz" % (last + 1))
    with gzip.open(supplement + ".part", "wt") as f:
        for p in missing:
            rec = gold[p]; info = fetched.get(p) or {}
            y = info.get("year")
            y = int(y) if (y or "").strip().isdigit() else None
            undated += y is None
            f.write(json.dumps({"pmid": p, "title": rec.get("title"),
                                "abstract": rec.get("abstract") or "", "year": y,
                                "doi": rec.get("doi"), "source": "gold_supplement"},
                               ensure_ascii=False) + "\n")
            years[y] += 1
    os.replace(supplement + ".part", supplement)
    with open(f"{TMP}/missing.sorted", "w") as f:
        f.write("\n".join(missing) + ("\n" if missing else ""))
    run(f"sort -m -u {TMP}/corpus_pmids.sorted {TMP}/missing.sorted -o {POOL}/pool_pmids.sorted")

    h = hashlib.sha256()
    with open(f"{POOL}/pool_pmids.sorted", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    total = int(subprocess.run(f"wc -l < {POOL}/pool_pmids.sorted", shell=True, check=True,
                               env=ENV, capture_output=True, text=True).stdout.strip())

    json.dump(cutoffs, open(os.path.join(POOL, "cutoffs.json"), "w"), indent=1)
    manifest = {
        "schema": "pool_manifest/1.0",
        "corpus_docs": n, "gold_pmids": len(gold), "gold_missing_from_export": len(missing),
        "gold_supplement_undated": undated,
        "pool_total": total, "pool_pmids_sha256": sha,
        "year_histogram": {str(k): years[k] for k in sorted(years, key=lambda y: (y is None, y or 0))},
        "export_state": st, "n_shards": len(shards),
    }
    json.dump(manifest, open(os.path.join(POOL, "pool_manifest.json"), "w"), indent=1)
    print("[freeze] pool_pmids_sha256 " + sha, flush=True)
    print(json.dumps({k: manifest[k] for k in ("corpus_docs","gold_pmids",
        "gold_missing_from_export","pool_total")}, indent=1), flush=True)

if __name__ == "__main__":
    main()
