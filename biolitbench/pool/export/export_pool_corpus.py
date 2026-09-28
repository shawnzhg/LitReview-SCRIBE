#!/usr/bin/env python3
"""Exports the article corpus of the retrieval pool from the GLKB Neo4j query API as resumable gzip
JSONL shards. Usage: GLKB_URL=<url> GLKB_USER=<user> GLKB_PASS=<secret> python
export_pool_corpus.py --out <dir>."""

import argparse, base64, gzip, json, os, sys, time, urllib.request

GLKB_ENV = ("GLKB_URL", "GLKB_USER", "GLKB_PASS")
GLKB_URL, _AUTH = None, None


def glkb_config():
    missing = [k for k in GLKB_ENV if not os.environ.get(k)]
    if missing:
        sys.exit(f"export_pool_corpus: set {', '.join(missing)} (GLKB Neo4j query API URL and credentials)")
    auth = "Basic " + base64.b64encode(f"{os.environ['GLKB_USER']}:{os.environ['GLKB_PASS']}".encode()).decode()
    return os.environ["GLKB_URL"], auth

PAGE = 5000
SHARD_ROWS = 500_000
MAX_YEAR = 2025
MIN_ABS = 200

def q(statement, parameters, timeout=120):
    body = json.dumps({"statement": statement, "parameters": parameters}).encode()
    req = urllib.request.Request(GLKB_URL, data=body, method="POST", headers={
        "Authorization": _AUTH, "Content-Type": "application/json", "Accept": "application/json"})
    last = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            if d.get("errors"):
                raise RuntimeError(f"cypher error: {d['errors']}")
            f = d.get("data", {}).get("fields", [])
            return [dict(zip(f, row)) for row in d.get("data", {}).get("values", [])]
        except Exception as e:
            last = e
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"GLKB query failed after retries: {last}")

def main():
    global GLKB_URL, _AUTH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    GLKB_URL, _AUTH = glkb_config()
    os.makedirs(args.out, exist_ok=True)
    state_path = os.path.join(args.out, "export_state.json")
    st = {"last_pmid": "", "raw_seen": 0, "kept": 0, "shard": 0, "shard_rows": 0,
          "excl_no_abs": 0, "excl_short_abs": 0, "excl_year": 0}
    if os.path.exists(state_path):
        st = json.load(open(state_path))
        print(f"[resume] {st}", flush=True)

    shard_f = None
    def open_shard():
        nonlocal shard_f
        shard_f = gzip.open(os.path.join(args.out, f"corpus_{st['shard']:04d}.jsonl.gz.part"), "at")
    def seal_shard():
        nonlocal shard_f
        if shard_f:
            shard_f.close(); shard_f = None
            p = os.path.join(args.out, f"corpus_{st['shard']:04d}.jsonl.gz")
            os.replace(p + ".part", p)
            st["shard"] += 1; st["shard_rows"] = 0
    open_shard()

    t0 = time.time()
    while True:
        rows = q("MATCH (n:Article) WHERE n.pubmedid > $last "
                 "RETURN n.pubmedid AS pmid, n.title AS title, n.abstract AS abstract, "
                 "n.pubdate AS year, n.doi AS doi ORDER BY n.pubmedid LIMIT $lim",
                 {"last": st["last_pmid"], "lim": PAGE})
        if not rows:
            break
        for r in rows:
            st["raw_seen"] += 1
            a = r.get("abstract")
            y = r.get("year")
            if a is None:
                st["excl_no_abs"] += 1
            elif len(a) <= MIN_ABS:
                st["excl_short_abs"] += 1
            elif not isinstance(y, int) or y > MAX_YEAR:
                st["excl_year"] += 1
            else:
                shard_f.write(json.dumps({"pmid": r["pmid"], "title": r.get("title"),
                                          "abstract": a, "year": y, "doi": r.get("doi")},
                                         ensure_ascii=False) + "\n")
                st["kept"] += 1; st["shard_rows"] += 1
                if st["shard_rows"] >= SHARD_ROWS:
                    seal_shard(); open_shard()
        st["last_pmid"] = rows[-1]["pmid"]
        json.dump(st, open(state_path + ".tmp", "w")); os.replace(state_path + ".tmp", state_path)
        if st["raw_seen"] % 100_000 < PAGE:
            el = time.time() - t0
            print(f"[{el/60:.1f}m] raw={st['raw_seen']} kept={st['kept']} last={st['last_pmid']}", flush=True)
    seal_shard()
    json.dump(st, open(state_path + ".tmp", "w")); os.replace(state_path + ".tmp", state_path)
    print(f"[done] {st}", flush=True)

if __name__ == "__main__":
    main()
