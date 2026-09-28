#!/usr/bin/env python3
"""Exports per-article metadata (citation count, authors, journal) from the GLKB Neo4j query API as
resumable gzip JSONL shards. Usage: GLKB_URL=<url> GLKB_USER=<user> GLKB_PASS=<secret> python
export_pool_meta.py --out <dir>."""

import argparse, base64, gzip, json, os, sys, time, urllib.request

GLKB_ENV = ("GLKB_URL", "GLKB_USER", "GLKB_PASS")
GLKB_URL, _AUTH = None, None

def glkb_config():
    missing = [k for k in GLKB_ENV if not os.environ.get(k)]
    if missing:
        sys.exit(f"export_pool_meta: set {', '.join(missing)} (GLKB Neo4j query API URL and credentials)")
    auth = "Basic " + base64.b64encode(f"{os.environ['GLKB_USER']}:{os.environ['GLKB_PASS']}".encode()).decode()
    return os.environ["GLKB_URL"], auth

PAGE = 5000
SHARD_ROWS = 500_000

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
            last = e; time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"GLKB query failed after retries: {last}")

def main():
    global GLKB_URL, _AUTH
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    GLKB_URL, _AUTH = glkb_config()
    os.makedirs(a.out, exist_ok=True)
    sp = os.path.join(a.out, "meta_state.json")
    st = {"last_pmid": "", "raw": 0, "shard": 0, "rows": 0}
    if os.path.exists(sp): st = json.load(open(sp)); print(f"[resume] {st}", flush=True)
    f = gzip.open(os.path.join(a.out, f"meta_{st['shard']:04d}.jsonl.gz.part"), "at")
    def seal():
        nonlocal f
        f.close()
        p = os.path.join(a.out, f"meta_{st['shard']:04d}.jsonl.gz")
        os.replace(p + ".part", p); st["shard"] += 1; st["rows"] = 0
        f = gzip.open(os.path.join(a.out, f"meta_{st['shard']:04d}.jsonl.gz.part"), "at")
    t0 = time.time()
    while True:
        rows = q("MATCH (n:Article) WHERE n.pubmedid > $last "
                 "RETURN n.pubmedid AS pmid, n.n_citation AS nc, n.authors AS au, n.journal AS j "
                 "ORDER BY n.pubmedid LIMIT $lim", {"last": st["last_pmid"], "lim": PAGE})
        if not rows: break
        for r in rows:
            f.write(json.dumps({"pmid": r["pmid"], "n_citation": r.get("nc") or 0,
                                "authors": (r.get("au") or [])[:20], "journal": r.get("j")},
                               ensure_ascii=False) + "\n")
            st["rows"] += 1
            if st["rows"] >= SHARD_ROWS: seal()
        st["raw"] += len(rows); st["last_pmid"] = rows[-1]["pmid"]
        json.dump(st, open(sp + ".tmp", "w")); os.replace(sp + ".tmp", sp)
        if st["raw"] % 500_000 < PAGE:
            print(f"[{(time.time()-t0)/60:.1f}m] raw={st['raw']} last={st['last_pmid']}", flush=True)
    f.close()
    part = os.path.join(a.out, f"meta_{st['shard']:04d}.jsonl.gz.part")
    if os.path.exists(part): os.replace(part, part[:-5])
    print(f"[done] {st}", flush=True)

if __name__ == "__main__":
    main()
