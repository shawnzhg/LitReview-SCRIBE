#!/usr/bin/env python3
"""Selects the evaluation tasks from the held-out split by clustering topic embeddings and sampling
proportionally across clusters. Usage: python select_eval_tasks.py --tasks <dir of
<task>/refs.json> --out <json>."""

import argparse, json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "embed"))


def main():
    import embed_common as C
    from embed_shard import build_model
    from sklearn.cluster import KMeans
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    SEED, K, N = 42, 15, 50
    src = Path(a.tasks)
    tasks = sorted(os.listdir(src))
    topics = [json.load(open(src / t / "refs.json"))["topic"] for t in tasks]
    cutoffs = {t: json.load(open(src / t / "refs.json"))["cutoff_year"] for t in tasks}

    sp = C.spec("nomic")
    model = build_model(sp, "cpu")
    vecs = np.asarray(model.encode([sp.get("query_prefix", "") + t for t in topics],
                                   batch_size=32, show_progress_bar=False), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10).fit(vecs)
    lab = km.labels_

    rng = np.random.default_rng(SEED)
    quota = {c: int(N * (lab == c).sum() / len(tasks)) for c in range(K)}
    rem = N - sum(quota.values())
    fracs = sorted(range(K), key=lambda c: -(N * (lab == c).sum() / len(tasks) % 1))
    for c in fracs[:rem]: quota[c] += 1

    chosen = []
    for c in range(K):
        members = [i for i in range(len(tasks)) if lab[i] == c]
        rng.shuffle(members)
        chosen += members[:quota[c]]
    chosen = sorted(chosen)
    sel = [tasks[i] for i in chosen]
    out = {"schema": "campaign50/1.0", "seed": SEED, "k": K,
           "method": "nomic topic embeddings + KMeans(%d, seed %d) + proportional largest-remainder sampling"
                     % (K, SEED),
           "tasks": sel, "cutoffs": {t: cutoffs[t] for t in sel}}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"{len(sel)} tasks -> {a.out}")


if __name__ == "__main__":
    main()
