#!/usr/bin/env python3
"""Embeds one worker's share of the pool corpus shards with a published pipeline's encoder and text
rules into <out>/<model>_<field>/. Usage: python embed_shard.py --model nomic|gte --field title|abs
--shards <glob> --out <emb root>."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed_common as C


def build_model(sp: dict, device: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(sp["model_path"], trust_remote_code=True, device=device)
    got = model.get_sentence_embedding_dimension()
    if got != sp["dim"]:
        raise SystemExit("model dim %s != spec dim %s" % (got, sp["dim"]))
    mods = [type(m).__name__ for m in model]
    if ("Normalize" in mods) != sp["st_normalize_builtin"]:
        raise SystemExit("sentence-transformers module stack %s disagrees with spec st_normalize_builtin=%s"
                         % (mods, sp["st_normalize_builtin"]))
    return model


def encode_documents(model, sp: dict, texts, batch_size: int):
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=sp["encode_normalize"], show_progress_bar=False)


def read_shard(path: str, sp: dict, field: str):
    pmids, texts = [], []
    for rec in C.iter_corpus(path):
        pmids.append(str(rec.get("pmid")))
        texts.append(C.doc_text(sp, field, rec))
    return pmids, texts


def output_is_complete(out_dir: str, field: str, model: str, idx: int, dim: int) -> bool:
    import numpy as np

    npy = os.path.join(out_dir, C.emb_name(field, model, idx))
    pj = os.path.join(out_dir, C.pmids_name(idx))
    if not (os.path.exists(npy) and os.path.exists(pj)):
        return False
    try:
        with open(pj, "r", encoding="utf-8") as f:
            n_ids = len(json.load(f)["pmids"])
        a = np.load(npy, mmap_mode="r")
    except (OSError, ValueError, KeyError):
        return False
    return a.ndim == 2 and a.shape[0] == n_ids and a.shape[1] == dim


def embed_one_shard(path, sp, args, model, out_dir) -> dict:
    import numpy as np

    idx = C.shard_index(path)
    npy = os.path.join(out_dir, C.emb_name(args.field, args.model, idx))
    pj = os.path.join(out_dir, C.pmids_name(idx))
    pmids, texts = read_shard(path, sp, args.field)
    n = len(pmids)
    tmp = npy + ".partial"
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype="float32", shape=(n, sp["dim"]))
    t0 = time.time()
    step = args.batch_size * 64
    for lo in range(0, n, step):
        hi = min(lo + step, n)
        arr[lo:hi] = np.asarray(encode_documents(model, sp, texts[lo:hi], args.batch_size), dtype="float32")
        if args.progress:
            print("  shard %03d %d/%d" % (idx, hi, n), flush=True)
    arr.flush()
    del arr
    os.replace(tmp, npy)
    with open(pj + ".partial", "w", encoding="utf-8") as f:
        json.dump({"shard": idx, "source": os.path.abspath(path), "model": args.model, "field": args.field,
                   "dim": sp["dim"], "doc_prefix": sp["doc_prefix"], "abs_text": sp["abs_text"],
                   "encode_normalize": sp["encode_normalize"], "count": n, "pmids": pmids}, f)
    os.replace(pj + ".partial", pj)
    print("[done] shard %03d n=%d embed=%.1fs" % (idx, n, time.time() - t0), flush=True)
    return {"shard": idx, "n": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(C.SPECS))
    ap.add_argument("--field", required=True, choices=C.FIELDS)
    ap.add_argument("--shards", required=True, help="glob for corpus_*.jsonl.gz")
    ap.add_argument("--shard-index", type=int, default=0, help="this worker's index N")
    ap.add_argument("--num-workers", type=int, default=1, help="total workers M")
    ap.add_argument("--out", required=True, help="embedding root; the shards go to <out>/<model>_<field>/")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    ap.add_argument("--force", action="store_true", help="re-embed even if outputs look complete")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    sp = C.spec(args.model)
    out_dir = os.path.join(args.out, "%s_%s" % (args.model, args.field))
    os.makedirs(out_dir, exist_ok=True)
    files = C.list_shards(args.shards)
    mine = [f for i, f in enumerate(files) if i % args.num_workers == args.shard_index]
    todo = [f for f in mine
            if args.force or not output_is_complete(out_dir, args.field, args.model, C.shard_index(f), sp["dim"])]
    print("[embed] model=%s field=%s: %d of %d shards for worker %d/%d, %d to do -> %s"
          % (args.model, args.field, len(mine), len(files), args.shard_index, args.num_workers, len(todo), out_dir),
          flush=True)
    if not todo:
        return 0
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(sp, device)
    for f in todo:
        embed_one_shard(f, sp, args, model, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
