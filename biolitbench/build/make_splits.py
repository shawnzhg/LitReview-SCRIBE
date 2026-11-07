#!/usr/bin/env python3
"""Splits the articles into paper-disjoint training and held-out manifests by topic-stratified
sampling. Usage: python make_splits.py --dataset <dir> --out <dir>."""

from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

SEED = 20260824
HELD_OUT = 200


def stable_rank(paper: str) -> str:
    return hashlib.sha256(f"{SEED}:{paper}".encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    DS, OUT = Path(a.dataset), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)

    pool = sorted(p.name for p in (DS / "graphs").iterdir() if p.is_dir())
    writing_ok = set((DS / "writing_usable.txt").read_text().split())
    retrieval_ok = set((DS / "retrieval_usable.txt").read_text().split())
    domains = json.loads((DS / "manifests/domain_index.json").read_text())

    topic_of = {}
    for topic, rec in domains.items():
        for p in rec.get("papers", []):
            topic_of.setdefault(p, topic)
    by_topic = {}
    for p in pool:
        if p in writing_ok:
            by_topic.setdefault(topic_of.get(p, "_untopiced"), []).append(p)
    for t in by_topic:
        by_topic[t].sort(key=stable_rank)

    dev, round_i = [], 0
    topics_sorted = sorted(by_topic)
    while len(dev) < HELD_OUT:
        added = 0
        for t in topics_sorted:
            if len(dev) >= HELD_OUT:
                break
            if round_i < len(by_topic[t]):
                dev.append(by_topic[t][round_i]); added += 1
        if not added:
            break
        round_i += 1
    dev = sorted(dev)
    train = sorted(set(pool) - set(dev))

    for name, papers in (("biolit_train.jsonl", train), ("biolit_dev.jsonl", dev)):
        with (OUT / name).open("w") as f:
            for p in papers:
                f.write(json.dumps({"task_id": p, "writing_usable": p in writing_ok,
                                    "retrieval_usable": p in retrieval_ok}) + "\n")
    print(json.dumps({"train": len(train), "held_out": len(dev)}))


if __name__ == "__main__":
    main()
