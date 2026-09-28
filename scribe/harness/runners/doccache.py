#!/usr/bin/env python3
"""Per-task document store of the pool backend, keyed by snapshot id and PMID, holding each paper's
title, abstract and sentences once."""

from __future__ import annotations
import json
import threading
from pathlib import Path

from common import sha256_str

_LOCK = threading.Lock()


class DocCache:
    def __init__(self, snapshot_id: str, path: Path):
        self.snapshot_id = snapshot_id
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: dict = {}
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("snapshot_id") == snapshot_id:
                        self._seen[d["paper_id"]] = d

    def has(self, pmid) -> bool:
        return str(pmid) in self._seen

    def get(self, pmid):
        return self._seen.get(str(pmid))

    def put(self, doc: dict) -> dict:
        pid = str(doc["paper_id"])
        prev = self._seen.get(pid)
        if prev is not None and not (doc.get("sentences") and not prev.get("sentences")):
            return prev
        rec = {"snapshot_id": self.snapshot_id, "paper_id": pid,
               "title": doc.get("title"), "abstract": doc.get("abstract"),
               "doi": doc.get("doi"), "year": doc.get("year"),
               "sentences": doc.get("sentences") or []}
        rec["text_hash"] = sha256_str((rec["abstract"] or "") + "|" + (rec["title"] or ""))
        with _LOCK, self.path.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._seen[pid] = rec
        return rec

    @staticmethod
    def span_hash(text: str) -> str:
        return sha256_str(text or "")

    def stats(self):
        n_sent = sum(len(d.get("sentences") or []) for d in self._seen.values())
        return {"snapshot_id": self.snapshot_id, "n_documents": len(self._seen),
                "n_sentences": n_sent, "path": str(self.path)}
