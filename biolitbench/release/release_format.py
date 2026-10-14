"""Shared definitions of the release data format: kinds, contract paths, serialisation, placeholders, digests."""

from __future__ import annotations

import hashlib
import json
import os

FORMAT = "biolit-dehydrated/1"
REPORT_SCHEMA = "biolit-rehydrate-report/1"
DATASET = "BioLitBench"
VERSION = "v1.0"
PROFILES = ("test", "train")

PATHS = {
    "rs": "biolit-bench/results_2000/benchmark_claims/{t}/review_structure_llm.json",
    "graph": "data_2000/dataset_2000/graphs/{t}/graph_enhanced.json",
    "references": "biolit-bench/results_2000/benchmark_claims/{t}/references.json",
    "refs_enriched": "data_2000/dataset_2000/refs_enriched/{t}.json",
    "gold": "runs/gold/{split}/{t}.json",
    "taskspec": "runs/taskspecs/{split}/{t}.json",
    "grounding": "data/reference_corpus/train_2000/grounding/{t}.json",
    "bundle_test": "runs/canonical/campaign50_ref/evidence_bundle/{t}.json",
    "texts_test": "runs/canonical/campaign50_ref/texts/{t}.json",
    "oracle_bundle": "runs/oracle/dev/evidence_bundle/{t}.json",
    "oracle_sentences": "runs/oracle/dev/sentences/{t}.json",
    "baseline_refs": "runs/baseline_inputs/gold/{t}/refs.json",
    "allowlist_test": "eval/test/allowlists/{t}.json",
    "allowlist_train": "runs/canonical/train/allowlists/{t}.json",
    "bundle_train": "runs/canonical/train/bundles/evidence_bundle/{t}.json",
    "texts_train": "runs/canonical/train/bundles/texts/{t}.json",
}

SERIALIZATION = {
    "rs": "i2u+nl",
    "graph": "u",
    "references": "i2u+nl",
    "refs_enriched": "a",
    "gold": "i1u",
    "taskspec": "bytes",
    "grounding": "bytes",
    "bundle_test": "bytes",
    "texts_test": "i1u",
    "oracle_bundle": "bytes",
    "oracle_sentences": "a",
    "baseline_refs": "a",
    "allowlist_test": "bytes",
    "allowlist_train": "bytes",
    "bundle_train": "bytes",
    "texts_train": "i1u",
}

RESTORE_ORDER = ("rs", "graph", "oracle_sentences", "texts_test", "texts_train", "baseline_refs")
REVIEW_KINDS = ("rs", "graph")
ABSTRACT_KINDS = ("oracle_sentences", "texts_test", "texts_train", "baseline_refs")
UNIT_BUNDLE = {"oracle_sentences": "oracle_bundle", "texts_test": "bundle_test", "texts_train": "bundle_train"}
EDGE_LISTS = ("review_level_edges", "restatement_edges", "hard_negatives", "local_edges")
SPLIT_DIR = {"test": "dev", "train": "train"}

PM = "pm"
ORACLE_JOIN = "oracle_join"
ABSTRACT_SPAN = "a"
TITLE_SPAN = "t"

_DUMPS = {
    "i2u+nl": ({"indent": 2, "ensure_ascii": False}, "\n"),
    "u": ({"ensure_ascii": False}, ""),
    "a": ({}, ""),
    "i1u": ({"indent": 1, "ensure_ascii": False}, ""),
}

ORIGINAL_COLUMNS = ("path", "kind", "task_id", "origin", "gate", "profiles", "sha256")
ABSTRACT_COLUMNS = ("pmid", "sha256", "len", "profiles")
SOURCE_COLUMNS = ("task_id", "pmcid", "pmcid_ver", "rehydratable", "profiles")


def contract_path(kind: str, task_id: str, split: str | None = None) -> str:
    return PATHS[kind].format(t=task_id, split=split or "")


def dumps(obj, profile: str) -> bytes:
    kw, tail = _DUMPS[profile]
    return (json.dumps(obj, **kw) + tail).encode("utf-8")


def dumps_wrapper(wrapper: dict) -> bytes:
    return json.dumps(wrapper, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha16(s: str) -> str:
    return sha256_hex(s)[:16]


def text_hash(s: str) -> str:
    return "sha256:" + sha256_hex(s)


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_placeholder(s: str) -> dict:
    return {"sha256": sha256_hex(s), "len": len(s)}


def sentences_placeholder(sentences: list) -> dict:
    return {"lens": [len(s) for s in sentences], "sha16": [sha16(s) for s in sentences]}


def marker_spans(sentence: str, markers: list) -> list:
    spans, pos = [], 0
    for m in markers:
        i = sentence.find(m, pos)
        if i < 0:
            i = sentence.find(m)
        if i < 0:
            raise ValueError("marker not in sentence")
        spans.append([i, i + len(m)])
        pos = i + len(m)
    return spans


def make_wrapper(kind: str, task_id: str, path: str, target_sha256: str, data, source=None) -> dict:
    return {"format": FORMAT, "kind": kind, "task_id": task_id, "path": path,
            "serialization": SERIALIZATION[kind], "target_sha256": target_sha256,
            "source": source, "data": data}


def is_wrapper(obj) -> bool:
    return isinstance(obj, dict) and str(obj.get("format", "")).startswith("biolit-dehydrated/")


def oracle_locator(locator: str) -> tuple:
    head, k = locator.rsplit("_", 1)
    return head[4:], int(k)


def parse_profiles(value: str) -> set:
    return {p.strip() for p in (value or "").split(",") if p.strip()}


def flag(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def safe_relpath(path: str) -> str:
    norm = os.path.normpath(path)
    if os.path.isabs(norm) or norm == ".." or norm.startswith(".." + os.sep):
        raise ValueError("path escapes the root")
    return norm


def read_tsv(path, columns) -> list:
    rows = []
    with open(path, encoding="utf-8") as f:
        lines = [line.rstrip("\n").rstrip("\r") for line in f if line.strip()]
    if lines and tuple(lines[0].split("\t")[:len(columns)]) == tuple(columns):
        header = lines[0].split("\t")
        lines = lines[1:]
    else:
        header = list(columns)
    for line in lines:
        rows.append(dict(zip(header, line.split("\t"))))
    return rows


def read_originals(snapshot) -> list:
    return read_tsv(os.path.join(snapshot, "manifests", "original_sha256.tsv"), ORIGINAL_COLUMNS)


def read_abstract_digests(snapshot, profile: str | None = None) -> dict:
    out = {}
    for r in read_tsv(os.path.join(snapshot, "manifests", "abstract_sha256.tsv"), ABSTRACT_COLUMNS):
        if profile is None or profile in parse_profiles(r.get("profiles")):
            out[str(r["pmid"])] = (r["sha256"], int(r["len"]))
    return out


def read_article_sources(snapshot) -> list:
    return read_tsv(os.path.join(snapshot, "manifests", "article_sources.tsv"), SOURCE_COLUMNS)
