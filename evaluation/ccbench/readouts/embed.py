"""Local sentence embeddings with an on-disk content-hash cache."""

from __future__ import annotations

import functools
import hashlib
import os

import numpy as np

from ccbench import paths

DOC_PREFIX = "search_document: "
_CACHE_DIR = paths.OUT / "_embed_cache"


@functools.lru_cache(maxsize=1)
def model():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(str(paths.nomic_model_dir()), trust_remote_code=True, device=os.environ.get("CCBENCH_DEVICE", "cpu"))
    m.max_seq_length = 256
    return m


def _key(texts: list[str], prefix: str) -> str:
    h = hashlib.sha256()
    h.update(prefix.encode())
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


def encode(texts: list[str], prefix: str = DOC_PREFIX, batch_size: int = 32) -> np.ndarray:
    if not texts:
        return np.zeros((0, 768), dtype=np.float32)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = _CACHE_DIR / f"{_key(texts, prefix)}.npy"
    if cp.exists():
        return np.load(cp)
    emb = model().encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    emb = emb.astype(np.float32)
    np.save(cp, emb)
    return emb


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return a @ b.T
