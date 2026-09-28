"""The three planning-exit readouts (lexical and embedding title F1, org_size_fit) and their
z-normalisation on a task's peer band, with the title extraction and embedding clients they use."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import threading
import urllib.request
from pathlib import Path

import numpy as np
from windowbench.planning_exit import org_size_fit, title_f1_emb, title_f1_lex, top_level

READOUTS = ("outline_title_f1_lex", "outline_title_f1_emb", "org_size_fit")
DIRECTION = {"outline_title_f1_lex": "quality", "outline_title_f1_emb": "quality",
             "org_size_fit": "style"}
DOC_PREFIX = "search_document: "
WINDOW = "planning_exit"


def nodes_from_plan(plan: dict) -> list:
    return [{"id": s.get("section_id"), "title": s.get("title") or "",
             "level": 1 if not s.get("parent_id") else 2, "parent": s.get("parent_id")}
            for s in (plan or {}).get("sections") or []]


def nodes_from_top_sections(top_sections: list) -> list:
    return [{"id": x["id"], "title": x["title"], "level": 1, "parent": None} for x in top_sections]


def _ws(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def sys_top_titles(nodes: list, cco) -> tuple:
    view_titles = [_ws(n.get("title")) for n in top_level(nodes, cco)]
    titled = [t for t in view_titles if t]
    return [cco._norm(t) for t in titled], len(titled)


def human_top_titles(top_sections: list, cco) -> tuple:
    return [cco._norm(t["title"]) for t in top_sections], len(top_sections)


def readouts(sys_top: list, n_sys: int, hum_top: list, n_hum: int, cc) -> dict:
    return {"outline_title_f1_lex": title_f1_lex(sys_top, hum_top, cc),
            "outline_title_f1_emb": title_f1_emb(sys_top, hum_top, cc),
            "org_size_fit": org_size_fit(n_sys, n_hum)}


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def z_value(s, band: dict, direction: str, readout: str = ""):
    import pandas as pd
    from ccbench.rankability import normalise as NRM
    name = readout or "readout"
    num = lambda v: float("nan") if _nan(v) else float(v)
    row = {"task": "task", "readout": name, "value": num(s), "direction": direction}
    cal = {"task": "task", "readout": name, "own": num(band.get("own")), "q_lo": num(band.get("q_lo")),
           "q_hi": num(band.get("q_hi"))}
    m = NRM.normalise(pd.DataFrame([row]), pd.DataFrame([cal])).iloc[0]
    return (None if _nan(m.z) else float(m.z)), str(m.z_note)


def z_all(raw: dict, bands: dict) -> dict:
    out = {}
    for r in READOUTS:
        b = bands.get(r)
        out[r] = None if not b else z_value(raw.get(r), b, DIRECTION[r], r)[0]
    return out


def b64_f32(a) -> str:
    return base64.b64encode(np.asarray(a, dtype=np.float32).tobytes()).decode()


def from_b64_f32(s: str, dim: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float32).reshape(-1, dim).copy()


class _MemoEmbed:

    def __init__(self):
        self._memo: dict = {}
        self._lock = threading.Lock()

    def seed(self, texts, vecs):
        with self._lock:
            self._memo[tuple(texts)] = np.asarray(vecs, dtype=np.float32)

    def encode(self, texts):
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        key = tuple(texts)
        with self._lock:
            v = self._memo.get(key)
        if v is None:
            v = self._raw(list(texts))
            with self._lock:
                self._memo[key] = v
        return v

    @staticmethod
    def cosine_matrix(a, b):
        from ccbench.readouts.embed import cosine_matrix
        return cosine_matrix(a, b)


class NomicHTTPEmbed(_MemoEmbed):

    def __init__(self, url: str, prefix: str = DOC_PREFIX, timeout: int = 600):
        super().__init__()
        if not url:
            raise ValueError("NomicHTTPEmbed needs a URL (SCRIBE_EMBED_URL)")
        self.url, self.prefix, self.timeout = url.rstrip("/"), prefix, timeout

    def health(self) -> dict:
        with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
            return json.loads(r.read().decode())

    def _raw(self, texts):
        req = urllib.request.Request(self.url + "/embed",
                                     data=json.dumps({"texts": [self.prefix + t for t in texts]}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            d = json.loads(r.read().decode())
        return np.asarray(d["vecs"], dtype=np.float32)


def make_cc(embed) -> dict:
    from ccbench.config import prereg
    from ccbench.readouts import outline as ccoutline
    cfg = prereg()
    return {"cco": ccoutline, "embed": embed,
            "SEC_COS": float(cfg["channel"]["section_match_cosine"]),
            "TITLE_FUZZ": int(cfg["readouts"]["title_match_fuzzy"])}


def load_bands(path) -> dict:
    p = Path(path)
    d = json.loads(p.read_text())
    body = {k: v for k, v in d.items() if k != "sha256_16"}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    if d.get("sha256_16") and d["sha256_16"] != h:
        raise ValueError(f"train bands {p}: sha256_16 {d['sha256_16']} != recomputed {h}")
    if d.get("schema") != "scribe_train_bands/1":
        raise ValueError(f"{p}: not a scribe_train_bands/1 file (schema {d.get('schema')!r})")
    d["_path"], d["_sha256_16"] = str(p), h
    return d
