#!/usr/bin/env python3
"""HTTP embedding service with a sentence-transformers model on GPU. Usage: python embed_service.py
[--model <dir>] [--port <port>] [--device cuda]."""

from __future__ import annotations
import argparse, json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("SCRIBE_EMBED_MODEL_DIR", "")


class Svc:
    def __init__(self, model_path, device, batch):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, trust_remote_code=True, device=device)
        self.model.max_seq_length = 512
        self.batch = batch
        self.lock = threading.Lock()
        self.n_calls = 0; self.n_texts = 0
        self.model_path = model_path

    def embed(self, texts):
        with self.lock:
            self.n_calls += 1; self.n_texts += len(texts)
            v = self.model.encode(texts, batch_size=self.batch, normalize_embeddings=True,
                                  convert_to_numpy=True, show_progress_bar=False)
        return v


def make_handler(svc):
    class H(BaseHTTPRequestHandler):
        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._json(200, {"ok": True, "model": svc.model_path, "n_calls": svc.n_calls,
                                 "n_texts": svc.n_texts})
            else:
                self._json(404, {"error": "no such path"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            try:
                d = json.loads(self.rfile.read(n).decode() or "{}")
                texts = d.get("texts") or []
                if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                    return self._json(400, {"error": "texts must be a list of strings"})
                if not texts:
                    return self._json(200, {"dim": 1024, "vecs": []})
                v = svc.embed(texts)
                self._json(200, {"dim": int(v.shape[1]), "vecs": v.astype("float32").tolist()})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8610)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    if not a.model:
        sys.exit("embed_service: --model (or $SCRIBE_EMBED_MODEL_DIR) is required")
    t0 = time.time()
    svc = Svc(a.model, a.device, a.batch)
    v = svc.embed(["warm-up sentence"])
    print(f"[embed_service] model={a.model} dim={v.shape[1]} device={a.device} "
          f"ready in {time.time()-t0:.1f}s on {a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(svc)).serve_forever()


if __name__ == "__main__":
    main()
