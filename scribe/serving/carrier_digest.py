"""Computes a canonical sha256 digest of a KV-Skill payload from its metadata and tensors."""

import hashlib, json
from safetensors import safe_open


def canon_sha(path):
    h = hashlib.sha256()
    with safe_open(path, framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        h.update(json.dumps(sorted(md.items())).encode())
        for k in sorted(f.keys()):
            t = f.get_tensor(k).contiguous()
            h.update(json.dumps([k, str(t.dtype), list(t.shape)]).encode())
            h.update(t.view(-1).view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()
