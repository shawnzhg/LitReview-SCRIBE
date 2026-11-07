"""vLLM worker extension that prints a [carrier_digest] line with the digest of every KV-Skill
payload it stages."""

import hashlib
import os

import kvskill.vllm_patch as _vp
from carrier_digest import canon_sha

_orig_stage = _vp.stage_payload


def _file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _buf_sha(path):
    import json
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        depths = json.loads((f.metadata() or {})["cn_depths"])
    h = hashlib.sha256()
    for i in [int(d) for d in depths]:
        m = _vp._LAYERS[i]
        for b in _vp._BUFS:
            t = getattr(m, b).detach().float().cpu().contiguous()
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def stage_payload(path=None, *, force=False):
    v = _orig_stage(path, force=force)
    p = path or os.environ.get("KVSKILL_CN_PATH", "")
    if p and _vp._LAYERS:
        print(f"[carrier_digest] staged canon={canon_sha(p)} file_sha256={_file_sha(p)} path={p} buffers={_buf_sha(p)} version={v}", flush=True)
    return v


_vp.stage_payload = stage_payload


class DigestWorkerExtension(_vp.SkillWorkerExtension):
    pass
