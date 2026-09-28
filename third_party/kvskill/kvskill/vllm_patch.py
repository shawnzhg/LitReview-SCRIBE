"""Installs the KV-Skill residual branch into vLLM and loads carrier payloads into its buffers."""

from __future__ import annotations

import json
import os
import threading

import torch

_LOCK = threading.Lock()
_LAYERS: dict[int, torch.nn.Module] = {}
_STAGED_PATH: str | None = None
_VERSION = 0
_STAGED_V_NORM = float("nan")

_BUFS = ("_cn_a", "_cn_b", "_cn_k", "_cn_v", "_cn_w", "_cn_bl")


def _branch(mod, h):
    x = h.float()
    rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    q = torch.nn.functional.normalize(rms @ mod._cn_a.T, dim=-1)
    a = q @ mod._cn_k
    rv = a @ mod._cn_v.T
    g = torch.sigmoid(rms @ mod._cn_w + mod._cn_bl)
    return (g.unsqueeze(-1) * (rv @ mod._cn_b.T)).to(h.dtype)


def _patch_layer_class(cls) -> None:
    if getattr(cls, "_kvskill_cn_patched", False):
        return
    orig = cls.forward

    def forward(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if isinstance(out, tuple):
            h, res = out[0], out[1]
            if res is None:
                return (h + _branch(self, h),) + tuple(out[1:])
            return (h + _branch(self, h + res),) + tuple(out[1:])
        return out + _branch(self, out)

    cls.forward = forward
    cls._kvskill_cn_patched = True


def _register(model, d_s: int, d_m: int, r: int, n_layers: int) -> int:
    from kvskill.dims import _find_layer_stack

    stack = _find_layer_stack(model, n_layers)
    classes = set()
    for i, layer in enumerate(stack.layers):
        dev = next(layer.parameters()).device
        for name, shape in (("_cn_a", (d_s, d_m)), ("_cn_b", (d_m, d_s)), ("_cn_k", (d_s, r)),
                            ("_cn_v", (d_s, r)), ("_cn_w", (d_m,)), ("_cn_bl", ())):
            layer.register_buffer(name, torch.zeros(shape, dtype=torch.float32, device=dev),
                                  persistent=False)
        classes.add(type(layer))
        with _LOCK:
            _LAYERS[i] = layer
    for cls in classes:
        _patch_layer_class(cls)
    print(f"[cn_vllm] armed {len(_LAYERS)} layers ({[c.__name__ for c in classes]}), "
          f"d_s={d_s} d_m={d_m} r={r}", flush=True)
    return len(_LAYERS)


def stage_payload(path: str | None = None, *, force: bool = False) -> int:
    global _STAGED_PATH, _VERSION, _STAGED_V_NORM
    path = path or os.environ.get("KVSKILL_CN_PATH", "")
    with _LOCK:
        if not path or not _LAYERS:
            return 0
        if _STAGED_PATH == path and not force:
            return _VERSION
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            assert meta.get("cn") == "1", f"{path}: not a KV-Skill payload"
            depths = json.loads(meta["cn_depths"])
            t = {k: f.get_tensor(k).float() for k in
                 ("cn_a", "cn_b", "cn_k", "cn_v", "cn_w", "cn_bl")}
        khat = torch.nn.functional.normalize(t["cn_k"], dim=0)
        for layer in _LAYERS.values():
            for b in _BUFS:
                getattr(layer, b).zero_()
        for li, idx in enumerate(depths):
            m = _LAYERS[int(idx)]
            m._cn_a.copy_(t["cn_a"].to(m._cn_a.device))
            m._cn_b.copy_(t["cn_b"].to(m._cn_b.device))
            m._cn_k.copy_(khat.to(m._cn_k.device))
            m._cn_v.copy_(t["cn_v"].to(m._cn_v.device))
            m._cn_w.copy_(t["cn_w"].to(m._cn_w.device))
            m._cn_bl.copy_(t["cn_bl"][li].to(m._cn_bl.device))
        _STAGED_PATH = path
        _VERSION += 1
        _STAGED_V_NORM = float(t["cn_v"].norm())
        print(f"[cn_vllm] staged KV-Skill θ into depths {depths} "
              f"(‖V‖={_STAGED_V_NORM:.4f}, version {_VERSION})", flush=True)
        return _VERSION


def apply_patch() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_kvskill_cn_patched", False):
        return
    orig = GPUModelRunner.load_model

    def load_model(self, *a, **kw):
        out = orig(self, *a, **kw)
        spec = os.environ.get("KVSKILL_CN_SHAPE", "")
        if spec:
            d_s, d_m, r, L = (int(x) for x in spec.split(","))
            _register(self.model, d_s, d_m, r, L)
            stage_payload()
        return out

    GPUModelRunner.load_model = load_model
    GPUModelRunner._kvskill_cn_patched = True


class SkillWorkerExtension:

    def kvskill_cn_reload(self, path: str) -> int:
        os.environ["KVSKILL_CN_PATH"] = str(path)
        return stage_payload(str(path), force=True)


if os.environ.get("KVSKILL_CN_NO_AUTOPATCH") != "1":
    apply_patch()
