"""Core KV-Skill parameters and forward hooks: SkillTheta holds the trainable tensors and
SkillBranch adds the gated slot lookup to the residual stream."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from kvskill.artifact import KVSkillArtifact


class SkillTheta:

    def __init__(self, art: KVSkillArtifact, *, device="cuda", compute_dtype=torch.bfloat16):
        assert art.meta.get("is_kvskill"), \
            "SkillTheta requires a KV-Skill artifact (kvskill.init_theta)"
        self.device = device
        self.compute_dtype = compute_dtype
        self.meta = dict(art.meta)
        self.meta["logical_len"] = 0
        self.full_idx = list(art.meta["full_attn_layers"])
        self.depths = list(art.meta["cn_depths"])
        self.d_s = int(art.meta["cn_ds"])
        self.r = int(art.meta["cn_r"])
        leaf = lambda t: (t.detach().to(device=device, dtype=torch.float32)
                          .clone().requires_grad_(True))
        self.A = leaf(art.cn_a)
        self.B = leaf(art.cn_b)
        self.K = leaf(art.cn_k)
        self.V = leaf(art.cn_v)
        self.wg = leaf(art.cn_w)
        self.bl = leaf(art.cn_bl)
        self._n_kv, self._head_dim = art.n_kv_heads, art.head_dim

    def param_groups(self, lr: float, gate_lr: float):
        return [{"params": [self.K, self.V, self.A, self.B], "lr": lr},
                {"params": [self.wg, self.bl], "lr": gate_lr}]

    def trainable_params(self):
        return [self.A, self.B, self.K, self.V, self.wg, self.bl]

    def num_trainable(self):
        return sum(t.numel() for t in self.trainable_params())

    def freeze_(self):
        for t in self.trainable_params():
            t.requires_grad_(False)
        return self

    @torch.no_grad()
    def clip_delta_grads_(self, max_norm: float) -> float:
        mx = 0.0
        for t in self.trainable_params():
            if t.grad is not None:
                n = t.grad.norm()
                mx = max(mx, float(n))
                t.grad.mul_((max_norm / (n + 1e-12)).clamp(max=1.0))
        return mx

    def to_artifact(self, *, stage: str, **extra_meta) -> KVSkillArtifact:
        meta = dict(self.meta)
        meta.pop("format_version", None)
        meta["logical_len"] = 0
        meta.update(stage=stage, **extra_meta)
        n_kv, D = self._n_kv, self._head_dim
        art = KVSkillArtifact(
            keys=[torch.zeros(n_kv, 0, D, dtype=torch.bfloat16) for _ in self.full_idx],
            values=[torch.zeros(n_kv, 0, D, dtype=torch.bfloat16) for _ in self.full_idx],
            cn_a=self.A.detach().float().cpu(),
            cn_b=self.B.detach().float().cpu(),
            cn_k=self.K.detach().float().cpu(),
            cn_v=self.V.detach().float().cpu(),
            cn_w=self.wg.detach().float().cpu(),
            cn_bl=self.bl.detach().float().cpu(),
            meta=meta,
        )
        art.validate()
        return art


class SkillBranch:

    def __init__(self, flex, depths: list[int]):
        self.flex = flex
        self.depths = list(depths)
        self.src: SkillTheta | None = None
        self.probe = False
        self.scales: dict[int, float] = {}
        self._handles = []
        layers = flex.text_model.layers
        bad = [d for d in self.depths if not 0 <= d < len(layers)]
        if bad:
            raise ValueError(
                f"SkillBranch: injection depths {bad} out of range for a {len(layers)}-layer "
                f"model ({type(flex.model).__name__}); check that --model_name matches the "
                f"artifact's base_model")
        for li, idx in enumerate(depths):
            self._handles.append(layers[idx].register_forward_hook(self._hook(li, idx)))

    def bind(self, theta):
        self.src = theta if isinstance(theta, SkillTheta) else None

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _hook(self, li: int, layer_idx: int):
        def fn(module, args, output):
            src = self.src
            if src is None:
                return None
            is_t = torch.is_tensor(output)
            h = output if is_t else output[0]
            if not torch.is_tensor(h):
                raise TypeError(
                    f"KV-Skill hook on layer {layer_idx}: expected the residual stream as a "
                    f"tensor (or first tuple element), got {type(output).__name__}/{type(h).__name__}")
            x = h.float()
            rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
            q = F.normalize(rms @ src.A.T, dim=-1)
            a = q @ F.normalize(src.K, dim=0)
            rv = a @ src.V.T
            g = torch.sigmoid(rms @ src.wg + src.bl[li])
            delta = g.unsqueeze(-1) * (rv @ src.B.T)
            if self.probe:
                with torch.no_grad():
                    self.scales[layer_idx] = float(delta.norm() / (x.norm() + 1e-12))
            out = h + delta.to(h.dtype)
            if is_t:
                return out
            return (out,) + tuple(output[1:])
        return fn
