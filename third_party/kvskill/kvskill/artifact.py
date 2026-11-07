"""Defines KVSkillArtifact, the in-memory and on-disk format of a skill carrier, with validation and
safetensors save and load."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

FORMAT_VERSION = 1
CN_NAMES = ("cn_a", "cn_b", "cn_k", "cn_v", "cn_w", "cn_bl")


@dataclass
class KVSkillArtifact:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    cn_a: torch.Tensor | None = None
    cn_b: torch.Tensor | None = None
    cn_k: torch.Tensor | None = None
    cn_v: torch.Tensor | None = None
    cn_w: torch.Tensor | None = None
    cn_bl: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return len(self.keys)

    @property
    def p(self) -> int:
        return self.keys[0].shape[1]

    @property
    def n_kv_heads(self) -> int:
        return self.keys[0].shape[0]

    @property
    def head_dim(self) -> int:
        return self.keys[0].shape[2]

    @property
    def is_kvskill(self) -> bool:
        return self.cn_a is not None

    def num_params(self) -> int:
        n = sum(t.numel() for t in self.keys) + sum(t.numel() for t in self.values)
        if self.cn_a is not None:
            n += sum(getattr(self, k).numel() for k in CN_NAMES)
        return n

    def validate(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError("keys/values layer count mismatch")
        shape = self.keys[0].shape
        for i, (k, v) in enumerate(zip(self.keys, self.values)):
            if k.shape != shape or v.shape != shape:
                raise ValueError(f"layer {i}: inconsistent shape {k.shape}/{v.shape} vs {shape}")
        if self.cn_a is not None:
            missing = [n for n in CN_NAMES if getattr(self, n) is None]
            if missing:
                raise ValueError(f"KV-Skill artifact missing {missing}")
            n_dep = len(self.meta.get("cn_depths", []))
            if self.cn_a.dim() != 2:
                raise ValueError(f"cn_a {tuple(self.cn_a.shape)} must be (d_s, d_m)")
            d_s, d_m = self.cn_a.shape
            if tuple(self.cn_b.shape) != (d_m, d_s):
                raise ValueError(f"cn_b {tuple(self.cn_b.shape)} != ({d_m}, {d_s})")
            if self.cn_k.dim() != 2 or self.cn_k.shape[0] != d_s:
                raise ValueError(f"cn_k {tuple(self.cn_k.shape)} must be (d_s={d_s}, r)")
            if tuple(self.cn_v.shape) != tuple(self.cn_k.shape):
                raise ValueError(f"cn_v {tuple(self.cn_v.shape)} != cn_k {tuple(self.cn_k.shape)}")
            if tuple(self.cn_w.shape) != (d_m,):
                raise ValueError(f"cn_w {tuple(self.cn_w.shape)} != {(d_m,)}")
            if self.cn_bl.dim() != 1 or self.cn_bl.shape[0] != n_dep:
                raise ValueError(f"cn_bl {tuple(self.cn_bl.shape)} must be one bias per "
                                 f"injection depth ({self.meta.get('cn_depths')})")
        mode = self.meta.get("position_mode")
        if mode not in ("orig_phase", "contiguous"):
            raise ValueError(f"meta.position_mode must be set (got {mode!r})")

    def save(self, path: str | Path) -> None:
        from safetensors.torch import save_file

        self.validate()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {}
        for i, (k, v) in enumerate(zip(self.keys, self.values)):
            tensors[f"key.{i}"] = k.contiguous().cpu()
            tensors[f"value.{i}"] = v.contiguous().cpu()
        if self.cn_a is not None:
            for n in CN_NAMES:
                tensors[n] = getattr(self, n).contiguous().cpu()
        save_file(tensors, str(path / "tensors.safetensors"))
        meta = dict(self.meta)
        meta.update(format_version=FORMAT_VERSION, n_layers=self.n_layers, p=self.p,
                    n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
                    has_theta=self.cn_a is not None, saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        (path / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "KVSkillArtifact":
        from safetensors.torch import load_file

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        tensors = load_file(str(path / "tensors.safetensors"), device=device)
        L = meta["n_layers"]
        art = cls(keys=[tensors[f"key.{i}"] for i in range(L)],
                  values=[tensors[f"value.{i}"] for i in range(L)],
                  **({n: tensors[n] for n in CN_NAMES} if meta.get("has_theta") else {}),
                  meta=meta)
        art.validate()
        return art
