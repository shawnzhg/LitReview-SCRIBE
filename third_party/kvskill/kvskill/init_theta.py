"""Builds a new KV-Skill carrier on CPU for a backbone with zero slot values, so that it leaves the
backbone's output unchanged. Usage: python -m kvskill.init_theta --out <dir> --model <dir> --d_s <n>
--r <n> [--seed <n>]."""

from __future__ import annotations

import argparse

import torch

from kvskill.artifact import KVSkillArtifact
from kvskill.dims import SkillDims

RHOS = (0.25, 0.5, 0.75, 1.0)


def build_theta(out_path: str, *, model_name: str, d_s: int, r: int, seed: int = 0) -> None:
    from transformers import AutoConfig

    dims = SkillDims.from_config(AutoConfig.from_pretrained(model_name))
    d_m = dims.d_m
    depths = dims.depths(RHOS)
    g = torch.Generator().manual_seed(seed)
    meta = dict(
        arch=dims.model_type,
        base_model=model_name,
        position_mode="contiguous",
        logical_len=0,
        stage="kvskill-init",
        full_attn_layers=list(dims.full_attn_layers),
        linear_layers=list(dims.linear_layers),
        is_kvskill=True,
        cn_depths=depths,
        cn_rhos=list(RHOS),
        cn_ds=d_s,
        cn_r=r,
        cn_dm=d_m,
        cn_nlayers=dims.n_layers,
        cn_seed=seed,
    )
    art = KVSkillArtifact(
        keys=[torch.zeros(dims.n_kv, 0, dims.head_dim, dtype=torch.bfloat16)
              for _ in dims.full_attn_layers],
        values=[torch.zeros(dims.n_kv, 0, dims.head_dim, dtype=torch.bfloat16)
                for _ in dims.full_attn_layers],
        cn_a=torch.randn(d_s, d_m, generator=g) / (d_m ** 0.5),
        cn_b=torch.randn(d_m, d_s, generator=g) / (d_s ** 0.5),
        cn_k=torch.randn(d_s, r, generator=g),
        cn_v=torch.zeros(d_s, r),
        cn_w=torch.zeros(d_m),
        cn_bl=torch.zeros(len(depths)),
        meta=meta,
    )
    art.validate()
    art.save(out_path)
    n = art.num_params()
    print(f"[theta_init] {out_path}: arch={dims.model_type} d_m={d_m} L={dims.n_layers} d_s={d_s} "
          f"r={r} depths={depths} params={n:,} ({n/1e6:.3f}M), V=0", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--d_s", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build_theta(a.out, model_name=a.model, d_s=a.d_s, r=a.r, seed=a.seed)


if __name__ == "__main__":
    main()
