"""Model introspection for KV-Skill: layer and head geometry from a HF config, carrier depths, and
loading of the frozen backbone with its tokenizer."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SkillDims:
    d_m: int
    n_layers: int
    full_attn_layers: list[int] = field(default_factory=list)
    linear_layers: list[int] = field(default_factory=list)
    n_kv: int = 1
    head_dim: int = 1
    model_type: str = ""

    @classmethod
    def from_config(cls, cfg) -> "SkillDims":
        text = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
        L = int(text.num_hidden_layers)
        lt = list(getattr(text, "layer_types", None) or [])
        linear_idx = [i for i, t in enumerate(lt) if t == "linear_attention"]
        if not hasattr(text, "linear_key_head_dim"):
            linear_idx = []
        n_q = int(getattr(text, "num_attention_heads", 1))
        return cls(
            d_m=int(text.hidden_size),
            n_layers=L,
            full_attn_layers=[i for i in range(L) if i not in set(linear_idx)],
            linear_layers=linear_idx,
            n_kv=int(getattr(text, "num_key_value_heads", n_q)),
            head_dim=int(getattr(text, "head_dim", 0) or (int(text.hidden_size) // max(1, n_q))),
            model_type=str(getattr(text, "model_type", "")),
        )

    def depths(self, rhos=(0.25, 0.5, 0.75, 1.0)) -> list[int]:
        return sorted({int(rho * self.n_layers) - 1 for rho in rhos})


def _find_layer_stack(model, n_layers: int):
    import torch.nn as nn

    def ok(m):
        st = getattr(m, "layers", None)
        return isinstance(st, nn.ModuleList) and len(st) == n_layers

    inner = getattr(model, "model", None)
    if inner is not None and ok(inner):
        return inner
    for _, m in model.named_modules():
        if ok(m):
            return m
    raise RuntimeError(f"{type(model).__name__}: no submodule with a {n_layers}-layer `.layers` "
                       f"ModuleList")


class FlexAuto:

    def __init__(self, model_name: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        cfg = AutoConfig.from_pretrained(model_name)
        self.dims = SkillDims.from_config(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model, info = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, output_loading_info=True)
        missing = list(info["missing_keys"])
        if missing:
            raise RuntimeError(f"{model_name}: {len(missing)} weights were not loaded "
                               f"(e.g. {missing[:5]}); refusing a partly random backbone")
        self.model = model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.text_cfg = cfg.get_text_config()
        self.text_model = _find_layer_stack(self.model, self.dims.n_layers)
        self.lm_head = self.model.get_output_embeddings()
        assert self.lm_head is not None, f"{model_name}: no output embedding"

    def __repr__(self):
        d = self.dims
        return (f"FlexAuto({self.model_name}, type={d.model_type}, d_m={d.d_m}, "
                f"L={d.n_layers}, linear_layers={len(d.linear_layers)})")
