"""Losses of GRPO training: group-normalised advantages, the clipped policy loss, truncated importance
weights and the KL estimator."""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, bool]:
    r = rewards.float()
    std = r.std(correction=0)
    if std.item() < 1e-8:
        return torch.zeros_like(r), False
    return (r - r.mean()) / (std + eps), True


def grpo_clip_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    adv: torch.Tensor,
    tok_weight: torch.Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.28,
) -> tuple[torch.Tensor, dict]:
    ratio = torch.exp(new_logp - old_logp)
    s1 = ratio * adv
    s2 = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
    per_tok = -torch.min(s1, s2)
    loss = (per_tok * tok_weight).sum()
    with torch.no_grad():
        stats = {
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item() if ratio.numel() else 0.0,
            "clip_frac": ((s2 < s1).float() * (adv.abs() > 0).float()).mean().item(),
            "approx_kl_old": (old_logp - new_logp).mean().item(),
        }
    return loss, stats


def tis_weights(
    learner_logp: torch.Tensor,
    behavior_logp: torch.Tensor,
    cap: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    log_w = (learner_logp.detach() - behavior_logp.detach()).float()
    w = log_w.exp().clamp(max=cap)
    with torch.no_grad():
        stats = {
            "tis_w_mean": w.mean().item() if w.numel() else 1.0,
            "tis_clip_frac": (log_w.exp() > cap).float().mean().item() if w.numel() else 0.0,
            "tis_abs_logw_mean": log_w.abs().mean().item() if w.numel() else 0.0,
        }
    return w, stats


def k3_kl(new_logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    d = ref_logp - new_logp
    return torch.exp(d) - d - 1.0
