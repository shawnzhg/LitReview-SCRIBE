"""Moves the weights of a fixed model to pinned CPU memory and back, so that a learner and a rollout
engine can share one GPU."""

from __future__ import annotations

import time

import torch


class BaseWeightOffloader:
    def __init__(self, module: torch.nn.Module, min_numel: int = 1 << 20,
                 pin: bool = True):
        self.entries: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        self.pinned = False
        for p in module.parameters():
            if p.device.type == "cuda" and p.numel() >= min_numel:
                assert not p.requires_grad, \
                    "BaseWeightOffloader is for FROZEN base weights only"
                try:
                    buf = torch.empty(p.shape, dtype=p.dtype, device="cpu",
                                      pin_memory=pin)
                    self.pinned = self.pinned or pin
                except RuntimeError:
                    buf = torch.empty(p.shape, dtype=p.dtype, device="cpu")
                self.entries.append((p, buf))
        self.bytes = sum(p.numel() * p.element_size() for p, _ in self.entries)
        self.offloaded = False

    def offload(self) -> float:
        if self.offloaded or not self.entries:
            return 0.0
        t0 = time.time()
        for p, buf in self.entries:
            buf.copy_(p.data, non_blocking=True)
        torch.cuda.synchronize()
        for p, _ in self.entries:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        torch.cuda.empty_cache()
        self.offloaded = True
        return time.time() - t0

    def restore(self) -> float:
        if not self.offloaded or not self.entries:
            return 0.0
        t0 = time.time()
        for p, buf in self.entries:
            t = torch.empty(buf.shape, dtype=buf.dtype, device="cuda")
            t.copy_(buf, non_blocking=True)
            p.data = t
        torch.cuda.synchronize()
        self.offloaded = False
        return time.time() - t0
