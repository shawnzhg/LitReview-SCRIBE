"""Learner-forward helpers of the KV-Skill branch: batch padding with response indices and chunked
per-token log-probabilities."""

from __future__ import annotations

import torch


def _pad_batch(prompts, responses, pad_id, device):
    m = len(prompts)
    seqs = [list(p) + list(r) for p, r in zip(prompts, responses)]
    T = max(len(s) for s in seqs)
    input_ids = torch.full((m, T), pad_id, dtype=torch.long)
    rows, cols, targets = [], [], []
    for i, (p, r) in enumerate(zip(prompts, responses)):
        input_ids[i, : len(seqs[i])] = torch.tensor(seqs[i], dtype=torch.long)
        base = len(p)
        for t in range(len(r)):
            j = base + t
            rows.append(i); cols.append(j - 1); targets.append(r[t])
    return (input_ids.to(device),
            torch.tensor(rows, device=device),
            torch.tensor(cols, device=device),
            torch.tensor(targets, device=device))


def _logprobs_from_hidden(hidden, rows, cols, targets, lm_head, chunk=1024, want_entropy=False):
    pred = hidden[rows, cols, :]

    def _chunk_fn(h, tgt):
        logits = lm_head(h).float()
        logZ = torch.logsumexp(logits, dim=-1)
        picked = logits.gather(1, tgt.unsqueeze(1)).squeeze(1)
        lp = picked - logZ
        if want_entropy:
            logp_all = logits - logZ.unsqueeze(1)
            ent = -(logp_all.exp() * logp_all).sum(dim=-1)
        else:
            ent = lp.new_zeros(0)
        return lp, ent

    lp_out, ent_out = [], []
    for i in range(0, pred.shape[0], chunk):
        h = pred[i : i + chunk]
        tgt = targets[i : i + chunk]
        if h.requires_grad:
            lp, ent = torch.utils.checkpoint.checkpoint(
                _chunk_fn, h, tgt, use_reentrant=False)
        else:
            lp, ent = _chunk_fn(h, tgt)
        lp_out.append(lp)
        ent_out.append(ent)
    lp = torch.cat(lp_out)
    return (lp, torch.cat(ent_out)) if want_entropy else lp


def batched_logprobs(flex, branch, theta, prompts, responses, pad_id, *,
                     grad: bool, want_entropy: bool = False):
    branch.bind(theta)
    input_ids, rows, cols, targets = _pad_batch(prompts, responses, pad_id,
                                                flex.device)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    if grad:
        flex.text_model.train()
    try:
        with ctx:
            hidden = flex.text_model(input_ids=input_ids,
                                     use_cache=False).last_hidden_state
            return _logprobs_from_hidden(hidden, rows, cols, targets,
                                         flex.lm_head, want_entropy=want_entropy)
    finally:
        if grad:
            flex.text_model.eval()
