"""Initialises a KV-Skill carrier from a text skill by solving the slot values so that the injection
reproduces the shift the skill causes in the task's prompts. Usage: python -m kvskill.seed_init
--base_init <dir> --skill <file> --out <dir> --fit_task <task>."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

FIT_ITEMS = 64
SUFFIX_CAP = 128
MIN_SUFFIX = 8
RIDGE = 1e-6
N_PROBE = 4
PROBE_LEN = 512
INJ_MAX = 0.30
SEP = "\n\n"


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def insert_skill(msgs: list[dict], skill: str) -> list[dict]:
    out = [dict(m) for m in msgs]
    for m in out:
        if m.get("role") == "system":
            m["content"] = (m.get("content") or "").rstrip() + SEP + skill.strip()
            return out
    return [{"role": "system", "content": skill.strip()}] + out


def common_suffix_len(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def render_pair(tokenizer, msgs: list[dict], skill: str) -> tuple[list[int], list[int]]:
    from kvskill.chat import tokenize_prompt
    return tokenize_prompt(tokenizer, msgs), tokenize_prompt(tokenizer, insert_skill(msgs, skill))


@torch.no_grad()
def prefill_states(flex, depths: list[int], text: str):
    layers = flex.text_model.layers
    cap: dict[int, torch.Tensor] = {}
    handles = []
    for idx in depths:
        def mk(idx):
            def hook(module, args, output):
                h = output if torch.is_tensor(output) else output[0]
                cap[idx] = h.detach()[0]
            return hook
        handles.append(layers[idx].register_forward_hook(mk(idx)))
    try:
        ids = flex.tokenizer(text, return_tensors="pt").input_ids.to(flex.device)
        flex.text_model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {d: cap[d].float() for d in depths}, int(ids.shape[1])


def derive_slots(states: dict[int, torch.Tensor], A: torch.Tensor, r: int):
    d_s = A.shape[0]
    R = torch.zeros(d_s, d_s, dtype=torch.float64)
    for h in states.values():
        s = rms_norm(h) @ A.to(h).T
        s_hat = F.normalize(s, dim=-1)
        R += (s_hat.T @ s_hat).double().cpu()
    R /= len(states)
    R = 0.5 * (R + R.T)
    evals, evecs = torch.linalg.eigh(R)
    spec = evals.flip(0).clamp_min(0.0)
    lam, U = spec[:r], evecs.flip(1)[:, :r]
    energy = spec.pow(2).sum().clamp_min(1e-30)
    retained = float(lam.pow(2).sum() / energy)
    return U, lam, retained, spec


@torch.no_grad()
def collect_fit_normal_eqs(flex, depths, task, skill_text, art, K):
    d_m, r = art.cn_a.shape[1], K.shape[1]
    A = art.cn_a.double()
    wg, blv = art.cn_w.double(), art.cn_bl.double()
    Kn = F.normalize(K.double(), dim=0)
    C = torch.zeros(r, r, dtype=torch.float64)
    Yc = torch.zeros(d_m, r, dtype=torch.float64)
    ysq, used, rows = 0.0, 0, 0
    layers = flex.text_model.layers
    cap: dict[int, torch.Tensor] = {}
    handles = []
    for idx in depths:
        def mk(idx):
            def hook(module, args, output):
                h = output if torch.is_tensor(output) else output[0]
                cap[idx] = h.detach()[0]
            return hook
        handles.append(layers[idx].register_forward_hook(mk(idx), prepend=True))

    def run(ids):
        flex.text_model(input_ids=torch.tensor(ids, device=flex.device).unsqueeze(0), use_cache=False)

    try:
        for it in task.train_items[: FIT_ITEMS * 3]:
            if used >= FIT_ITEMS:
                break
            sp, tp = render_pair(flex.tokenizer, task.build_messages(it), skill_text)
            sfx = min(common_suffix_len(sp, tp), SUFFIX_CAP, len(sp) - 4)
            if sfx < MIN_SUFFIX:
                continue
            run(sp)
            h_s = {d: cap[d][-sfx:].double().cpu() for d in depths}
            run(tp)
            h_t = {d: cap[d][-sfx:].double().cpu() for d in depths}
            for li, d in enumerate(depths):
                x = rms_norm(h_s[d])
                q = F.normalize(x @ A.T, dim=-1)
                g = torch.sigmoid(x @ wg + blv[li]).unsqueeze(-1)
                c = g * (q @ Kn)
                y = h_t[d] - h_s[d]
                C += c.T @ c
                Yc += y.T @ c
                ysq += float(y.pow(2).sum())
                rows += int(c.shape[0])
            used += 1
    finally:
        for h in handles:
            h.remove()
    if used == 0:
        raise RuntimeError(f"no usable training item: every prompt pair shares fewer than "
                           f"{MIN_SUFFIX} final tokens")
    print(f"[seed_init] fit pairs: {used} items x {len(depths)} depths -> {rows} rows", flush=True)
    return C, Yc, ysq, used, rows


def solve_v(art, C, Yc, ysq, *, ridge: float = RIDGE):
    B = art.cn_b.double()
    G0 = B.T @ B
    RHS = B.T @ Yc
    d_s, r = G0.shape[0], C.shape[0]
    G = G0 + ridge * (G0.diagonal().mean()) * torch.eye(d_s, dtype=torch.float64)
    Cr = C + ridge * (C.diagonal().mean()) * torch.eye(r, dtype=torch.float64)
    V = torch.linalg.solve(Cr.T, torch.linalg.solve(G, RHS).T).T
    sq = max(ysq - 2.0 * float((V * RHS).sum()) + float((V.T @ G0 @ V * C.T).sum()), 0.0)
    rel = float((sq / max(ysq, 1e-30)) ** 0.5)
    return V, rel, sq


@torch.no_grad()
def measure_inj(flex, branch, theta, ids) -> dict[int, float]:
    branch.scales.clear()
    branch.probe = True
    branch.bind(theta)
    flex.text_model(input_ids=ids, use_cache=False)
    branch.bind(None)
    branch.probe = False
    return {int(k): float(v) for k, v in sorted(branch.scales.items())}


def probe_ids(flex, task):
    from kvskill.chat import tokenize_prompt
    prompts = [tokenize_prompt(flex.tokenizer, task.build_messages(it))[-PROBE_LEN:]
               for it in task.train_items[:N_PROBE]]
    pad = flex.tokenizer.pad_token_id
    if pad is None:
        pad = flex.tokenizer.eos_token_id or 0
    T = max(len(p) for p in prompts)
    out = torch.full((len(prompts), T), int(pad), dtype=torch.long)
    for i, p in enumerate(prompts):
        out[i, : len(p)] = torch.tensor(p)
    return out.to(flex.device)


def build_seeded(base_init: str, skill_path: str, out_path: str, *, fit_task: str,
                 model_name: str | None = None) -> dict:
    from kvskill.artifact import KVSkillArtifact
    from kvskill.dims import FlexAuto
    from kvskill.operator import SkillBranch, SkillTheta
    from kvskill.rl.tasks import get_task

    art = KVSkillArtifact.load(base_init)
    assert art.meta.get("is_kvskill"), f"{base_init} is not a KV-Skill artifact"
    assert float(art.cn_v.abs().max()) == 0.0, (
        f"--base_init must be a V=0 initial carrier (got ||V|| = {float(art.cn_v.norm()):.4f})")
    model = art.meta["base_model"]
    if model_name is not None:
        assert model == model_name, (model, model_name)
    depths = list(art.meta["cn_depths"])
    d_s, r = int(art.meta["cn_ds"]), int(art.meta["cn_r"])
    skill_text = open(skill_path).read()
    task = get_task(fit_task)
    flex = FlexAuto(model)
    states, n_tok = prefill_states(flex, depths, skill_text)
    U, lam, retained, spec = derive_slots(states, art.cn_a, r)
    del states
    K = U.float().contiguous()
    branch = SkillBranch(flex, depths)
    C, Yc, ysq, used, rows = collect_fit_normal_eqs(flex, depths, task, skill_text, art, K)
    V, rel, _sq = solve_v(art, C, Yc, ysq)
    V = V.float().contiguous()
    print(f"[seed_init] rel_residual ||dh-BVc||/||dh|| = {rel:.4f}  ||V||_F = {float(V.norm()):.4f}",
          flush=True)
    theta = SkillTheta(KVSkillArtifact(keys=art.keys, values=art.values, cn_a=art.cn_a,
                                       cn_b=art.cn_b, cn_k=K, cn_v=V, cn_w=art.cn_w,
                                       cn_bl=art.cn_bl, meta=dict(art.meta)),
                       device=flex.device, compute_dtype=flex.dtype)
    inj = measure_inj(flex, branch, theta, probe_ids(flex, task))
    branch.remove()
    hot = {li: v for li, v in inj.items() if v > INJ_MAX}
    if hot:
        raise ValueError(f"seed_init: injection ratio ||delta||/||h|| above {INJ_MAX} at depth(s) "
                         f"{hot} (profile {inj}); refusing to write the carrier")
    stats = dict(mode="fit", seed_skill=skill_path, seed_tokens=n_tok, slots=r, d_s=d_s,
                 retained_energy=retained, spectrum_top8=[round(float(x), 6) for x in spec[:8]],
                 fit=dict(task=fit_task, items=used, rows=rows, rel_residual=rel,
                          suffix_cap=SUFFIX_CAP, ridge=RIDGE),
                 inj_final={str(k): round(v, 6) for k, v in inj.items()}, v_fro=float(V.norm()))
    out = KVSkillArtifact(keys=art.keys, values=art.values, cn_a=art.cn_a, cn_b=art.cn_b,
                          cn_k=K, cn_v=V, cn_w=art.cn_w, cn_bl=art.cn_bl,
                          meta=dict(art.meta, stage="KV-Skill-init-seed", seed_init=stats,
                                    base_init=base_init))
    out.save(out_path)
    print(f"[seed_init] {out_path}\n  {json.dumps(stats, indent=2)}", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_init", required=True, help="V=0 initial carrier: supplies A, B, w, depths")
    ap.add_argument("--skill", required=True, help="the skill text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit_task", required=True, help="registered task whose training prompts carry the fit")
    ap.add_argument("--model", default=None, help="asserted against the artifact's base_model")
    a = ap.parse_args()
    build_seeded(a.base_init, a.skill, a.out, fit_task=a.fit_task, model_name=a.model)


if __name__ == "__main__":
    main()
