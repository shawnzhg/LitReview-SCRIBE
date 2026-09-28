"""GRPO trainer of a KV-Skill carrier on a registered task, with one prompt order per epoch and
rollouts on a co-located vLLM server. Usage: python -m kvskill.grpo --task <task> --model_name <dir>
--artifact_path <dir> --out_dir <dir> --vllm_url <url> --base_cache <dir>."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from kvskill.artifact import KVSkillArtifact
from kvskill.chat import stop_token_ids, tokenize_prompt
from kvskill.dims import FlexAuto
from kvskill.operator import SkillBranch, SkillTheta
from kvskill.rl.losses import grpo_clip_loss, group_advantages, k3_kl, tis_weights
from kvskill.rl.tasks import get_task
from kvskill.runtime import batched_logprobs

FAILURE_REWARD = 0.0


def as_reward(x) -> tuple[float, dict, str]:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x), {}, "ok"
    if isinstance(x, dict):
        return (float(x.get("reward") or 0.0), dict(x.get("components") or {}),
                str(x.get("status") or "ok"))
    return (float(getattr(x, "reward", 0.0)), dict(getattr(x, "components", None) or {}),
            str(getattr(x, "status", "ok")))


@dataclass
class SkillConfig:
    task: str = ""
    model_name: str = ""
    artifact_path: str = ""
    out_dir: str = ""
    vllm_url: str = ""
    base_cache: str = ""
    steps: int = 48
    prompts_per_step: int = 4
    group_size: int = 8
    max_new_tokens: int = 8000
    max_prompt_tokens: int = 20000
    lr: float = 1e-4
    gate_lr: float = 1e-3
    eps_low: float = 0.2
    eps_high: float = 0.28
    beta_kl: float = 0.01
    beta_ent: float = 0.05
    grad_clip: float = 1.0
    tis_cap: float = 2.0
    learner_micro_seqs: int = 1
    seed: int = 17


def epoch_indices(seed: int, n: int, k: int, step: int) -> list[int]:
    out = []
    for pos in range((step - 1) * k, step * k):
        epoch, j = divmod(pos, n)
        perm = list(range(n))
        random.Random(f"{seed}:{epoch}").shuffle(perm)
        out.append(perm[j])
    return out


CARRIER_SYNC_RTOL = 1e-4
CARRIER_SYNC_ATOL = 1e-6


class CarrierSyncError(RuntimeError):
    pass


def learner_cn_v_norm(theta) -> float:
    return float(theta.V.detach().float().norm())


def payload_cn_v_norm(path) -> float:
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return float(f.get_tensor("cn_v").float().norm())


def check_carrier_sync(theta, path, *, label: str = "carrier sync",
                       rtol: float = CARRIER_SYNC_RTOL, atol: float = CARRIER_SYNC_ATOL):
    p, l = payload_cn_v_norm(path), learner_cn_v_norm(theta)
    print(f"### {label}: payload |cn_v|={p:.6f} learner |V|={l:.6f}", flush=True)
    if not abs(p - l) <= max(rtol * max(abs(p), abs(l)), atol):
        raise CarrierSyncError(
            f"carrier sync ({label}): the payload {path} carries |cn_v|={p:.6f} but the "
            f"learner's carrier is |V|={l:.6f}; the rollout server would sample a different "
            f"carrier than the one trained")
    return p, l


class SkillTrainer:
    def __init__(self, cfg: SkillConfig):
        from kvskill.rl.offload import BaseWeightOffloader

        self.cfg = cfg
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        torch.manual_seed(cfg.seed)
        self.task = get_task(cfg.task)
        self.task.configure_reward_batch(base_cache=cfg.base_cache)
        self.last_reward_mask = None
        self._reward_comp_f = None
        self.flex = FlexAuto(cfg.model_name)
        self.tokenizer = self.flex.tokenizer
        art0 = KVSkillArtifact.load(cfg.artifact_path)
        assert art0.meta.get("is_kvskill"), "kvskill.grpo needs a KV-Skill artifact (meta.is_kvskill)"
        assert art0.meta.get("base_model") == cfg.model_name, (
            f"artifact base_model {art0.meta.get('base_model')!r} != --model_name {cfg.model_name!r}")
        self.theta = SkillTheta(art0, device=self.flex.device, compute_dtype=self.flex.dtype)
        self.ref = SkillTheta(art0, device=self.flex.device, compute_dtype=self.flex.dtype).freeze_()
        self.branch = SkillBranch(self.flex, self.theta.depths)
        print(f"[KV-Skill] {self.flex} depths={self.theta.depths} d_s={self.theta.d_s} "
              f"r={self.theta.r} trainable={self.theta.num_trainable()/1e6:.3f}M; KL reference = "
              f"the starting carrier {cfg.artifact_path}", flush=True)
        self.opt = torch.optim.AdamW(self.theta.param_groups(cfg.lr, cfg.gate_lr), weight_decay=0.0)
        self.flex.text_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        self.eos_ids = stop_token_ids(self.tokenizer, getattr(self.flex.model, "generation_config", None))
        self.pad_id = self.tokenizer.pad_token_id or next(iter(self.eos_ids))
        self.v0 = self.theta.V.detach().cpu().clone()
        self.offloader = BaseWeightOffloader(self.flex.model)
        print(f"[sleep-colocate] learner offloader armed: {self.offloader.bytes/2**30:.1f} GiB in "
              f"{len(self.offloader.entries)} params", flush=True)
        self.t_wake_last = self.t_sleep_last = 0.0
        self._audited = False
        self.metrics_f = open(self.out / "metrics.jsonl", "a")

    def _post(self, ep, payload, timeout):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(
            self.cfg.vllm_url.rstrip("/") + ep, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"cn-server {ep} -> HTTP {e.code}; "
                               f"{e.read().decode(errors='replace')[:4000]}") from e

    def _wake(self):
        t0 = time.time()
        self.offloader.offload()
        self._post("/wake", {}, timeout=1800)
        self.t_wake_last = time.time() - t0

    def _sleep(self):
        t0 = time.time()
        self._post("/sleep", {"level": 1}, timeout=1800)
        self.offloader.restore()
        self.t_sleep_last = time.time() - t0

    def _reload(self):
        from kvskill.export import export_from_theta
        p = self.out / "theta_payload.safetensors"
        export_from_theta(self.theta, str(p))
        check_carrier_sync(self.theta, p)
        v = self._post("/reload", {"path": str(p)}, timeout=600).get("version")
        print(f"### carrier sync: /reload -> server version {v}", flush=True)

    def _rollout(self, prompts):
        self._wake()
        self._reload()
        res = self._post("/generate", {"prompts": prompts, "max_new_tokens": self.cfg.max_new_tokens,
                                       "stop_token_ids": sorted(self.eos_ids)}, timeout=7200)["results"]
        self._sleep()
        return ([r["token_ids"] for r in res],
                [torch.tensor(r["logprobs"], dtype=torch.float32) for r in res])

    def _prompt(self, item):
        return tokenize_prompt(self.tokenizer, self.task.build_messages(item),
                               max_tokens=self.cfg.max_prompt_tokens)

    def _rewards(self, resp, flat_it, G, step):
        cfg = self.cfg
        items = [{"index": i, "prompt_index": i // G, "sample_index": i % G, "step": step, "item": it,
                  "text": self.tokenizer.decode(r, skip_special_tokens=True),
                  "n_completion_tokens": len(r), "truncated": len(r) >= int(cfg.max_new_tokens)}
                 for i, (r, it) in enumerate(zip(resp, flat_it))]
        t0 = time.time()
        out = list(self.task.reward_batch(items))
        if len(out) != len(items):
            raise RuntimeError(f"reward_batch returned {len(out)} results for {len(items)} items")
        vals, comps, n_failed, n_nonfinite = [], [], 0, 0
        for res in out:
            v, comp, status = as_reward(res)
            if not math.isfinite(v):
                comp = dict(comp, nonfinite_reward=repr(v))
                v, status = FAILURE_REWARD, "nonfinite"
                n_nonfinite += 1
            if status != "ok":
                n_failed += 1
            vals.append(v)
            comps.append(dict(comp, status=status))
        kinds = [self.task.classify_failure(c) for c in comps]
        mask = [k == "infra" for k in kinds]
        for comp, m, k in zip(comps, mask, kinds):
            comp["failure_kind"], comp["masked"] = k, bool(m)
        rewards = torch.tensor(vals, dtype=torch.float32)
        stats = {"n_items": len(items), "n_masked": int(sum(mask)),
                 "n_infra_failures": sum(1 for k in kinds if k == "infra"),
                 "n_candidate_failures": sum(1 for k in kinds if k == "candidate"),
                 "n_failed_rewards": n_failed, "n_nonfinite_rewards": n_nonfinite,
                 "reward_batch_mean": float(rewards.mean()), "reward_batch_min": float(rewards.min()),
                 "reward_batch_max": float(rewards.max()), "t_reward": time.time() - t0}
        for k, v in (getattr(self.task, "last_reward_batch_stats", None) or {}).items():
            stats.setdefault(k, v)
        self._dump_reward_components(step, items, vals, comps)
        live = rewards[torch.tensor([not m for m in mask], dtype=torch.bool)]
        stats["reward_mean_unmasked"] = float(live.mean()) if len(live) else 0.0
        return rewards, stats, mask

    def _dump_reward_components(self, step, items, vals, comps):
        if self._reward_comp_f is None:
            self._reward_comp_f = open(self.out / "reward_components.jsonl", "a")
        for it, v, comp in zip(items, vals, comps):
            self._reward_comp_f.write(json.dumps({
                "step": step, "index": it["index"], "prompt_index": it["prompt_index"],
                "sample_index": it["sample_index"], "id": (it["item"] or {}).get("id"),
                "reward": v, "n_completion_tokens": it["n_completion_tokens"],
                "truncated": it["truncated"], **comp}, default=str) + "\n")
        self._reward_comp_f.flush()

    def _advantages(self, rewards, n_prompts, G, resp, masked):
        keep_idx, adv_seq = [], []
        for g in range(n_prompts):
            r = rewards[g * G:(g + 1) * G]
            live = [j for j in range(G) if not masked[g * G + j]]
            if len(live) == G:
                adv, keep = group_advantages(r)
                adv = [adv[j].item() for j in range(G)]
            elif len(live) < 2:
                continue
            else:
                sub, keep = group_advantages(r[torch.tensor(live, dtype=torch.long)])
                adv = [0.0] * G
                for k, j in enumerate(live):
                    adv[j] = sub[k].item()
            if not keep:
                continue
            for j in live:
                if len(resp[g * G + j]) > 0:
                    keep_idx.append(g * G + j)
                    adv_seq.append(adv[j])
        return keep_idx, adv_seq

    def _micro(self, *lists):
        m = self.cfg.learner_micro_seqs
        for i in range(0, len(lists[0]), m):
            yield tuple(lst[i:i + m] for lst in lists)

    def _lp(self, theta, prompts, responses, *, grad, want_entropy=False):
        return batched_logprobs(self.flex, self.branch, theta, prompts, responses,
                                self.pad_id, grad=grad, want_entropy=want_entropy)

    def train_step(self, step):
        cfg = self.cfg
        n_items = len(self.task.train_items)
        idxs = epoch_indices(cfg.seed, n_items, min(cfg.prompts_per_step, n_items), step)
        items = [self.task.train_items[i] for i in idxs]
        prompt_ids = [self._prompt(it) for it in items]
        G = cfg.group_size
        flat_p = [p for p in prompt_ids for _ in range(G)]
        flat_it = [it for it in items for _ in range(G)]
        t0 = time.time()
        resp, behavior_lp = self._rollout(flat_p)
        t_roll = time.time() - t0
        rewards, rstats, mask = self._rewards(resp, flat_it, G, step)
        keep_idx, adv_seq = self._advantages(rewards, len(items), G, resp, mask)
        stats = {"step": step, "tasks": [it["id"] for it in items],
                 "reward_mean": float(rewards.mean()),
                 "groups_kept": len(adv_seq) / max(1, len(items) * G),
                 "resp_len": sum(len(r) for r in resp) / max(1, len(resp)),
                 "t_rollout": t_roll, "t_wake": self.t_wake_last, "t_sleep": self.t_sleep_last,
                 **rstats}
        if not keep_idx:
            stats["skipped"] = True
            stats.update(self._op_stats())
            return stats
        k_p = [flat_p[i] for i in keep_idx]
        k_r = [resp[i] for i in keep_idx]
        k_behavior = [behavior_lp[i] for i in keep_idx]
        n_seqs = len(k_p)
        ref_lp = []
        for mb_p, mb_r in self._micro(k_p, k_r):
            lp = self._lp(self.ref, mb_p, mb_r, grad=False)
            off = 0
            for r in mb_r:
                ref_lp.append(lp[off:off + len(r)].detach())
                off += len(r)
        self.opt.zero_grad(set_to_none=True)
        t0 = time.time()
        loss_stats, kl_sum, kl_n, tis_agg = {}, 0.0, 0, {}
        for mb_p, mb_r, mb_ref, mb_ix in self._micro(k_p, k_r, ref_lp, list(range(n_seqs))):
            m = len(mb_p)
            new_lp, new_ent = self._lp(self.theta, mb_p, mb_r, grad=True, want_entropy=True)
            olds = new_lp.detach()
            advs = torch.cat([torch.full((len(mb_r[j]),), adv_seq[mb_ix[j]])
                              for j in range(m)]).to(self.flex.device)
            w = torch.cat([torch.full((len(mb_r[j]),), 1.0 / (len(mb_r[j]) * n_seqs))
                           for j in range(m)]).to(self.flex.device)
            bl = torch.cat([k_behavior[ix] for ix in mb_ix]).to(self.flex.device)
            assert bl.shape == new_lp.shape, (bl.shape, new_lp.shape)
            tw, tis_s = tis_weights(new_lp, bl, cfg.tis_cap)
            w = w * tw
            for k_, v_ in tis_s.items():
                tis_agg[k_] = tis_agg.get(k_, 0.0) + v_ * len(tw)
            tis_agg["_n"] = tis_agg.get("_n", 0) + len(tw)
            loss, s = grpo_clip_loss(new_lp, olds, advs, w, cfg.eps_low, cfg.eps_high)
            loss = loss - cfg.beta_ent * (new_ent * w).sum()
            s["loss_entropy"] = float(new_ent.mean())
            refs = torch.cat(list(mb_ref)).to(self.flex.device)
            kl = k3_kl(new_lp, refs)
            loss = loss + cfg.beta_kl * (kl * w).sum()
            kl_sum += float(kl.mean()) * len(kl)
            kl_n += len(kl)
            loss.backward()
            if not self._audited:
                self._audited = True
                stats["base_params_with_grad"] = sum(
                    1 for p in self.flex.model.parameters() if p.grad is not None)
            loss_stats = s
        gmax = self.theta.clip_delta_grads_(cfg.grad_clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        stats.update(loss_stats)
        stats.update(kl_ref=kl_sum / max(1, kl_n), grad_max=gmax, t_learn=time.time() - t0,
                     gpu_peak_gb=torch.cuda.max_memory_allocated() / 2**30)
        if tis_agg.get("_n"):
            n_t = tis_agg.pop("_n")
            stats.update({k: v / n_t for k, v in tis_agg.items()})
        stats.update(self._op_stats())
        return stats

    def _op_stats(self):
        V = self.theta.V.detach().cpu()
        return {"Vnorm": float(V.norm()), "dV_norm": float((V - self.v0).norm())}

    def save(self, step):
        self.theta.to_artifact(stage="rl-last", rl_step=step).save(self.out / "artifact-last")

    def run(self):
        for step in range(1, self.cfg.steps + 1):
            stats = self.train_step(step)
            self.metrics_f.write(json.dumps(stats) + "\n")
            self.metrics_f.flush()
            print(f"[step {step}] " + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in stats.items() if k not in ("step", "tasks")), flush=True)
        self.save(self.cfg.steps)
        print(f"[done] {self.cfg.steps} steps", flush=True)


def build_argparser():
    ap = argparse.ArgumentParser()
    for f in SkillConfig.__dataclass_fields__.values():
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return ap


def main():
    cfg = SkillConfig(**vars(build_argparser().parse_args()))
    for k in ("task", "model_name", "artifact_path", "out_dir", "vllm_url", "base_cache"):
        assert getattr(cfg, k), f"--{k} is required"
    SkillTrainer(cfg).run()


if __name__ == "__main__":
    main()
