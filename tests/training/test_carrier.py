"""Tests the planning carrier on CPU without weights: parameter counts, the learner branch against the
vLLM branch on the exported payload, the closed-form solve of seed_init and the placement of the skill
text in the teacher prompt."""

import json
import types

import pytest
import torch

from training_env import MODEL, need

D_S, R, EXPECT = 1408, 32, 14_513_156


class _Cfg:

    def __init__(self, d):
        self._d, self.model_type = d, d.get("model_type")
        self._t = types.SimpleNamespace(**d["text_config"])

    def get_text_config(self):
        return self._t


@pytest.fixture(scope="module")
def theta_dir(tmp_path_factory):
    import transformers
    import kvskill.init_theta as IT
    need(MODEL / "config.json")
    cfg = _Cfg(json.load(open(MODEL / "config.json")))
    orig = transformers.AutoConfig.from_pretrained
    transformers.AutoConfig.from_pretrained = staticmethod(lambda *a, **k: cfg)
    try:
        out = tmp_path_factory.mktemp("initial_carrier")
        IT.build_theta(str(out), model_name=str(MODEL), d_s=D_S, r=R, seed=0)
    finally:
        transformers.AutoConfig.from_pretrained = orig
    return out


def test_param_count_is_14_513_156(theta_dir):
    from kvskill.artifact import KVSkillArtifact
    a = KVSkillArtifact.load(str(theta_dir))
    m = a.meta
    assert (m["cn_ds"], m["cn_r"], m["cn_dm"], m["cn_nlayers"]) == (D_S, R, 5120, 64)
    assert m["cn_depths"] == [15, 31, 47, 63]
    assert a.num_params() == EXPECT
    D, d_m = len(m["cn_depths"]), m["cn_dm"]
    assert 2 * D_S * d_m + 2 * D_S * R + d_m + D == EXPECT
    assert float(a.cn_v.abs().max()) == 0.0


def test_operator_trainable_count_and_update(theta_dir):
    from kvskill.artifact import KVSkillArtifact
    from kvskill.operator import SkillTheta
    a = KVSkillArtifact.load(str(theta_dir))
    th = SkillTheta(a, device="cpu", compute_dtype=torch.float32)
    assert th.d_s == D_S and th.num_trainable() == EXPECT
    assert th.V.requires_grad and torch.equal(th.V.detach(), a.cn_v)
    assert [len(g["params"]) for g in th.param_groups(1e-4, 1e-3)] == [4, 2]
    g = torch.Generator().manual_seed(2)
    with torch.no_grad():
        th.V.copy_(torch.randn(D_S, R, generator=g))
    n0 = float(th.V.norm())
    (th.V * torch.randn(D_S, R, generator=g)).sum().backward()
    with torch.no_grad():
        th.V -= 0.1 * th.V.grad
    assert abs(float(th.V.norm()) - n0) > 1e-3
    art = th.to_artifact(stage="t")
    assert art.num_params() == EXPECT and torch.equal(art.cn_v, th.V.detach().float().cpu())


def _fake_flex(n_layers=64):
    layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(n_layers)])

    class TM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers

        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return x
    return types.SimpleNamespace(text_model=TM(), model=None)


def test_branch_math_operator_equals_vllm_patch(theta_dir, tmp_path):
    from kvskill.artifact import KVSkillArtifact
    from kvskill.operator import SkillBranch, SkillTheta
    from kvskill import export as EX
    from kvskill import vllm_patch as VP
    from safetensors import safe_open
    a = KVSkillArtifact.load(str(theta_dir))
    flex = _fake_flex()
    th = SkillTheta(a, device="cpu", compute_dtype=torch.float32)
    br = SkillBranch(flex, th.depths)
    x = torch.randn(1, 5, 5120)
    br.bind(th)
    assert torch.equal(flex.text_model(x), x)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        th.V.copy_(torch.randn(D_S, R, generator=g) * 0.5)
    y = flex.text_model(x)
    assert not torch.equal(y, x)
    br.bind(None)
    d = tmp_path / "carrier"
    th.to_artifact(stage="t").save(str(d))
    p = EX.export_from_artifact(str(d), str(tmp_path / "payload.safetensors"))
    with safe_open(p, framework="pt", device="cpu") as f:
        meta = f.metadata()
        t = {k: f.get_tensor(k).float() for k in ("cn_a", "cn_b", "cn_k", "cn_v", "cn_w", "cn_bl")}
    assert (meta["cn_ds"], meta["cn_dm"], meta["cn_r"], meta["cn_nlayers"]) == ("1408", "5120", "32", "64")
    li = 0
    mod = torch.nn.Module()
    for name, val in (("_cn_a", t["cn_a"]), ("_cn_b", t["cn_b"]),
                      ("_cn_k", torch.nn.functional.normalize(t["cn_k"], dim=0)),
                      ("_cn_v", t["cn_v"]), ("_cn_w", t["cn_w"]), ("_cn_bl", t["cn_bl"][li])):
        mod.register_buffer(name, val.clone())
    h = torch.randn(1, 5, 5120)
    dv = VP._branch(mod, h)
    rms = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
    q = torch.nn.functional.normalize(rms @ th.A.T, dim=-1)
    ref = torch.sigmoid(rms @ th.wg + th.bl[li]).unsqueeze(-1) * (
        (q @ torch.nn.functional.normalize(th.K, dim=0)) @ th.V.T @ th.B.T)
    assert torch.allclose(dv, ref.detach(), rtol=1e-4, atol=1e-5)


def test_seed_init_closed_form(theta_dir):
    from kvskill.artifact import KVSkillArtifact
    from kvskill import seed_init as SI
    a = KVSkillArtifact.load(str(theta_dir))
    g = torch.Generator().manual_seed(0)
    states = {d: torch.randn(40, 5120, generator=g) for d in a.meta["cn_depths"]}
    U, lam, retained, spec = SI.derive_slots(states, a.cn_a, R)
    assert U.shape == (D_S, R) and torch.allclose(U.T @ U, torch.eye(R, dtype=U.dtype), atol=1e-8)
    C = torch.eye(R, dtype=torch.float64) + 0.01
    Yc = torch.randn(5120, R, generator=g, dtype=torch.float64)
    V, rel, sq = SI.solve_v(a, C, Yc, float(Yc.pow(2).sum()))
    assert V.shape == (D_S, R) and 0.0 <= rel <= 1.0


def test_skill_goes_to_the_system_segment_and_keeps_the_prompt_tail():
    from kvskill import seed_init as SI
    from kvskill.chat import tokenize_prompt
    from transformers import AutoTokenizer
    need(MODEL / "tokenizer.json")
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    msgs = [{"role": "system", "content": "You plan review outlines."},
            {"role": "user", "content": " ".join(f"claim c{i} reports finding {i}." for i in range(80))}]
    skill = "TEST SKILL: always end with a conclusion section."
    assert SI.insert_skill(msgs, skill)[0]["content"].endswith(skill) and msgs[0]["content"] == "You plan review outlines."
    assert SI.insert_skill(msgs[1:], skill)[0] == {"role": "system", "content": skill}
    student, teacher = SI.render_pair(tok, msgs, skill)
    assert student == tokenize_prompt(tok, msgs) and len(teacher) > len(student)
    assert SI.common_suffix_len(student, teacher) >= SI.SUFFIX_CAP
    assert skill in tok.decode(teacher)
    bad = [dict(m) for m in msgs]
    bad[-1]["content"] += SI.SEP + skill
    assert SI.common_suffix_len(student, tokenize_prompt(tok, bad)) < SI.SUFFIX_CAP
