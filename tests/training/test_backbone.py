"""Tests the backbone pin: building the backbone directory from a snapshot with the shipped
chat-template patch, the checks that refuse modified weights or templates, and the GRPO guard against
trainable backbone parameters."""

import hashlib
import json
import os
import types

import pytest
import torch

from training_env import MODEL, PIN, hf_snapshot, need

STOCK = "a\n{%- if add_generation_prompt %}\n    {%- if thinking %}\n        x\n    {%- endif %}\n{%- endif %}"
EDITED = "a\n{%- if add_generation_prompt %}\n    y\n{%- endif %}"
PATCH = ("--- a/chat_template.jinja\n+++ b/chat_template.jinja\n@@ -1,6 +1,4 @@\n a\n"
         " {%- if add_generation_prompt %}\n-    {%- if thinking %}\n-        x\n-    {%- endif %}\n"
         "+    y\n {%- endif %}\n\\ No newline at end of file\n")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _snapshot(tmp_path):
    blobs, snap = tmp_path / "blobs", tmp_path / "snap"
    blobs.mkdir()
    snap.mkdir()
    shards = {}
    for i in (1, 2):
        data = f"weights {i}".encode()
        (blobs / _sha(data)).write_bytes(data)
        name = f"model-0000{i}-of-00002.safetensors"
        os.symlink(blobs / _sha(data), snap / name)
        shards[name] = _sha(data)
    (snap / "config.json").write_text('{"x": 1}')
    (snap / "chat_template.jinja").write_text(STOCK)
    (tmp_path / "t.patch").write_text(PATCH)
    pin = {"hf_id": "x/y", "revision": "r", "schema": "scribe_backbone/1", "chat_template_patch": "t.patch",
           "source_sha256": {"chat_template.jinja": _sha(STOCK.encode())},
           "sha256": {"chat_template.jinja": _sha(EDITED.encode()), "config.json": _sha(b'{"x": 1}')},
           "weight_shards": shards}
    (tmp_path / "pin.json").write_text(json.dumps(pin))
    return snap, tmp_path / "pin.json"


def test_apply_patch_handles_a_missing_final_newline():
    import backbone as BB
    assert BB.apply_patch(STOCK, PATCH) == EDITED
    with pytest.raises(ValueError, match="does not apply"):
        BB.apply_patch(STOCK.replace("thinking", "other"), PATCH)


def test_prepare_then_check_on_a_synthetic_snapshot(tmp_path):
    import backbone as BB
    snap, pin = _snapshot(tmp_path)
    rec = BB.prepare(snap, tmp_path / "model", pin)
    assert rec["pinned"] and rec["n_shards"] == 2
    assert (tmp_path / "model" / "chat_template.jinja").read_text() == EDITED
    assert BB.check(tmp_path / "model", pin)["pinned"]
    with pytest.raises(ValueError, match="not empty"):
        BB.prepare(snap, tmp_path / "model", pin)
    with pytest.raises(ValueError, match="differs from the pin"):
        BB.check(snap, pin)


def test_modified_weights_or_template_are_refused(tmp_path):
    import backbone as BB
    snap, pin = _snapshot(tmp_path)
    (snap / "chat_template.jinja").write_text(STOCK + " ")
    with pytest.raises(ValueError, match="differs from the pin"):
        BB.prepare(snap, tmp_path / "m1", pin)
    (snap / "chat_template.jinja").write_text(STOCK)
    shard = snap / "model-00002-of-00002.safetensors"
    os.remove(shard)
    shard.write_bytes(b"other weights")
    with pytest.raises(ValueError, match="not the pinned weight"):
        BB.check_source(snap, pin)
    os.remove(shard)
    with pytest.raises(ValueError, match="weight shards"):
        BB.check_source(snap, pin)


def test_the_shipped_pin_builds_the_backbone_from_the_hf_snapshot(tmp_path):
    import backbone as BB
    snap = hf_snapshot()
    need(snap / "chat_template.jinja")
    rec = BB.prepare(snap, tmp_path / "model")
    assert rec["pinned"] and rec["n_shards"] == 18
    assert json.loads(PIN.read_text())["sha256"]["chat_template.jinja"] == BB.sha256_file(
        tmp_path / "model" / "chat_template.jinja")


def test_the_site_backbone_matches_the_pin():
    import backbone as BB
    need(MODEL / "config.json")
    assert BB.check(MODEL)["n_shards"] == 18


def test_grpo_guard_asserts_frozen_backbone(monkeypatch):
    import kvskill.grpo as G
    import backbone as BB

    class Dummy:
        def __init__(self, cfg):
            m = torch.nn.Linear(4, 4)
            m.requires_grad_(bool(cfg.trainable))
            self.flex = types.SimpleNamespace(model=m)
            self.audit = cfg.audit

        def train_step(self, step):
            return {"step": step, "base_params_with_grad": self.audit}

    monkeypatch.setattr(G, "SkillTrainer", Dummy)
    monkeypatch.delenv("SCRIBE_BACKBONE_PIN", raising=False)
    BB.install_grpo_guard()
    ok = G.SkillTrainer(types.SimpleNamespace(model_name=str(MODEL), trainable=False, audit=0))
    assert ok.train_step(1)["base_params_with_grad"] == 0
    with pytest.raises(RuntimeError, match="require grad"):
        G.SkillTrainer(types.SimpleNamespace(model_name=str(MODEL), trainable=True, audit=0))
    bad = G.SkillTrainer(types.SimpleNamespace(model_name=str(MODEL), trainable=False, audit=3))
    with pytest.raises(RuntimeError, match="base_params_with_grad = 3"):
        bad.train_step(1)


def test_kv_entry_installs_the_guard_for_grpo_only():
    import kv_entry
    txt = open(kv_entry.__file__).read()
    assert "install_grpo_guard()" in txt and 'if cmd == "grpo":' in txt
    assert set(kv_entry.MODS) == {"grpo", "seed_init"}
