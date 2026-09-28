"""Tests that the manifest prompt hash covers every prompt template of the SCRIBE windows and of the
lever writer."""

from __future__ import annotations

import hashlib

import pytest

from harness_env import R, W, LV, TASK, MockClient, install_levers

BASE_PREFIX = "sha256:c9992283"
LEVER_WU_PREFIX = "efba50fc"
LEVER_WU = W.WRITE_USER + "\n\n" + LV.PARA_TEXT.replace("{", "{{").replace("}", "}}")


def test_base_hash_covers_every_window_prompt():
    from common import sha256_str
    prompts = {n for n, v in vars(W).items() if n.isupper() and isinstance(v, str)} - {"BUDGET_RULE"}
    assert set(R.PROMPT_TEMPLATES) == prompts and len(R.PROMPT_TEMPLATES) == len(prompts) == 10
    assert R.prompt_hash() == R.prompt_hash(write_user=W.WRITE_USER)
    assert R.prompt_hash() == sha256_str("".join(getattr(W, n) for n in R.PROMPT_TEMPLATES))
    assert R.prompt_hash().startswith(BASE_PREFIX)


@pytest.mark.parametrize("name", ["SYN_SYS", "SYN_EXTRACT", "SYN_REL", "SYN_CROSS", "PLAN_SYS", "PLAN_USER"])
def test_changing_a_synthesis_or_planning_prompt_changes_the_hash(name, monkeypatch):
    base, base_lever = R.prompt_hash(), R.prompt_hash(write_user=LEVER_WU)
    monkeypatch.setattr(W, name, getattr(W, name) + " ")
    assert R.prompt_hash() != base and R.prompt_hash(write_user=LEVER_WU) != base_lever
    rec = install_levers(monkeypatch)
    assert rec["prompt_hash"] == R.prompt_hash(write_user=LEVER_WU) != base_lever


def test_hash_changes_when_levers_are_installed(levers_installed):
    rec = levers_installed
    assert rec["write_user_sha256"] == hashlib.sha256(LEVER_WU.encode()).hexdigest()
    assert rec["write_user_sha256"].startswith(LEVER_WU_PREFIX)
    assert R.PROMPT_HASH == rec["prompt_hash"] == R.prompt_hash(write_user=LEVER_WU) != R.prompt_hash()


def test_manifests_carry_the_lever_hash(levers_installed, task_runs):
    out, mans = R.run_generation(TASK, "SCRIBE", 0, MockClient(), level=7, mode="bundle_entry",
                                 entry_mode="bundle_entry")
    assert mans and all(m["prompt_hash"] == levers_installed["prompt_hash"] for m in mans)
    assert not any(m["prompt_hash"].startswith(BASE_PREFIX) for m in mans)
