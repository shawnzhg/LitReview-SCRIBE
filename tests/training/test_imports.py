"""Tests that every training and kvskill module imports on CPU and that the planning task replays
candidates through the repo's runners."""

import importlib

import pytest

from training_env import REPO

MODULES = ["build_train_tasks", "reward_weights", "readout_reward", "planning_exit_readouts",
           "precompute_train_bands", "backbone", "planning_task", "kv_entry", "embed_service",
           "finalize_train_list", "kvskill.grpo", "kvskill.seed_init", "kvskill.init_theta",
           "kvskill.export", "kvskill.vllm_patch", "kvskill.vllm_server", "kvskill.rl.offload"]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(name)


def test_parse_path_uses_the_repo_harness():
    import planning_task as LP
    assert LP.RUNNERS == REPO / "scribe" / "harness" / "runners"
    assert LP.W.__file__.startswith(str(REPO / "scribe" / "harness" / "runners"))


def test_planning_registers_in_the_kvskill_registry():
    import planning_task as LP
    T = LP.register()
    assert T.TASKS["planning"] is LP.PlanningTask
