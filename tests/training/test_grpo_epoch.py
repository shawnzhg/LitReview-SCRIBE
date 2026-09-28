"""Tests that GRPO draws its tasks from a per-epoch permutation, so that every training task is used
once per epoch, and that the launcher and the list builder fix one epoch over 192 tasks."""

import inspect
import re

from training_env import REPO

from kvskill import grpo as G


def test_one_epoch_uses_every_training_task_exactly_once():
    n, k, steps = 192, 4, 48
    drawn = [i for s in range(1, steps + 1) for i in G.epoch_indices(17, n, k, s)]
    assert sorted(drawn) == list(range(n))
    assert drawn != list(range(n))
    nxt = [i for s in range(steps + 1, 2 * steps + 1) for i in G.epoch_indices(17, n, k, s)]
    assert sorted(nxt) == list(range(n)) and nxt != drawn
    assert G.epoch_indices(17, n, k, 5) == G.epoch_indices(17, n, k, 5)
    assert G.epoch_indices(18, n, k, 1) != G.epoch_indices(17, n, k, 1)


def test_the_train_step_uses_the_epoch_permutation():
    src = inspect.getsource(G.SkillTrainer.train_step)
    assert "epoch_indices(cfg.seed" in src


def test_launcher_and_list_fix_one_epoch_over_192_tasks():
    import finalize_train_list as FT
    s = (REPO / "scribe" / "training" / "launch" / "03_grpo.slurm").read_text()
    m = re.search(r"STEPS=(\d+); PPS=(\d+);.*MAX_PROMPT=(\d+)", s)
    steps, pps, max_prompt = (int(x) for x in m.groups())
    n = int(re.search(r"N_TRAIN_ITEMS=(\d+)", s).group(1))
    assert steps * pps == n == FT.N_TRAIN == 192
    assert max_prompt == FT.MAX_PROMPT_TOKENS == G.SkillConfig().max_prompt_tokens
    assert '[ $(( STEPS * PPS )) -eq "$N_TRAIN_ITEMS" ]' in s
    assert '[ "$(grep -c . "$TASKS_FILE")" -eq "$N_TRAIN_ITEMS" ]' in s
