"""Pytest fixtures of the training tests: reward targets built from a synthetic planning-window
cells table."""

import pytest

from training_env import synthetic_cells


@pytest.fixture()
def targets(tmp_path):
    import reward_weights as RW
    cells = tmp_path / "cells.csv"
    cells.write_text(synthetic_cells())
    out = tmp_path / "targets.json"
    assert RW.main(["--cells", str(cells), "--out", str(out)]) == 0
    return {"targets": out, "cells": cells}
