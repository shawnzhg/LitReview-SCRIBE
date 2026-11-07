"""Tests the three launcher configs by sourcing them in bash: carriers per stage, the one
backbone, refusal of a missing carrier, the default seed and the repo's lever file."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from harness_env import ROOT, LEVERS_FILE

CFG = ROOT / "scribe" / "launchers" / "configs"
SITE = {"SCRIBE_THETA0": "/placeholder/theta0", "SCRIBE_MODEL_DIR": "/placeholder/backbone"}


def _source(name, extra=None, site=True):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LITREVIEW_ROOT": str(ROOT)}
    if site:
        env.update(SITE)
    env.update(extra or {})
    r = subprocess.run(["bash", "-c", f"set -a; source {CFG / name} || exit 7; set +a; env"], capture_output=True,
                       text=True, env=env)
    return r.returncode, dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l), r.stderr


def _common(v, key="LEVERS"):
    assert v["SYSTEMS"] == v["SYSTEM"] == "SCRIBE" and v["SEEDS"] == "0"
    assert v[key].endswith("writing_levers.json")
    assert json.loads(Path(v[key]).read_text())["levers"] == json.loads(LEVERS_FILE.read_text())["levers"]


def test_untrained_is_the_initial_carrier_everywhere():
    rc, v, err = _source("scribe_untrained.env")
    assert rc == 0, err
    _common(v)
    t0 = v["SCRIBE_THETA0"]
    assert v["THETA"] == v["THETA_PLANNING"] == t0


def test_seed_can_be_set_for_the_replicates():
    for seed in ("1", "2"):
        rc, v, err = _source("scribe_untrained.env", {"SEEDS": seed})
        assert rc == 0 and v["SEEDS"] == seed, err


def test_trained_is_planning_only_on_the_same_backbone():
    rc, v, err = _source("scribe_trained.env", {"SCRIBE_PLANNING_CARRIER": "/placeholder/planning_carrier"})
    assert rc == 0, err
    _common(v)
    t0 = v["SCRIBE_THETA0"]
    assert v["THETA_PLANNING"] == "/placeholder/planning_carrier"
    assert v["THETA"] == t0


def test_trained_refuses_a_missing_carrier():
    rc, v, err = _source("scribe_trained.env")
    assert rc != 0 and "SCRIBE_PLANNING_CARRIER" in err


def test_configs_refuse_missing_site_variables():
    rc, v, err = _source("scribe_untrained.env", site=False)
    assert rc != 0 and "SCRIBE_THETA0" in err
    roots = {"/" + Path(v).parts[1] + "/" for k, v in os.environ.items()
             if k.startswith("SCRIBE_") and v.startswith("/") and len(Path(v).parts) > 2}
    for name in ("scribe_untrained.env", "scribe_trained.env", "scribe_luna.env"):
        text = (CFG / name).read_text()
        assert not [r for r in roots if r in text], name


def test_luna_config_is_the_same_pipeline():
    rc, v, err = _source("scribe_luna.env", site=False)
    assert rc == 0, err
    _common(v, "SCRIBE_LEVERS")
    assert v["LUNA_MODEL"] == "gpt-5.6-luna" and Path(v["SCRIBE_LEVERS"]) == LEVERS_FILE and "LEVERS" not in v
