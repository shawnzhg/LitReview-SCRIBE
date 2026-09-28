"""Static checks: every harness module imports, every launcher passes bash -n, accepts seeds 0, 1 and
2, carries the pipeline settings and runs the repo's code."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

from harness_env import ROOT

L = ROOT / "scribe" / "launchers"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rel", [
    "scribe/harness/runners/runner.py", "scribe/harness/runners/windows.py", "scribe/harness/runners/run_campaign.py",
    "scribe/harness/runners/pool_backend.py", "scribe/retrieval/budget.py",
    "scribe/harness/runners/run_campaign_pool.py", "scribe/retrieval/run_budgeted_campaign.py",
    "scribe/levers/writing_levers.py", "scribe/launchers/generate.py", "scribe/launchers/same_pool_prepare.py",
    "scribe/launchers/same_pool_finalize.py", "scribe/retrieval/run_acquisition.py",
])
def test_module_imports(rel):
    p = ROOT / rel
    import sys
    for d in (p.parent, ROOT / "scribe" / "retrieval"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    _load(p, "t_" + p.stem)


LAUNCHERS = ["run_fixed_input.slurm", "run_same_pool_acquisition.slurm", "same_pool_shard.slurm",
             "same_pool_prepare.slurm", "run_same_pool.sh"]


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_syntax(name):
    r = subprocess.run(["bash", "-n", str(L / name)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def _seed_check(path, marker):
    line = next(l for l in (L / path).read_text().splitlines() if marker in l and "SEEDS" in l)
    return line


@pytest.mark.parametrize("path,marker,ok,bad", [
    ("run_fixed_input.slurm", "must be distinct seeds", ["0", "1", "2", "0,1,2"], ["3", "0,0", "", "0,1,"]),
    ("run_same_pool_acquisition.slurm", "one seed per same-pool level", ["0", "1", "2"], ["3", "0,1", ""]),
])
def test_launchers_accept_seeds_zero_one_two(path, marker, ok, bad):
    check = _seed_check(path, marker)
    for seeds in ok + bad:
        r = subprocess.run(["bash", "-c", check + "\necho accepted"], env={"SEEDS": seeds, "PATH": os.environ["PATH"]},
                           capture_output=True, text=True)
        assert ("accepted" in r.stdout) == (seeds in ok), (path, seeds, r.stdout, r.stderr)


def test_fixed_input_launcher_release_knobs():
    s = (L / "run_fixed_input.slurm").read_text()
    assert "SYSTEMS=${SYSTEMS:-SCRIBE}" in s and "MAXLEN=65536" in s
    assert "egress_guard.sh \"$RUN/egress_proof.json\" bash $RUN/inner.sh" in s
    assert 'echo "--tasks-file $TASKS_FILE" || echo "--tasks $TASKS"' in s


def test_same_pool_acquisition_launcher_release_knobs():
    s = (L / "run_same_pool_acquisition.slurm").read_text()
    assert "SYSTEMS=${SYSTEMS:-SCRIBE}" in s and "ARM=same_pool_${TAG}_cap" in s
    assert 'EXPECT_IDENTITY=0; [ "$(readlink -f "$THETA")" = "$(readlink -f "$SCRIBE_THETA0")" ] && EXPECT_IDENTITY=1' in s
    assert "--ranking-audit" in s and "$RETRIEVAL/acquisition_audit.py" in s
    assert "--phase generation" not in s
    assert "egress_guard.sh \"$RUN/egress_proof.json\" bash $RUN/inner.sh" in s
    assert re.search(r"--system \$SYSTEMS --out \$RUN/acquisition_audit\.json", s)
    assert '\\"run\\": \\"$RUN\\"' in s


def test_same_pool_generation_chain_release_knobs():
    sub = (L / "run_same_pool.sh").read_text()
    assert '[ "$SYSTEM" = SCRIBE ] ||' in sub and "LV.load(lf)" in sub and "GEN_MAXLEN=65536" in sub
    assert "_level_claim_*.json" in sub and '"seed": $SEED' in sub
    assert "$L/same_pool_prepare.slurm" in sub and "$L/same_pool_shard.slurm" in sub and "$L/same_pool_finalize.py" in sub
    shard = (L / "same_pool_shard.slurm").read_text()
    assert "$L/generate.py --phase generation --systems $SYSTEM --mode native_chain --level $LEVEL --seeds $SEED" in shard
    prep = (L / "same_pool_prepare.slurm").read_text()
    assert "--system $SYSTEM --out $RUN/provenance.json" in prep and "--seeds $SEED" in prep


def test_generation_driver_runs_the_repo_harness_in_place():
    s = (L / "run_fixed_input.slurm").read_text()
    assert "DRIVER=$LITREVIEW_ROOT/scribe/launchers/generate.py" in s
    assert "LEVERS=$LITREVIEW_ROOT/scribe/levers/writing_levers.json" in s
    assert "$SERVING/route_check.py" in s and "carrier_digest_ext.DigestWorkerExtension" in s
    from harness_env import RUNNERS
    d = _load(L / "generate.py", "t_driver_generate")
    assert d.RUNNERS == str(RUNNERS) and d.LEVERS == str(ROOT / "scribe" / "levers")


def site_roots():
    out = set()
    for k, v in os.environ.items():
        if k.startswith(("SCRIBE_", "CCBENCH_", "LITREVIEW_")) and v.startswith("/") and len(Path(v).parts) > 2:
            out.add("/" + Path(v).parts[1] + "/")
    return sorted(out)


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launchers_carry_no_site_values_and_no_code_copies(name):
    s = (L / name).read_text()
    for bad in site_roots() + ["--account", "--partition", "-A ", "--output=/", "$RUN/code/"]:
        assert bad not in s, (name, bad)
    assert "third_party/kvskill" in s or name in ("same_pool_prepare.slurm", "run_same_pool.sh")
    for line in s.splitlines():
        if line.startswith("#SBATCH"):
            assert all(o.split("=")[0] in ("--nodes", "--ntasks", "--cpus-per-task", "--mem", "--gres", "--time")
                       for o in line.split()[1:]), line
            assert "gpu:" not in line or line.split("gpu:")[1][0].isdigit(), line


def test_serving_marker_is_read_by_the_route_check():
    ext = (ROOT / "scribe" / "serving" / "carrier_digest_ext.py").read_text()
    rc = (ROOT / "scribe" / "serving" / "route_check.py").read_text()
    assert 'print(f"[carrier_digest] staged canon=' in ext and r"\[carrier_digest\] staged canon=" in rc
