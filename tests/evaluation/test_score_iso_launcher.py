"""Tests the argument handling of evaluation/board/score_iso.sh with a stub sbatch."""

import os
import subprocess
import sys
import types

import pytest

from eval_env import ROOT

SH = ROOT / "evaluation" / "board" / "score_iso.sh"


@pytest.fixture
def site(tmp_path):
    port, data = tmp_path / "port", tmp_path / "data"
    (port / "out" / "E1").mkdir(parents=True)
    (port / "out" / "E1" / "conformance.csv").write_text("system,task\n")
    (port / "out" / "rollouts" / "claude_science_mcp").mkdir(parents=True)
    runs_root = data / "tree" / "runs"
    (runs_root / "level5" / "SCRIBE" / "native_chain").mkdir(parents=True)
    (runs_root / "level6" / "SCRIBE" / "native_chain").mkdir(parents=True)
    (runs_root / "level6" / "OTHER" / "native_chain").mkdir(parents=True)
    env_sh = tmp_path / "site_env.sh"
    env_sh.write_text(f"export PY={sys.executable}\nexport CCBENCH_ROOT={port}\nexport CCBENCH_PARENT={data}\n")
    stub = tmp_path / "bin" / "sbatch"
    stub.parent.mkdir()
    stub.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$SBATCH_ARGS"\necho 4242\n')
    stub.chmod(0o755)
    runs = tmp_path / "runs"
    (runs / "fixinput").mkdir(parents=True)
    (runs / "samepool").mkdir()
    (tmp_path / "bundle").mkdir()
    env = {"PATH": f"{stub.parent}:{os.environ['PATH']}", "HOME": str(tmp_path), "CCBENCH_ENV": str(env_sh),
           "SCRIBE_RUNS_ROOT": str(runs_root),
           "ISO_ROOT": str(tmp_path / "iso"), "COMMERCIAL_OUT": str(tmp_path / "bundle"),
           "SBATCH_ARGS": str(tmp_path / "sbatch_args"), "PYTHONDONTWRITEBYTECODE": "1"}
    return types.SimpleNamespace(tmp=tmp_path, port=port, data=data, runs=runs, runs_root=runs_root, env=env)


def run(site, *args, **env):
    e = {k: v for k, v in {**site.env, **env}.items() if v is not None}
    return subprocess.run(["bash", str(SH), *map(str, args)], env=e, capture_output=True, text=True, cwd=site.tmp)


def refused(r, text):
    return r.returncode == 2 and "FATAL" in r.stdout and text in r.stdout


def test_the_launcher_parses():
    assert subprocess.run(["bash", "-n", str(SH)]).returncode == 0


def test_bad_modes_and_argument_counts_are_refused(site):
    assert refused(run(site), "usage")
    assert refused(run(site, "view", "openai_tool_loop", site.runs), "usage")
    assert refused(run(site, "agent", "openai_tool_loop"), "usage")
    assert refused(run(site, "native", "k", "5", "extra"), "usage")


def test_the_site_environment_and_the_output_root_are_required(site):
    assert refused(run(site, "agent", "openai_tool_loop", site.runs, CCBENCH_ENV=None), "CCBENCH_ENV")
    assert refused(run(site, "agent", "openai_tool_loop", site.runs, ISO_ROOT=None), "ISO_ROOT")
    inside = site.port / "out" / "iso"
    assert refused(run(site, "agent", "openai_tool_loop", site.runs, ISO_ROOT=str(inside)), "lies inside")
    assert not inside.exists()
    assert refused(run(site, "native", "newkey", "5", ISO_ROOT=str(site.data / "iso")), "lies inside")


def test_agent_arguments_are_checked_against_the_agent_table(site):
    assert refused(run(site, "agent", "../openai_tool_loop", site.runs), "unknown agent")
    assert refused(run(site, "agent", "no_such_agent", site.runs), "unknown agent")
    assert refused(run(site, "agent", "openai_tool_loop", site.runs, COMMERCIAL_OUT=None), "COMMERCIAL_OUT")
    assert refused(run(site, "agent", "openai_tool_loop", site.tmp / "nowhere"), "not a directory")
    assert refused(run(site, "agent", "elicit", site.runs), "does not run")
    (site.runs / "samepool").rmdir()
    assert refused(run(site, "agent", "openai_tool_loop", site.runs), "has no samepool/")
    (site.runs / "samepool").mkdir()
    assert refused(run(site, "agent", "claude_science", site.runs), "already has rollouts in the shared tables")


def test_native_arguments_are_checked(site):
    assert refused(run(site, "native", "claude_science_mcp", "5"), "already has rollouts in the shared tables")
    assert refused(run(site, "native", "a/b", "5"), "bare native label")
    assert refused(run(site, "native", "newkey", "x"), "run level number")
    assert refused(run(site, "native", "newkey", "7"), "level7 missing")
    assert refused(run(site, "native", "newkey", "6"), "exactly one is scored")
    assert refused(run(site, "native", "newkey", "5", SCRIBE_RUNS_ROOT=None), "SCRIBE_RUNS_ROOT")
    outside = site.tmp / "elsewhere" / "runs"
    (outside / "level5").mkdir(parents=True)
    assert refused(run(site, "native", "newkey", "5", SCRIBE_RUNS_ROOT=str(outside)), "does not lie inside CCBENCH_PARENT")


def test_valid_calls_are_submitted_with_their_arguments_and_write_nothing_else(site):
    r = run(site, "agent", "openai_tool_loop", site.runs)
    assert r.returncode == 0, r.stdout + r.stderr
    args = (site.tmp / "sbatch_args").read_text().splitlines()
    iso = site.tmp / "iso"
    assert f"--output={iso}/score_iso_%j.log" in args and args[-4:] == [str(SH), "agent", "openai_tool_loop", str(site.runs)]
    exp = [a.split("LITREVIEW_ROOT=", 1)[1] for a in args if a.startswith("--export=ALL,LITREVIEW_ROOT=")]
    assert len(exp) == 1 and os.path.realpath(exp[0]) == os.path.realpath(ROOT)
    r = run(site, "native", "newkey", "5")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (site.tmp / "sbatch_args").read_text().splitlines()[-4:] == [str(SH), "native", "newkey", "5"]
    assert list(iso.iterdir()) == []
    assert not list((ROOT / "evaluation" / "board").rglob("__pycache__"))


def _iso_dir(root, key):
    (root / "out" / "E13").mkdir(parents=True)
    (root / "out" / "E13" / "window_scores.parquet").write_bytes(b"rows")
    (root / "out" / "E1").mkdir()
    (root / "out" / "E1" / "conformance.csv").write_text("system,task\n")
    ro = root / "out" / "rollouts" / key
    ro.mkdir(parents=True)
    for i in range(50):
        (ro / f"pmcid_T{i}.json").write_text("{}")
    return root


def test_the_view_builder_reads_the_launchers_output_root(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("board_view_builder_iso", ROOT / "evaluation" / "board" / "view_builder.py")
    vb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vb)
    log = ("### shared tables and the shared embed cache unchanged (3 files); private cache misses: 0\n"
           "### end=<date> rc=0 (rows: out/E13/window_scores.parquet)\n")
    d = _iso_dir(tmp_path / "claude_science_iso_77", "claude_science_mcp")
    assert vb.find_ext("claude_science_mcp", str(d))["dir"] is None
    (d / vb.ISO_LOG).write_text(log)
    f = vb.find_ext("claude_science_mcp", str(d))
    assert f["dir"] == str(d) and f["log"] == str(d / "score_iso.log") and f["level_gate"]["n_units_linked"] == 50
    n = _iso_dir(tmp_path / "newkey_iso_78", "newkey")
    (n / "level_gate.json").write_text('{"record": {"level": 5, "retrieval_budget": "cap"}, "retrieval_budget": "cap", '
                                       '"problems": [], "n_units_linked": 50, "units": "report", "allow_partial": false}')
    (n / vb.ISO_LOG).write_text(log)
    vb.declare_native("newkey", str(n))
    g = vb.find_iso("newkey", 5, str(n))
    assert g["dir"] == str(n) and g["level_gate"]["record"]["level"] == 5
    assert vb.find_ext("claude_science_mcp")["dir"] is None and vb.find_iso("newkey", 5)["dir"] is None
