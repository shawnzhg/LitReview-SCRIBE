"""CPU tests of the published-pipeline drivers and input builders on synthetic tasks, and of the
launchers' refusals and integrity checks."""

import ast
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import numpy as np
import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BL = ROOT / "scribe" / "harness" / "baselines"
EMBED = ROOT / "biolitbench" / "pool" / "embed"
BRIDGE = ROOT / "scribe" / "harness" / "runners" / "sock_bridge.py"
SHARED = [BRIDGE, EMBED / "assemble_db.py", EMBED / "cutoff_views.py", EMBED / "tap" / "sitecustomize.py",
          EMBED / "tap" / "pool_selector.py", EMBED / "tap" / "pool_docstore.py"]
PY_FILES = sorted(BL.rglob("*.py")) + SHARED
SH_FILES = sorted(list(BL.glob("*.sh")) + list(BL.glob("*.slurm")))
SF_COMMIT = "9114a0b7895a0f7eb614938d9bc0c956cf25245b"
LIRA_COMMIT = "2bc77a6e7b30586343119bd104cc07ebaec380a1"
DT_COMMIT = "9d7b0371c085e9311ddec483ed39768c0bd9fe99"


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, rel if isinstance(rel, Path) else BL / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean_env(**extra):
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
    env.update(extra)
    return env


def stubs(tmp_path, head, diff=b"", gpu=False):
    d = tmp_path / "stubbin"
    d.mkdir()
    (tmp_path / "fake.diff").write_bytes(diff)
    (d / "git").write_text("#!/bin/bash\n"
                           f'case "$3" in rev-parse) echo {head} ;; diff) cat "{tmp_path / "fake.diff"}" ;; *) exit 1 ;; esac\n')
    (d / "git").chmod(0o755)
    (d / "nvidia-smi").write_text("#!/bin/bash\n" + ("exit 0\n" if gpu else "exit 1\n"))
    (d / "nvidia-smi").chmod(0o755)
    return str(d)


def fake_python(tmp_path, version):
    p = tmp_path / f"py{version}"
    p.write_text("#!/bin/bash\n"
                 f'if [ "$1" = -c ] && [[ "$2" == *version_info* ]]; then echo {version}; exit 0; fi\n'
                 f'exec "{sys.executable}" "$@"\n')
    p.chmod(0o755)
    return str(p)


def campaign_env(tmp_path, bl, stub_path, evidence="pool", **extra):
    app = tmp_path / "app"
    app.mkdir(exist_ok=True)
    env = clean_env(LITREVIEW_ROOT=str(ROOT), RUNS_ROOT=str(tmp_path / "runs"),
                    BASELINE_INPUTS=str(tmp_path), TASKLIST=str(tmp_path), CUTOFFS=str(tmp_path),
                    POOL_INDEX=str(tmp_path), POOL_MANIFEST=str(tmp_path), HOSTPY=sys.executable,
                    OPENAI_KEY_FILE=str(tmp_path / "k"), SCRIBE_RUN_PY=sys.executable, HF_HOME=str(tmp_path),
                    POOL_GOLD_DIR=str(tmp_path),
                    POOL_PY=sys.executable, BL=bl, EVIDENCE=evidence, **extra)
    env["PATH"] = stub_path + ":" + env["PATH"]
    return env


def test_expected_files_present():
    names = {str(p.relative_to(BL)) for p in BL.rglob("*") if p.is_file()}
    for n in ("campaign.slurm", "drtulu_pool.slurm", "drtulu_pool_inner.sh", "drtulu_env.sh",
              "drtulu_task.py", "serving_proof.py", "make_baseline_inputs.py",
              "make_lira_input.py", "make_allowlists_idx.py", "README.md"):
        assert n in names, n
    for p in SHARED:
        assert p.is_file(), p


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: p.name)
def test_python_header_only(path):
    src = path.read_text()
    tree = ast.parse(src)
    assert ast.get_docstring(tree)
    n_doc = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                n_doc += 1
    assert n_doc == 1
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT:
            assert t.start[0] == 1 and t.string.startswith("#!")


@pytest.mark.parametrize("path", SH_FILES, ids=lambda p: p.name)
def test_shell_syntax(path):
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_no_site_paths():
    for p in list(BL.rglob("*")) + SHARED:
        if p.is_file() and p.suffix in (".py", ".sh", ".slurm", ".md"):
            s = p.read_text()
            for bad in (f"/{d}/" for d in ("nfs", "scratch", "sw", "home")):
                assert bad not in s, (p.name, bad)


def test_no_automatic_reruns():
    for p in SH_FILES:
        s = p.read_text()
        assert ".attempt" not in s and "RETRY" not in s and "retry" not in s, p.name
    assert "retried" not in (BL / "drtulu_task.py").read_text()


def test_refs_record_and_task_list(tmp_path):
    m = load("make_baseline_inputs.py", "baseline_mbi")
    spec = {"question": "Topic X", "publication_cutoff": 2019,
            "output_spec": {"target_words": 4000}}
    bundle = {"papers": [{"paper_id": "11", "title": "A", "doi": "10.1/a"},
                         {"paper_id": "22", "title": None}]}
    bundle["papers"] += [{"paper_id": "33", "title": "same year"}, {"paper_id": "44", "title": "undated"},
                         {"paper_id": "55", "title": "the review"}]
    years = {"11": 2010, "22": 2018, "33": 2019, "55": 2001}
    d = m.refs_record("pmcid_PMC1", spec, bundle, {"11": "abs one"}, years, "55")
    assert d["topic"] == "Topic X" and d["cutoff_year"] == 2019 and d["target_words"] == 4000
    assert d["n_refs"] == 2 and d["task_id"] == "pmcid_PMC1" and d["source"] == "gold"
    assert d["refs"][0] == {"pmid": "11", "doi": "10.1/a", "title": "A", "year": 2010, "abstract": "abs one"}
    assert d["refs"][1]["abstract"] == "" and d["refs"][1]["title"] == "" and d["refs"][1]["year"] == 2018
    gold = tmp_path / "gold"
    gold.mkdir()
    (gold / "pmcid_PMC1.json").write_text(json.dumps({"review_pmid": 55}))
    assert m.review_pmid(gold, "pmcid_PMC1") == "55"
    (gold / "pmcid_PMC2.json").write_text(json.dumps({}))
    with pytest.raises(SystemExit):
        m.review_pmid(gold, "pmcid_PMC2")
    j = tmp_path / "c.json"
    j.write_text(json.dumps({"tasks": ["b", "a"]}))
    assert m.task_list(j) == ["b", "a"]
    jl = tmp_path / "i.jsonl"
    jl.write_text('{"task_id": "x"}\n\n{"task_id": "y"}\n')
    assert m.task_list(jl) == ["x", "y"]


def test_lira_adapter(tmp_path):
    m = load("make_lira_input.py", "baseline_lira")
    for t, refs in (("pmcid_PMC2", [{"pmid": "5", "title": "T5", "abstract": "a5", "year": 2000}]),
                    ("pmcid_PMC1", [{"pmid": "7", "title": None, "abstract": "a7", "year": 2000},
                                    {"pmid": "8", "title": "T8", "abstract": " ", "year": 2000},
                                    {"pmid": "x9", "title": "T9", "abstract": "a9", "year": 2000},
                                    {"pmid": "10", "title": "T10", "abstract": "a10", "year": 2019},
                                    {"pmid": "11", "title": "T11", "abstract": "a11", "year": None}]),
                    ("pmcid_PMC3", [{"pmid": "4", "title": "T4", "abstract": "", "year": 2000}])):
        (tmp_path / "in" / t).mkdir(parents=True)
        (tmp_path / "in" / t / "refs.json").write_text(
            json.dumps({"topic": "topic " + t, "task_id": t, "cutoff_year": 2019, "refs": refs}))
    surveys, dropped = m.lira_surveys(tmp_path / "in")
    assert dropped == 5
    assert [s["task_id"] for s in surveys] == ["pmcid_PMC1", "pmcid_PMC2"]
    assert [s["id"] for s in surveys] == [0, 1]
    assert surveys[0]["references"] == [{"num": 1, "id": 7, "title": "", "content": "a7"}]
    assert surveys[1]["references"][0]["id"] == 5
    m.main(["--inputs", str(tmp_path / "in"), "--out", str(tmp_path / "lira")])
    got = json.loads((tmp_path / "lira" / "scireviewgen" / "full_data_abs.json").read_text())
    assert got == surveys


def test_index_space_allowlists_are_autosurvey_only(tmp_path):
    m = load("make_allowlists_idx.py", "baseline_idx")
    allow = tmp_path / "allow"
    allow.mkdir()
    (allow / "t1.json").write_text(json.dumps(["100", "200", "999"]))
    pool = tmp_path / "pool"
    (pool / "autosurvey_db").mkdir(parents=True)
    (pool / "autosurvey_db" / "arxivid_to_index_abs.json").write_text(json.dumps({"100": 3, "200": 1, "999": 5}))
    views = tmp_path / "views"
    views.mkdir()
    np.save(views / "2019.t1.npy", np.array([1, 3, 4], dtype="int64"))
    args = ["--allowlists", str(allow), "--pool-index", str(pool), "--cutoff-views", str(views)]
    m.main(["--arm", "autosurvey"] + args)
    assert np.load(allow / "t1.autosurvey.idx.npy").tolist() == [1, 3]
    with pytest.raises(SystemExit):
        m.main(["--arm", "surveyforge"] + args)
    (views / "2019.t1.npy").unlink()
    with pytest.raises(SystemExit):
        m.main(["--arm", "autosurvey"] + args)


def test_cutoff_views(tmp_path):
    m = load(EMBED / "cutoff_views.py", "baseline_cut")
    db, gold = tmp_path / "db", tmp_path / "gold"
    db.mkdir()
    gold.mkdir()
    np.save(db / "id_year.npy", np.array([1999, 2005, 0, 2019, 2020, 1980], dtype="int32"))
    np.save(db / "id_pmid.npy", np.array([11, 22, 33, 44, 55, 66], dtype="int64"))
    (gold / "t1.json").write_text(json.dumps({"review_pmid": 22}))
    (gold / "t2.json").write_text(json.dumps({"review_pmid": 66}))
    tc = tmp_path / "tc.json"
    tc.write_text(json.dumps({"t1": 2019, "t2": 2005}))
    m.build(str(tc), str(gold), str(db), str(tmp_path / "cut"))
    assert np.load(tmp_path / "cut" / "2019.t1.npy").tolist() == [1, 6]
    assert np.load(tmp_path / "cut" / "2005.t2.npy").tolist() == [1]
    meta = json.loads((tmp_path / "cut" / "task_to_cutoff.json").read_text())
    assert meta["tasks"] == {"t1": {"cutoff": 2019, "review_pmid": "22"}, "t2": {"cutoff": 2005, "review_pmid": "66"}}


def test_pool_selector_batch_filter():
    m = load(EMBED / "tap" / "pool_selector.py", "baseline_sel")
    out = m.batch_index_filter({"a": 5, "b": 9}, ["b", "a"])
    sel = out["id_selector"]
    assert sel.is_member(5) and sel.is_member(9) and not sel.is_member(6)


def test_sock_bridge_helpers():
    m = load(BRIDGE, "baseline_bridge")
    assert m.parse_hostport("8080") == ("127.0.0.1", 8080)
    assert m.parse_hostport("0.0.0.0:31") == ("0.0.0.0", 31)
    assert m.parse_hostport(":31") == ("127.0.0.1", 31)
    m.check_sun_path("/tmp/bl_lanes/1/0.sock")
    with pytest.raises(SystemExit):
        m.check_sun_path("/tmp/" + "x" * 120 + ".sock")


def test_drtulu_driver_helpers(tmp_path):
    m = load("drtulu_task.py", "baseline_dt")
    base = ["--task-id", "pmcid_PMC1", "--task-input", "r.json", "--out-dir", "o",
            "--llm-base-url", "http://127.0.0.1:1/v1", "--pool-port", "8931"]
    a = m.build_parser().parse_args(base)
    assert a.pool_port == 8931
    for extra in (["--keep-cache"], ["--timeout", "5"], ["--config", "x.yaml"]):
        with pytest.raises(SystemExit):
            m.build_parser().parse_args(base + extra)
    p = tmp_path / "d" / "report.json"
    m.write_atomic_json(str(p), {"answer": "x"})
    assert json.loads(p.read_text()) == {"answer": "x"}
    assert [f.name for f in p.parent.iterdir()] == ["report.json"]

    class Obj:
        def model_dump(self):
            return {"k": (1, 2), 3: {4}}
    assert m.jsonable(Obj()) == {"k": [1, 2], "3": [4]}


def test_serving_proof_records_the_command_it_is_given(tmp_path, monkeypatch):
    m = load("serving_proof.py", "baseline_sp")
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    assert m.sha256_file(str(f)) == "ba7816bf8f01cfea"
    assert m.sha256_file(str(tmp_path / "missing")) is None
    cmd = "py -m vllm.entrypoints.openai.api_server --model /m --dtype bfloat16 --max-model-len 40960"
    argv, flags = m.serve_flags(cmd)
    assert argv[0] == "py" and flags == {"max_model_len": "40960", "gpu_memory_utilization": None,
                                         "dtype": "bfloat16", "max_num_seqs": None}
    md = tmp_path / "model"
    md.mkdir()
    (md / "config.json").write_text("{}")
    log = tmp_path / "vllm.log"
    log.write_text("Using FlashInfer for top-p & top-k sampling\n")
    out = tmp_path / "proof.json"
    args = ["serving_proof.py", "--out", str(out), "--model-dir", str(md), "--python", "/bin/false",
            "--vllm-log", str(log), "--serve-cmd", cmd]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(SystemExit) as e:
        m.main()
    assert e.value.code == 3 and not out.exists()
    log.write_text("Using FlashInfer for top-p & top-k sampling\nUsing FLASH_ATTN attention backend\n")
    m.main()
    rec = json.loads(out.read_text())
    assert rec["serving"]["serve_cmd"] == cmd.split() and rec["serving"]["max_num_seqs"] is None
    assert rec["serving"]["backends_from_startup_log"] == {"sampler": "FlashInfer", "attention": "FLASH_ATTN"}
    assert "deviations" not in rec
    monkeypatch.setattr(sys, "argv", args[:-2])
    with pytest.raises(SystemExit):
        m.main()


def test_campaign_refuses_without_environment(tmp_path):
    r = subprocess.run(["bash", str(BL / "campaign.slurm")], capture_output=True, text=True,
                       env=clean_env(), cwd=tmp_path, timeout=60)
    assert r.returncode == 95 and "LITREVIEW_ROOT is not set" in r.stderr
    assert list(tmp_path.iterdir()) == []


def test_campaign_runs_the_pool_service_by_path_and_pool_py_is_an_interpreter():
    s = (BL / "campaign.slurm").read_text()
    assert 'POOL_SERVICE="$LITREVIEW_ROOT/biolitbench/pool/pool_service.py"' in s
    assert '"$HOSTPY" "$POOL_SERVICE" --host' in s
    assert not re.search(r"\bPOOL_PY\b", s)
    assert "POOL_PY" in (BL / "drtulu_env.sh").read_text()
    assert 'POOL_GOLD_DIR="$POOL_GOLD_DIR"' in s and 'POOL_LOG="$OUTD/search_calls.jsonl"' in s
    assert '--gold "$POOL_GOLD_DIR" --log "$POOL_LOG"' in (BL / "drtulu_pool_inner.sh").read_text()
    assert re.search(r"^    86\) echo .*retrieval tap", s, re.M)


def test_campaign_refuses_surveyforge_under_fixed_input(tmp_path):
    env = campaign_env(tmp_path, "surveyforge", stubs(tmp_path, SF_COMMIT, gpu=True), evidence="ref",
                       ALLOW_DIR=str(tmp_path))
    r = subprocess.run(["bash", str(BL / "campaign.slurm")], capture_output=True, text=True,
                       env=env, cwd=tmp_path, timeout=60)
    assert r.returncode == 2 and "same pool only" in r.stderr
    assert not (tmp_path / "runs").exists()


def test_campaign_requires_a_gpu_for_the_embedding_arms(tmp_path):
    env = campaign_env(tmp_path, "surveyforge", stubs(tmp_path, SF_COMMIT, gpu=False),
                       PY_SURVEYFORGE=fake_python(tmp_path, "3.11"), APP_SURVEYFORGE=str(tmp_path / "app"),
                       SURVEYFORGE_DB=str(tmp_path), SURVEYFORGE_EMBED_MODEL=str(tmp_path))
    r = subprocess.run(["bash", str(BL / "campaign.slurm")], capture_output=True, text=True,
                       env=env, cwd=tmp_path, timeout=60)
    assert r.returncode == 90 and "--gres=gpu:1" in r.stderr


def test_campaign_integrity_passes_only_on_the_pinned_clean_clone(tmp_path):
    env = campaign_env(tmp_path, "surveyforge", stubs(tmp_path, SF_COMMIT, gpu=True),
                       PY_SURVEYFORGE=fake_python(tmp_path, "3.11"), APP_SURVEYFORGE=str(tmp_path / "app"),
                       SURVEYFORGE_DB=str(tmp_path), SURVEYFORGE_EMBED_MODEL=str(tmp_path))
    r = subprocess.run(["bash", str(BL / "campaign.slurm")], capture_output=True, text=True,
                       env=env, cwd=tmp_path, timeout=60)
    assert r.returncode == 95 and "integrity OK" in r.stdout, (r.stdout, r.stderr)
    rec = json.loads((tmp_path / "runs" / "campaign50" / "surveyforge" / "_arm" / "integrity.json").read_text())
    assert rec["match"] is True and rec["head"] == SF_COMMIT and rec["patch"] is None


@pytest.mark.parametrize("head,diff", [("0" * 40, b"x"), (LIRA_COMMIT, b"x"), (LIRA_COMMIT, b"")])
def test_campaign_integrity_fails_closed(tmp_path, head, diff):
    patch = tmp_path / "lira.patch"
    patch.write_bytes(diff)
    env = campaign_env(tmp_path, "lira", stubs(tmp_path, head, diff),
                       PY_LIRA=fake_python(tmp_path, "3.10"), APP_LIRA=str(tmp_path / "app"), PATCH_LIRA=str(patch))
    r = subprocess.run(["bash", str(BL / "campaign.slurm")], capture_output=True, text=True,
                       env=env, cwd=tmp_path, timeout=60)
    assert r.returncode == 91 and "does not match its pinned code" in r.stderr
    rec = json.loads((tmp_path / "runs" / "campaign50" / "lira" / "_arm" / "integrity.json").read_text())
    assert rec["match"] is False and rec["pinned_commit"] == LIRA_COMMIT
    assert rec["worktree_diff_sha256"] == hashlib.sha256(diff).hexdigest()


def engaged(tmp_path, arm, lines):
    src = (BL / "campaign.slurm").read_text()
    tap = re.search(r"^tap_engaged\(\) \{\n.*?^\}\n", src, re.S | re.M).group(0)
    block = src[src.rindex(f"  {arm})\n"):]
    fn = re.search(r"    bl_engaged\(\) \{.*?; \}\n", block, re.S).group(0)
    d = tmp_path / f"pmcid_{arm}{len(list(tmp_path.iterdir()))}"
    d.mkdir()
    (d / "stderr.log").write_text("".join(l.format(t=d.name) + "\n" for l in lines))
    r = subprocess.run(["bash", "-c", tap + fn + f'bl_engaged "{d}"'], capture_output=True, text=True)
    return r.returncode == 0


AS_LINES = ["[pool_docstore] patched 1 name(s) in src.database",
            "[pool_selector] cutoff selector applied to src.database.database",
            "[pool_selector] cutoff selector: 25049808 ids from 2023.{t}.npy"]
SF_LINES = ["[pool_docstore] patched 1 name(s) in src.database",
            "[pool_selector] cutoff selector: 25049808 ids from 2023.{t}.npy",
            "[pool_selector] cutoff selector applied to src.database.database",
            "[pool_selector] cutoff selector applied to src.database.database_survey",
            "[pool_selector] cutoff post-filter applied to src.rag.GeneralRAG_langchain.retrieve",
            "[pool_selector] survey-db selector: 17133/20000 ids before 2023"]


def test_selector_and_docstore_must_engage(tmp_path):
    assert engaged(tmp_path, "autosurvey", AS_LINES)
    assert not engaged(tmp_path, "autosurvey", AS_LINES[:2])
    assert not engaged(tmp_path, "autosurvey", AS_LINES[:2] + ["[pool_selector] cutoff selector: 9 ids from 2023.npy"])
    assert not engaged(tmp_path, "autosurvey", AS_LINES + ["[pool_docstore] disabled: boom"])
    assert engaged(tmp_path, "surveyforge", SF_LINES)
    assert not engaged(tmp_path, "surveyforge", SF_LINES[:2] + SF_LINES[3:])
    assert not engaged(tmp_path, "surveyforge", SF_LINES[:1] + SF_LINES[2:])
    assert not engaged(tmp_path, "surveyforge", SF_LINES + ["[retrieval_tap] FATAL: cannot write x"])


def test_drtulu_launcher_refuses_without_environment(tmp_path):
    r = subprocess.run(["bash", str(BL / "drtulu_pool.slurm")], capture_output=True, text=True,
                       env=clean_env(), cwd=tmp_path, timeout=60)
    assert r.returncode == 95 and "LITREVIEW_ROOT is not set" in r.stdout
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("head", ["0" * 40, DT_COMMIT])
def test_drtulu_integrity_fails_closed(tmp_path, head):
    patch = tmp_path / "dt.patch"
    patch.write_bytes(b"y")
    env = clean_env(LITREVIEW_ROOT=str(ROOT), RUNS_ROOT=str(tmp_path / "runs"), DRTULU_MODEL_DIR=str(tmp_path),
                    DRTULU_AGENT_DIR=str(tmp_path), PATCH_DRTULU=str(patch), HF_HOME=str(tmp_path),
                    SCRIBE_RUN_PY=sys.executable, SHARD="0", SLICE="0-1")
    env["PATH"] = stubs(tmp_path, head, b"y") + ":" + env["PATH"]
    r = subprocess.run(["bash", str(BL / "drtulu_pool.slurm")], capture_output=True, text=True,
                       env=env, cwd=tmp_path, timeout=60)
    assert r.returncode == 91 and "does not match its pinned code" in r.stdout
    rec = json.loads((tmp_path / "runs" / "campaign50_native" / "drtulu" / "integrity_shard0.json").read_text())
    assert rec["match"] is False and rec["head"] == head and rec["pinned_commit"] == DT_COMMIT
