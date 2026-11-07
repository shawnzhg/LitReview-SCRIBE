"""Tests SCRIBE-Luna: the client through the recording proxy against a fake OpenAI endpoint, an
offline harness run, the driver wrapper and the launcher's refusals."""

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LUNA = ROOT / "scribe" / "luna"
RUNNERS = ROOT / "scribe" / "harness" / "runners"
sys.path.insert(0, str(ROOT / "tests" / "harness"))
sys.path.insert(0, str(LUNA))

from harness_env import TASK, MockClient, R, install_levers, install_task
import luna_client as LC
import run_luna as RL

MODEL = "gpt-5.6-luna"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", SCRIBE_RUN_PY=sys.executable)


@pytest.fixture
def levers_installed(monkeypatch):
    return install_levers(monkeypatch)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Upstream:

    def __init__(self):
        self.bodies, self.headers = [], []
        self.mock = MockClient()
        up = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                up.bodies.append(body)
                up.headers.append(dict(self.headers))
                text = up.mock.chat(body["messages"]).completion
                out = json.dumps({"id": "x", "model": body.get("model"), "service_tier": body.get("service_tier"),
                                  "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 100, "completion_tokens": 20}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


@pytest.fixture
def proxy(tmp_path):
    up = Upstream()
    key = tmp_path / "key"
    key.write_text("sk-test-not-a-real-key\n")
    key.chmod(stat.S_IRUSR | stat.S_IWUSR)
    port = free_port()
    log, costs = tmp_path / "_calls.jsonl", tmp_path / "costs.jsonl"
    p = subprocess.Popen([sys.executable, "-B", str(RUNNERS / "llm_proxy.py"), "--listen", str(port),
                          "--upstream-url", f"http://127.0.0.1:{up.port}/v1", "--api-key-file", str(key),
                          "--force-model", MODEL, "--service-tier", "flex", "--reasoning-effort", "none",
                          "--price-in", "0.1", "--price-out", "0.6", "--cost-file", str(costs), "--budget-usd", "5",
                          "--log", str(log)], env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if LC.proxy_health(f"http://127.0.0.1:{port}/v1", timeout=1):
                break
            time.sleep(0.1)
        rv = {"backend": "openai-api", "url": f"http://127.0.0.1:{port}/v1", "served_model": MODEL,
              "service_tier": "flex", "reasoning_effort": "none", "max_model_len": None, "client_timeout_s": 60}
        yield {"rv": rv, "up": up, "log": log, "costs": costs}
    finally:
        p.terminate()
        p.wait(10)
        up.srv.shutdown()


def luna_class():
    import llm
    assert Path(llm.__file__).resolve().parent == RUNNERS
    return LC.make_client_class(llm.VLLMClient)


def test_rendezvous_checks():
    good = {"backend": "openai-api", "url": "http://127.0.0.1:1/v1", "served_model": MODEL,
            "service_tier": "flex", "reasoning_effort": "none"}
    assert LC.rendezvous_problem(good) is None
    assert LC.rendezvous_problem(dict(good, backend="vllm-openai"))
    assert LC.rendezvous_problem(dict(good, url="http://127.0.0.1:1"))
    assert LC.rendezvous_problem({k: v for k, v in good.items() if k != "service_tier"})
    assert LC.rendezvous_problem(None)


def test_foreign_harness_modules_are_detected(tmp_path):
    class M:
        pass
    inside, outside = M(), M()
    inside.__file__ = str(RUNNERS / "windows.py")
    outside.__file__ = str(tmp_path / "windows.py")
    assert LC.foreign_modules({"windows": inside}) == {}
    assert LC.foreign_modules({"windows": outside, "json": outside}) == {"windows": str(tmp_path / "windows.py")}


def test_client_goes_through_the_recording_proxy(proxy):
    c = luna_class()(rendezvous=proxy["rv"])
    assert c.health()
    d = c.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "Write this section."}],
               max_tokens=64, seed=0)
    assert d.error is None and d.finish_reason == "stop" and "sentences" in d.completion
    assert d.params["backend"] == "openai-api" and d.params["service_tier"] == "flex" and d.model == MODEL
    body, head = proxy["up"].bodies[-1], proxy["up"].headers[-1]
    assert body["model"] == MODEL and body["service_tier"] == "flex" and body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 64 and "max_tokens" not in body
    assert body["temperature"] == 0.0 and body["seed"] == 0
    assert head.get("Authorization") == "Bearer sk-test-not-a-real-key"
    rec = [json.loads(l) for l in proxy["log"].read_text().splitlines()]
    assert len(rec) == 1 and rec[0]["model"] == MODEL and rec[0]["status"] == 200
    assert "sk-test" not in proxy["log"].read_text()
    assert json.loads(proxy["costs"].read_text().splitlines()[0])["cost_usd"] > 0
    desc = c.describe()
    assert desc["backend"] == "openai-api" and desc["served_model"] == MODEL and desc["max_model_len"] is None


def test_health_needs_the_declared_tier(proxy):
    c = luna_class()(rendezvous=dict(proxy["rv"], service_tier="default"))
    assert not c.health()
    with pytest.raises(SystemExit):
        luna_class()(rendezvous=dict(proxy["rv"], backend="vllm-openai"))


def test_scribe_harness_runs_on_luna(proxy, levers_installed, tmp_path, monkeypatch):
    runs = install_task(monkeypatch, tmp_path / "site")["runs"]
    entry = R.load_canonical(TASK, "evidence_bundle")
    c = luna_class()(rendezvous=proxy["rv"])
    out, mans = R.run_generation(TASK, "SCRIBE", 0, c, level=7, mode="bundle_entry", entry_bundle=entry,
                                 entry_mode="bundle_entry")
    d = R.run_dir(7, "SCRIBE", "bundle_entry", TASK, 0)
    assert d.is_relative_to(runs)
    assert [m["window"] for m in mans] == ["synthesis", "planning", "writing"]
    for m in mans:
        assert m["status"] == "ok" and m["system"] == "SCRIBE"
        assert m["serving"]["backend"] == "openai-api" and m["serving"]["served_model"] == MODEL
        assert m["prompt_hash"] == levers_installed["prompt_hash"]
    assert json.loads((d / "integrity.json").read_text())["n_findings"] == 0
    bodies = proxy["up"].bodies
    assert bodies and all(b["model"] == MODEL and b["service_tier"] == "flex" for b in bodies)
    assert all(b["temperature"] == 0.0 for b in bodies)
    assert len(proxy["log"].read_text().splitlines()) == len(bodies)
    trace = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines()]
    llm_events = [e for e in trace if e.get("kind") == "llm"]
    assert len(llm_events) == len(bodies)
    assert out["report_artifact"]["terminal_audit"]["writing_levers"]["levers"] == levers_installed["levers"]


def test_check_env_refuses_other_serving():
    good = json.dumps({"backend": "openai-api", "url": "http://127.0.0.1:1/v1", "served_model": MODEL,
                       "service_tier": "flex", "reasoning_effort": "none"})
    assert RL.check_env({"SCRIBE_RENDEZVOUS": good}) is None
    assert RL.check_env({})
    assert RL.check_env({"SCRIBE_RENDEZVOUS": "{"})
    assert RL.check_env({"SCRIBE_RENDEZVOUS": good, "SCRIBE_RENDEZVOUS_WRITING": good})
    assert set(RL.DRIVERS) == {"fixed", "acquisition", "generation"}
    assert all(p.is_file() and p.is_relative_to(ROOT) for p in RL.DRIVERS.values())


DRIVER = """import json, os, sys
{extra}
import llm
c = llm.VLLMClient(rendezvous=json.loads(os.environ["SCRIBE_RENDEZVOUS"]))
d = c.chat([{{"role": "user", "content": "Write this section."}}], max_tokens=32, seed=0)
print("RESULT " + json.dumps({{"cls": type(c).__name__, "backend": c.describe()["backend"], "err": d.error,
                              "argv": sys.argv[1:]}}))
"""


def run_wrapper(tmp_path, proxy, extra=""):
    drv = tmp_path / "driver.py"
    drv.write_text(DRIVER.format(extra=extra))
    rec = tmp_path / "modules.json"
    code = (f"import sys; from pathlib import Path; sys.path.insert(0, {str(LUNA)!r}); import run_luna as RL; "
            f"RL.DRIVERS['fixed'] = Path({str(drv)!r}); RL.main(['fixed', '--level', '7'])")
    env = dict(ENV, SCRIBE_RENDEZVOUS=json.dumps(proxy["rv"]), LUNA_MODULES_RECORD=str(rec))
    env.pop("SCRIBE_RENDEZVOUS_WRITING", None)
    env.pop("SCRIBE_RENDEZVOUS_PLANNING", None)
    r = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=120,
                       cwd=str(tmp_path))
    return r, rec


def test_wrapper_installs_the_luna_client_and_runs_the_driver(tmp_path, proxy):
    r, rec = run_wrapper(tmp_path, proxy)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout.split("RESULT ", 1)[1])
    assert res == {"cls": "LunaClient", "backend": "openai-api", "err": None, "argv": ["--level", "7"]}
    mods = json.loads(rec.read_text())["modules"]
    assert Path(mods["llm"]["file"]).parent == RUNNERS and len(mods["llm"]["md5"]) == 32
    assert proxy["up"].bodies[-1]["model"] == MODEL


def test_wrapper_refuses_harness_modules_from_elsewhere(tmp_path, proxy):
    other = tmp_path / "other"
    other.mkdir()
    (other / "windows.py").write_text("X = 1\n")
    r, _ = run_wrapper(tmp_path, proxy, extra=f"sys.path.insert(0, {str(other)!r}); import windows")
    assert r.returncode != 0 and "outside" in (r.stderr + r.stdout)
    assert not proxy["up"].bodies


def launcher_env(tmp_path, seeds="0", **kw):
    cfg = tmp_path / "luna.env"
    cfg.write_text(f"SYSTEM=SCRIBE\nSEEDS={seeds}\nLUNA_MODEL=gpt-5.6-luna\nLUNA_SERVICE_TIER=flex\n"
                   "LUNA_REASONING_EFFORT=none\n")
    (tmp_path / "tasks.txt").write_text(TASK + "\n")
    key = tmp_path / "key"
    key.write_text("sk-test-not-a-real-key\n")
    key.chmod(stat.S_IRUSR | stat.S_IWUSR)
    env = dict(ENV, LUNA_CONFIG=str(cfg), RUN_DIR=str(tmp_path / "run"), LEVEL="7",
               TASKS_FILE=str(tmp_path / "tasks.txt"), LUNA_API_KEY_FILE=str(tmp_path / "key"), LUNA_BUDGET_USD="1",
               SCRIBE_LEVERS=str(ROOT / "scribe" / "levers" / "writing_levers.json"))
    env.update(kw)
    return env


def test_launcher_syntax_and_refusals(tmp_path):
    sh = LUNA / "eval_luna.sh"
    assert subprocess.run(["bash", "-n", str(sh)]).returncode == 0
    run = lambda env, *a: subprocess.run(["bash", str(sh), *a], env=env, capture_output=True, text=True, timeout=60)
    assert run(launcher_env(tmp_path), "other").returncode == 2
    env = launcher_env(tmp_path)
    env.pop("LUNA_BUDGET_USD")
    r = run(env, "fixed")
    assert r.returncode == 2 and "LUNA_BUDGET_USD" in r.stderr
    r = run(launcher_env(tmp_path, seeds="3"), "fixed")
    assert r.returncode == 2 and "distinct seeds from 0, 1, 2" in r.stderr
    r = run(launcher_env(tmp_path, seeds="0,1"), "pool")
    assert r.returncode == 2 and "one seed per same-pool level" in r.stderr
    r = run(launcher_env(tmp_path, SCRIBE_POOL_PY=sys.executable, SCRIBE_POOL_INDEX=""), "pool")
    assert r.returncode == 2 and "SCRIBE_POOL_INDEX" in r.stderr
    r = run(launcher_env(tmp_path, EGRESS_LANE_BRIDGE=str(tmp_path / "none.py")), "fixed")
    assert r.returncode == 2 and "EGRESS_LANE_BRIDGE" in r.stderr
    assert not (tmp_path / "run").exists()


def run_stage(tmp_path, stage, *args, **env_extra):
    rv = {"backend": "openai-api", "url": "http://127.0.0.1:9/v1", "served_model": MODEL, "service_tier": "flex",
          "reasoning_effort": "none", "max_model_len": None}
    env = dict(ENV, SCRIBE_RENDEZVOUS=json.dumps(rv), SCRIBE_LEVERS=str(ROOT / "scribe" / "levers" / "writing_levers.json"),
               SCRIBE_DRIVER_RECORD=str(tmp_path / "driver.json"), LUNA_MODULES_RECORD=str(tmp_path / "modules.json"),
               **env_extra)
    for v in ("SCRIBE_RENDEZVOUS_WRITING", "SCRIBE_RENDEZVOUS_PLANNING"):
        env.pop(v, None)
    return subprocess.run([sys.executable, "-B", str(LUNA / "run_luna.py"), stage, *args], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120)


def test_fixed_stage_runs_the_repo_driver_with_the_lever_writer(tmp_path):
    r = run_stage(tmp_path, "fixed", "--phase", "generation", "--systems", "SCRIBE", "--mode", "bundle_entry",
                  "--level", "7", "--tasks-file", os.devnull)
    assert r.returncode != 0 and "no task ids" in (r.stderr + r.stdout)
    rec = json.loads((tmp_path / "driver.json").read_text())
    assert Path(rec["windows_file"]).resolve().parent == RUNNERS
    assert rec["levers"]["installed"] == ["evidence", "length", "paragraphs"]
    assert Path(rec["levers"]["module"]).resolve().parent == ROOT / "scribe" / "levers"
    mods = json.loads((tmp_path / "modules.json").read_text())["modules"]
    assert {"runner", "windows", "llm", "writing_levers"} <= set(mods)
    assert all(ROOT in Path(m["file"]).parents for m in mods.values())


def test_same_pool_stages_fail_closed_without_the_pool_setup(tmp_path):
    for stage, args in (("generation", ("--phase", "generation", "--systems", "SCRIBE", "--level", "7")),
                        ("acquisition", ("--phase", "acquisition", "--systems", "SCRIBE"))):
        r = run_stage(tmp_path, stage, *args)
        assert r.returncode != 0, (stage, r.stdout[-500:], r.stderr[-500:])
    assert {p.name for p in tmp_path.iterdir()} <= {"driver.json", "modules.json"}


LANE_CLIENT = """import json, os, sys
sys.path.insert(0, {luna!r})
import luna_client as LC
llm = LC.load_llm()
c = LC.make_client_class(llm.VLLMClient)(rendezvous=json.loads(os.environ["SCRIBE_RENDEZVOUS"]))
d = c.chat([{{"role": "user", "content": "Write this section."}}], max_tokens=16, seed=0)
print("LANE " + json.dumps({{"health": c.health(), "err": d.error, "ok": "sentences" in d.completion}}))
"""


def test_luna_calls_reach_the_proxy_only_through_the_guard_lane(tmp_path, proxy):
    if subprocess.run(["unshare", "-rn", "true"], capture_output=True).returncode != 0:
        pytest.skip("user network namespaces are not available")
    port = int(proxy["rv"]["url"].rsplit(":", 1)[1].split("/")[0])
    lane_dir = Path(tempfile.mkdtemp(prefix="lane_", dir="/tmp"))
    sock = lane_dir / "lane.sock"
    bridge = subprocess.Popen([sys.executable, "-B", str(ROOT / "scribe" / "harness" / "runners" / "sock_bridge.py"),
                               "--unix-listen", str(sock), "--tcp-connect", f"127.0.0.1:{port}"], env=ENV,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if sock.exists():
                break
            time.sleep(0.1)
        app = free_port()
        client = tmp_path / "client.py"
        client.write_text(LANE_CLIENT.format(luna=str(LUNA)))
        env = dict(ENV, PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
                   SCRIBE_RENDEZVOUS=json.dumps(dict(proxy["rv"], url=f"http://127.0.0.1:{app}/v1")),
                   EGRESS_LANE_SOCK=str(sock), EGRESS_LANE_PORT=str(app),
                   EGRESS_LANE_BRIDGE=str(ROOT / "scribe" / "harness" / "runners" / "sock_bridge.py"),
                   EGRESS_LANE_EXPECT_MODEL=MODEL, EGRESS_LANE_EXPECT_TIER="flex", EGRESS_DECLARED_PORTS=str(app))
        r = subprocess.run(["bash", str(RUNNERS / "egress_guard.sh"), str(tmp_path / "proof.json"), sys.executable, "-B",
                            str(client)], env=env, capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, (r.stdout[-1500:], r.stderr[-1500:])
        res = json.loads(r.stdout.split("LANE ", 1)[1].splitlines()[0])
        assert res == {"health": True, "err": None, "ok": True}
        proof = json.loads((tmp_path / "proof.json").read_text())
        assert proof["external_all_blocked"] is True and proof["sanctioned_lanes"][0]["kind"] == "unix-socket"
        assert proxy["up"].bodies[-1]["model"] == MODEL
        direct = subprocess.run(["unshare", "-rn", sys.executable, "-B", "-c",
                                 f"import urllib.request; urllib.request.urlopen('http://127.0.0.1:{port}/__proxy_health', timeout=3)"],
                                capture_output=True, text=True, timeout=60)
        assert direct.returncode != 0
    finally:
        bridge.terminate()
        bridge.wait(10)
        shutil.rmtree(lane_dir, ignore_errors=True)
