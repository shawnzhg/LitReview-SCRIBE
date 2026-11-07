"""Tests the retrieval check at the start of synthesis: a loaded retrieval module or a reachable
pool URL fails the run before any model call."""

from __future__ import annotations

import http.server
import json
import os
import socket
import sys
import threading
import types

import pytest

from harness_env import R, TASK, MockClient


def _run(task_runs, client, **k):
    out, mans = R.run_generation(TASK, "SCRIBE", 0, client, level=7, mode="bundle_entry", entry_mode="bundle_entry", **k)
    d = R.run_dir(7, "SCRIBE", "bundle_entry", TASK, 0)
    return out, mans, d


def _assert_failed_closed(out, mans, d, client, needle):
    assert out == {}
    assert len(mans) == 1 and mans[0]["window"] == "synthesis" and mans[0]["status"] == "failed"
    assert mans[0]["failure_reason"] == "harness_failure:retrieval_reachable"
    assert any(needle in f for f in mans[0]["retrieval_check"]["findings"]), mans[0]["retrieval_check"]
    assert client.calls == {}, "no model call may happen"
    on_disk = [json.loads(l) for l in (d / "manifests.jsonl").read_text().splitlines()]
    assert on_disk == mans
    assert not (d / "synthesis_graph.json").exists()
    tr = [json.loads(l) for l in (d / "trace.jsonl").read_text().splitlines()]
    assert any(e.get("kind") == "failure" and "RetrievalPathPresent" in json.dumps(e) for e in tr)


@pytest.mark.parametrize("mod", ["pool_retrieval_tool", "metered_tool"])
def test_loaded_retrieval_module_fails_the_run(mod, levers_installed, task_runs, monkeypatch):
    assert mod not in sys.modules
    monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    c = MockClient()
    out, mans, d = _run(task_runs, c)
    _assert_failed_closed(out, mans, d, c, mod)


def test_pool_backend_acquisition_half_fails_the_run_generation_half_does_not(levers_installed, task_runs):
    import pool_backend as PB
    saved = dict(PB._INSTALLED)
    try:
        PB._INSTALLED.clear()
        PB._INSTALLED["gen"] = True
        assert R.assert_no_retrieval_path() == []
        PB._INSTALLED["acq"] = object()
        c = MockClient()
        out, mans, d = _run(task_runs, c)
        _assert_failed_closed(out, mans, d, c, "pool_backend")
    finally:
        PB._INSTALLED.clear()
        PB._INSTALLED.update(saved)


class _Healthz(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"ok": true}'
        self.send_response(200 if self.path == "/healthz" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_reachable_pool_url_fails_the_run(levers_installed, task_runs, monkeypatch):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Healthz)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        monkeypatch.setenv("POOL_URL", f"http://127.0.0.1:{srv.server_address[1]}")
        c = MockClient()
        out, mans, d = _run(task_runs, c)
        _assert_failed_closed(out, mans, d, c, "POOL_URL")
        assert mans[0]["retrieval_check"]["pool_url_probe"]["http_status"] == 200
    finally:
        srv.shutdown()
        srv.server_close()


def test_unreachable_pool_url_is_recorded_and_the_run_proceeds(levers_installed, task_runs, monkeypatch):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    monkeypatch.setenv("POOL_URL", f"http://127.0.0.1:{port}")
    c = MockClient()
    out, mans, d = _run(task_runs, c)
    assert [m["status"] for m in mans] == ["ok", "ok", "ok"]
    probe = mans[0]["retrieval_check"]["pool_url_probe"]
    assert probe["configured"] and probe["reachable"] is False
    assert mans[0]["retrieval_check"]["findings"] == []


def test_no_pool_url_no_probe():
    assert "POOL_URL" not in os.environ
    reach, rec = R.probe_pool_url()
    assert reach is False and rec == {"configured": False}
