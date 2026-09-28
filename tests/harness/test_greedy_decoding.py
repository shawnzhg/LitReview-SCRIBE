"""Tests greedy decoding: the model client pins temperature 0, no harness module or launcher can set
a temperature, an offline run needs no temperature argument and an empty section is not resampled."""

from __future__ import annotations

import ast
import io
import json
import re

import pytest

from harness_env import ROOT, TASK, R, W, MockClient

CODE_DIRS = [ROOT / "scribe" / d for d in ("harness/runners", "harness/tools", "levers", "retrieval", "launchers", "luna")]
SHELL = [p for d in ("launchers", "luna") for p in sorted((ROOT / "scribe" / d).rglob("*")) if p.suffix in (".sh", ".slurm", ".env")]


class RecordingClient(MockClient):

    def __init__(self, empty_writing=False):
        super().__init__()
        self.empty_writing = empty_writing
        self.log = []

    def chat(self, messages, max_tokens=2048, seed=None, stop=None):
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        kind = self.kind(first_user)
        self.log.append({"kind": kind, "seed": seed})
        if self.empty_writing and kind == "writing":
            from llm import Draw
            flat = "\n".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
            text = json.dumps({"sentences": []})
            return Draw(prompt=flat, completion=text, finish_reason="stop",
                        params={"max_tokens": max_tokens, "seed": seed, "backend": "mock"},
                        n_prompt_tokens=len(flat) // 4, n_completion_tokens=len(text) // 4, wall_ms=1,
                        model=self.model, role_sequence=[m["role"] for m in messages])
        return super().chat(messages, max_tokens=max_tokens, seed=seed, stop=stop)


def _temperature_sites(path):
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.keyword) and node.arg in ("temperature", "top_p"):
            out.append(node.value.lineno)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args.args + node.args.kwonlyargs + ([node.args.vararg] if node.args.vararg else [])
            out += [node.lineno for a in args if a and a.arg in ("temperature", "top_p")]
    return out


def test_no_harness_code_passes_or_accepts_a_temperature():
    bad = {str(p.relative_to(ROOT)): hits for d in CODE_DIRS for p in sorted(d.rglob("*.py"))
           if (hits := _temperature_sites(p))}
    assert bad == {}


def test_launchers_and_configs_expose_no_temperature():
    for p in SHELL:
        s = p.read_text()
        assert not re.search(r"\bTEMP\b|--temperature|\"temperature\"", s), p.name


def test_the_model_client_sends_temperature_zero(monkeypatch):
    import llm
    sent = []

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        sent.append(json.loads(req.data))
        return Resp(json.dumps({"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode())
    monkeypatch.setattr(llm.urllib.request, "urlopen", urlopen)
    assert llm.TEMPERATURE == 0.0 and llm.TOP_P == 1.0
    d = llm.VLLMClient(url="http://127.0.0.1:9/v1").chat([{"role": "user", "content": "x"}], max_tokens=8, seed=2)
    assert sent[0]["temperature"] == 0.0 and sent[0]["top_p"] == 1.0 and sent[0]["seed"] == 2
    assert d.params["temperature"] == 0.0 and d.error is None


def test_release_configs_default_to_seed_zero():
    for name in ("scribe_untrained.env", "scribe_trained.env", "scribe_luna.env"):
        vals = dict(l.split("=", 1) for l in (ROOT / "scribe" / "launchers" / "configs" / name).read_text().splitlines()
                    if "=" in l and not l.startswith("#"))
        assert vals["SEEDS"] == "${SEEDS:-0}" and "TEMP" not in vals, name


def test_offline_run_calls_the_client_without_a_temperature(levers_installed, task_runs):
    c = RecordingClient()
    out, mans = R.run_generation(TASK, "SCRIBE", 0, c, level=7, mode="bundle_entry", entry_mode="bundle_entry")
    assert [m["status"] for m in mans] == ["ok", "ok", "ok"]
    assert c.log and {x["kind"] for x in c.log} >= {"extract", "relations", "cross", "planning", "writing"}
    assert all(m["seed"] == 0 for m in mans)


def _writer_inputs():
    spec = {"task_id": "t_synthetic", "question": "What is known about X?", "audience": "researchers",
            "output_spec": {"target_words": 1200}}
    graph = {"claims": [{"claim_id": "c0", "text": "Finding A.", "evidence_ids": ["e1"]},
                        {"claim_id": "c1", "text": "Finding B.", "evidence_ids": ["e2"]},
                        {"claim_id": "c2", "text": "Finding C.", "evidence_ids": ["e1", "e2"]}]}
    plan = {"content_hash": "0" * 64,
            "sections": [{"section_id": "s1", "title": "One", "objective": "o1", "claim_ids": ["c0", "c2"]},
                         {"section_id": "s2", "title": "Two", "objective": "o2", "claim_ids": ["c1"]}]}
    texts = {"111": {"abstract": "Abstract one."}, "222": {"abstract": "Abstract two."}}
    bundle = {"papers": [{"paper_id": "111"}, {"paper_id": "222"}],
              "evidence": [{"evidence_id": "e1", "paper_id": "111"}, {"evidence_id": "e2", "paper_id": "222"}]}
    return spec, graph, plan, texts, bundle


def test_empty_section_is_recorded_once_and_never_resampled(levers_installed, tmp_path):
    from calllog import CallLog
    spec, graph, plan, texts, bundle = _writer_inputs()
    c = RecordingClient(empty_writing=True)
    log = CallLog(tmp_path / "trace.jsonl", "run", spec["task_id"])
    report, _ = W.writing(spec, graph, plan, texts, bundle, c, log, seed=0)
    calls = [x for x in c.log if x["kind"] == "writing"]
    assert len(calls) == 2 and all(x["seed"] == 0 for x in calls)
    assert report["terminal_audit"]["sections_empty"] == ["s1", "s2"]
