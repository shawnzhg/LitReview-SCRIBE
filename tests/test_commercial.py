"""CPU tests of the commercial-agent drivers on synthetic fixtures: prompt, MCP servers, runners,
audits and record extraction."""

import ast
import importlib.util
import io
import json
import runpy
import sqlite3
import sys
import tokenize
import types
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
COMM = ROOT / "scribe" / "harness" / "commercial"
TASK = "pmcid_PMC1"
REVIEW = "999"
SEARCH_DESC = "Search the biomedical literature corpus. Returns a ranked list of papers (id = PMID)."
FETCH_DESC = "Fetch one paper by id (PMID): title, year and abstract."
SESSION_NOTE = "`session` is the session code given in your task."
SPEC = {"question": "Gene regulation in yeast", "audience": "biomedical researchers",
        "output_spec": {"target_words": 1234}, "content_hash": "sha256:abc"}


class Resp:
    def __init__(self, status, data):
        self.status_code, self._data = status, data

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def load(name, monkeypatch):
    monkeypatch.syspath_prepend(str(COMM))
    for p in COMM.glob("*.py"):
        monkeypatch.delitem(sys.modules, p.stem, raising=False)
    spec = importlib.util.spec_from_file_location(f"commercial_{name}", COMM / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def env(tmp_path, monkeypatch):
    for d in ("taskspecs", "gold", "allowlists", "out"):
        (tmp_path / d).mkdir()
    (tmp_path / "tasks.json").write_text(json.dumps([TASK]))
    (tmp_path / "taskspecs" / f"{TASK}.json").write_text(json.dumps(SPEC))
    (tmp_path / "gold" / f"{TASK}.json").write_text(json.dumps({"review_pmid": int(REVIEW)}))
    (tmp_path / "allowlists" / f"{TASK}.json").write_text(json.dumps(["1", "2"]))
    for k, d in (("COMMERCIAL_OUT", "out"), ("COMMERCIAL_TASKSPECS", "taskspecs"), ("COMMERCIAL_GOLD", "gold"),
                 ("COMMERCIAL_ALLOWLISTS", "allowlists"), ("COMMERCIAL_TASKS", "tasks.json")):
        monkeypatch.setenv(k, str(tmp_path / d))
    return tmp_path


def stub_mcp(monkeypatch, headers):
    tools = {}

    class FastMCP:
        def __init__(self, name, instructions):
            self.name = name

        def tool(self, description=None):
            def deco(fn):
                tools[fn.__name__] = description
                return fn
            return deco

    mods = {"fastmcp": {"FastMCP": FastMCP}, "fastmcp.server": {},
            "fastmcp.server.dependencies": {"get_http_headers": lambda include_all=False: headers},
            "starlette": {}, "starlette.middleware": {"Middleware": object},
            "starlette.middleware.base": {"BaseHTTPMiddleware": object},
            "starlette.responses": {"JSONResponse": object}}
    for name, attrs in mods.items():
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
    return tools


def fake_pool(calls, pmids=(REVIEW, "1", "2", "3")):
    def get(url, params=None, timeout=None):
        calls.append((url, params))
        if url.endswith("/search"):
            return Resp(200, [{"pmid": p, "title": "T" + p, "abstract": "x" * 400 if p == "1" else ""} for p in pmids])
        if url.endswith("PMID:404"):
            return Resp(404, {})
        return Resp(200, {"title": "T", "year": 2000, "abstract": "abs"})
    return get


def test_code_files_have_one_header_no_comments_no_site_paths():
    files = sorted(COMM.glob("*.py"))
    assert len(files) == 16
    for p in files:
        src = p.read_text()
        tree = ast.parse(src)
        assert ast.get_docstring(tree), p.name
        docs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)
                and isinstance(n.body[0].value.value, str)]
        assert not docs, p.name
        comments = [t for t in tokenize.generate_tokens(io.StringIO(src).readline)
                    if t.type == tokenize.COMMENT and not (t.start[0] == 1 and t.string.startswith("#!"))]
        assert not comments, p.name
        for bad in [f"/{d}/" for d in ("nfs", "scratch", "sw", "home")] + [".openai_key", ".elicit"]:
            assert bad not in src, (p.name, bad)


def test_prompt_is_built_from_the_taskspec_only(monkeypatch):
    P = load("agent_prompt", monkeypatch)
    text = P.prompt(SPEC)
    assert text.startswith('Write a scientific literature review on the topic: "Gene regulation in yeast".\n\n')
    assert "Target length: about 1234 words of body text." in text
    assert "they are your only source of papers" in text
    assert text.endswith("- End with a 'References' section listing every cited paper as: [n] Title. PMID: <pmid>. <url>\n")


def test_mcp_pool_multi_binds_the_task_and_suppresses_the_review(env, monkeypatch):
    headers = {"x-task": TASK}
    tools = stub_mcp(monkeypatch, headers)
    log = env / "mcp.jsonl"
    for k, v in (("MCP_TASKS", env / "tasks.json"), ("MCP_GOLD_DIR", env / "gold"), ("MCP_POOL_URL", "http://pool/"),
                 ("MCP_TOKEN", "t" * 32), ("MCP_LOG", log), ("MCP_K", "2")):
        monkeypatch.setenv(k, str(v))
    monkeypatch.delenv("MCP_ALLOWLIST_DIR", raising=False)
    m = load("mcp_pool_multi", monkeypatch)
    assert tools == {"search": SEARCH_DESC, "fetch": FETCH_DESC}
    calls = []
    monkeypatch.setattr(m.requests, "get", fake_pool(calls))
    task, excl = m.task_ctx()
    assert (task, excl) == (TASK, {REVIEW})
    out = m._search("yeast", task, excl)
    assert [r["id"] for r in out["results"]] == ["1", "2"]
    assert len(out["results"][0]["text"]) == 300
    assert calls[0] == (f"http://pool/t/{TASK}/search", {"q": "yeast", "k": 3})
    with pytest.raises(ValueError):
        m._fetch(f"PMID:{REVIEW}", task, excl)
    with pytest.raises(ValueError):
        m._fetch("404", task, excl)
    got = m._fetch("PMID:5", task, excl)
    assert got["text"] == "T\n(2000)\n\nabs" and got["url"] == "https://pubmed.ncbi.nlm.nih.gov/5/"
    assert calls[-1][0] == f"http://pool/t/{TASK}/graph/v1/paper/PMID:5"
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert recs[0]["returned"] == ["1", "2"] and recs[0]["suppressed"] == [REVIEW]
    assert recs[1]["suppressed"] == [REVIEW] and recs[1]["returned"] == []
    assert not any(c[0].endswith(f"PMID:{REVIEW}") for c in calls)
    headers["x-task"] = "pmcid_OTHER"
    with pytest.raises(ValueError):
        m.task_ctx()


def test_mcp_pool_multi_under_fixed_input_refuses_papers_outside_the_reference_list(env, monkeypatch):
    stub_mcp(monkeypatch, {"x-task": TASK})
    log = env / "mcp.jsonl"
    for k, v in (("MCP_TASKS", env / "tasks.json"), ("MCP_GOLD_DIR", env / "gold"), ("MCP_POOL_URL", "http://ref"),
                 ("MCP_ALLOWLIST_DIR", env / "allowlists"), ("MCP_TOKEN", "t" * 32), ("MCP_LOG", log)):
        monkeypatch.setenv(k, str(v))
    m = load("mcp_pool_multi", monkeypatch)
    calls = []
    monkeypatch.setattr(m.requests, "get", fake_pool(calls, pmids=("1", "2")))
    assert [r["id"] for r in m._search("q", TASK, {REVIEW})["results"]] == ["1", "2"]
    monkeypatch.setattr(m.requests, "get", fake_pool(calls))
    with pytest.raises(ValueError):
        m._search("q", TASK, {REVIEW})
    with pytest.raises(ValueError):
        m._fetch("5", TASK, {REVIEW})
    recs = [json.loads(l) for l in log.read_text().splitlines()]
    assert recs[-1]["tool"] == "_allowlist_violation" and recs[-1]["pmids"] == ["5"]
    assert recs[-2]["tool"] == "_allowlist_violation" and recs[-2]["pmids"] == [REVIEW, "3"]


def test_mcp_pool_session_resolves_task_and_condition_from_the_code(env, monkeypatch):
    tools = stub_mcp(monkeypatch, {})
    (env / "sessions.json").write_text(json.dumps({"cs-abc": {"task": TASK, "cond": "fixinput"},
                                                   "cs-def": {"task": TASK, "cond": "samepool"}}))
    for k, v in (("CS_SESSIONS", env / "sessions.json"), ("MCP_GOLD_DIR", env / "gold"),
                 ("CS_POOL_URL_FIXINPUT", "http://ref/"), ("CS_POOL_URL_SAMEPOOL", "http://full"),
                 ("MCP_ALLOWLIST_DIR", env / "allowlists"),
                 ("MCP_TOKEN", "t" * 32), ("MCP_LOG", env / "cs.jsonl"), ("MCP_K", "20")):
        monkeypatch.setenv(k, str(v))
    m = load("mcp_pool_session", monkeypatch)
    assert tools == {"search": SEARCH_DESC + "\n" + SESSION_NOTE, "fetch": FETCH_DESC + " " + SESSION_NOTE}
    calls = []
    monkeypatch.setattr(m.requests, "get", fake_pool(calls, pmids=("1", "2")))
    with pytest.raises(ValueError):
        m.ctx("cs-nope")
    task, cond, excl = m.ctx("cs-abc")
    out = m._search("q", task, cond, excl)
    assert calls[-1][0] == f"http://ref/t/{TASK}/search" and [r["id"] for r in out["results"]] == ["1", "2"]
    monkeypatch.setattr(m.requests, "get", fake_pool(calls))
    with pytest.raises(ValueError):
        m._search("q", task, cond, excl)
    task, cond, excl = m.ctx("cs-def")
    out = m._search("q", task, cond, excl)
    assert calls[-1][0] == f"http://full/t/{TASK}/search" and REVIEW not in [r["id"] for r in out["results"]]
    with pytest.raises(ValueError):
        m._fetch(REVIEW, task, cond, excl)
    recs = [json.loads(l) for l in (env / "cs.jsonl").read_text().splitlines()]
    assert recs[0]["tool"] == "_bad_session" and recs[1]["cond"] == "fixinput"
    assert recs[2]["tool"] == "_allowlist_violation" and recs[3]["cond"] == "samepool"


def fake_response(status, error=None, text="# Intro\nBody [1].\n"):
    return types.SimpleNamespace(id="resp_1", status=status, output_text=text, model_dump=lambda: {
        "id": "resp_1", "status": status, "output": [{"type": "mcp_call", "error": error}],
        "usage": {"input_tokens": 1000, "output_tokens": 100, "input_tokens_details": {"cached_tokens": 200}}})


def openai_env(env, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-key")
    for c in ("FIXINPUT", "SAMEPOOL"):
        (env / f"tok_{c}").write_text("s" * 40)
        monkeypatch.setenv(f"MCP_URL_{c}", "https://mcp.example/")
        monkeypatch.setenv(f"MCP_TOKEN_FILE_{c}", str(env / f"tok_{c}"))
    return load("run_openai_tool_loop", monkeypatch)


def test_openai_tool_loop_records_an_accepted_run(env, monkeypatch):
    m = openai_env(env, monkeypatch)
    seen = {}

    class Responses:
        def create(self, **kw):
            seen.update(kw)
            return fake_response("completed")

        def retrieve(self, rid):
            return fake_response("completed")
    monkeypatch.setattr(m, "client", types.SimpleNamespace(responses=Responses()))
    m.one(TASK, "fixinput")
    d = env / "out" / "runs" / "fixinput" / TASK
    assert (d / "report.md").read_text() == "# Intro\nBody [1].\n" and (d / "response.json").exists()
    assert seen["model"] == "gpt-5.6-luna" and seen["reasoning"] == {"effort": "none"} and seen["max_tool_calls"] == 80
    tool = seen["tools"][0]
    assert tool["type"] == "mcp" and tool["headers"]["X-Task"] == TASK and tool["server_url"].startswith("https://mcp.example/")
    req = (d / "request.json").read_text()
    assert "s" * 40 not in req and json.loads(req)["tools"][0]["headers"] == {"X-Task": TASK, "Authorization": "<redacted>"}
    rec = json.loads((env / "out" / "openai_costs.jsonl").read_text())
    assert rec["cost_usd"] == round(800 * 0.20e-6 + 200 * 0.02e-6 + 100 * 1.20e-6, 6) and rec["accepted"]


def test_openai_tool_loop_rejects_a_run_with_a_tool_error_and_never_reruns_it(env, monkeypatch):
    m = openai_env(env, monkeypatch)
    n = []

    def create(**kw):
        n.append(1)
        return fake_response("completed", error="timeout")
    resp = types.SimpleNamespace(create=create, retrieve=lambda rid: fake_response("completed", error="timeout"))
    monkeypatch.setattr(m, "client", types.SimpleNamespace(responses=resp))
    m.one(TASK, "samepool")
    d = env / "out" / "runs" / "samepool" / TASK
    assert (d / "response.json").exists() and (d / "report.md").exists()
    rec = json.loads((env / "out" / "openai_costs.jsonl").read_text())
    assert rec["accepted"] is False and rec["n_mcp_errors"] == 1
    m.one(TASK, "samepool")
    assert len(n) == 1


def test_openai_tool_loop_runs_the_task_list_and_writes_nothing_else(env, monkeypatch):
    (env / "tasks.json").write_text(json.dumps(["pmcid_PMC2", TASK]))
    m = openai_env(env, monkeypatch)
    seen = []
    monkeypatch.setattr(m, "one", lambda t, c: seen.append((t, c)))
    m.main(1)
    assert sorted(seen) == sorted((t, c) for t in ("pmcid_PMC2", TASK) for c in ("fixinput", "samepool"))
    assert list((env / "out").iterdir()) == []


def claude_env(env, monkeypatch, model="claude-sonnet-5"):
    for c in ("FIXINPUT", "SAMEPOOL"):
        (env / f"tok_{c}").write_text("c" * 40)
        monkeypatch.setenv(f"MCP_TOKEN_FILE_{c}", str(env / f"tok_{c}"))
    m = load("run_claude_code", monkeypatch)
    cap = {"n": 0}

    def fake_run(cmd, cwd, stdout, stderr, env):
        cap.update(cmd=cmd, cwd=cwd, empty=not any(Path(cwd).iterdir()), n=cap["n"] + 1)
        msgs = [{"type": "system", "subtype": "init", "model": model, "tools": ["mcp__pool__fetch", "mcp__pool__search"]},
                {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__pool__search"}]}},
                {"type": "result", "subtype": "success", "result": "# R\nBody [1].", "num_turns": 3, "usage": {}, "total_cost_usd": 0.1}]
        stdout.write("\n".join(json.dumps(x) for x in msgs) + "\n")
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(m.subprocess, "run", fake_run)
    return m, cap


def test_claude_code_runs_with_only_the_pool_tools(env, monkeypatch):
    m, cap = claude_env(env, monkeypatch)
    assert m.run(TASK, "fixinput") == "success"
    cmd = cap["cmd"]
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5"
    assert cmd[cmd.index("--tools") + 1] == "" and "--strict-mcp-config" in cmd and cap["empty"]
    i = cmd.index("--allowedTools")
    assert cmd[i + 1:i + 3] == ["mcp__pool__search", "mcp__pool__fetch"] and cmd[i + 3].startswith("--")
    assert cmd[cmd.index("--effort") + 1] == "low" and cmd[cmd.index("--setting-sources") + 1] == "project"
    cfg = json.loads(Path(cmd[cmd.index("--mcp-config") + 1]).read_text())
    assert cfg["mcpServers"]["pool"]["headers"]["X-Task"] == TASK and cfg["mcpServers"]["pool"]["url"].startswith("http://127.0.0.1:31971/")
    d = env / "out" / "runs" / "fixinput" / TASK
    assert "c" * 40 not in (d / "request.json").read_text()
    res = json.loads((d / "result.json").read_text())
    assert res["tools_called"] == ["mcp__pool__search"] and res["model"] == "claude-sonnet-5" and res["words"] == 4
    assert res["model_ok"] and res["n_search"] == 1
    assert (env / "out" / "claude_costs.jsonl").exists()
    B = load("run_claude_code_batch", monkeypatch)
    assert B.ok(d) and B.OUT == env / "out" / "runs"
    assert m.run(TASK, "fixinput") == "skip" and cap["n"] == 1


def test_claude_code_fails_a_run_on_another_model(env, monkeypatch):
    m, cap = claude_env(env, monkeypatch, model="claude-sonnet-4")
    assert m.run(TASK, "samepool") == "model_mismatch"
    d = env / "out" / "runs" / "samepool" / TASK
    res = json.loads((d / "result.json").read_text())
    assert res["model_ok"] is False and res["requested_model"] == "claude-sonnet-5"
    B = load("run_claude_code_batch", monkeypatch)
    assert not B.ok(d)
    seen = []
    monkeypatch.setattr(B.R, "run", lambda t, c: seen.append((t, c)))
    B.one(TASK, "samepool")
    assert seen == []


def write_claude_run(env, cond, cited, returned):
    d = env / "out" / "runs" / cond / TASK
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"subtype": "success", "n_tool_errors": 0, "words": 5, "n_tool_calls": 2,
                                               "tools_called": ["mcp__pool__search"], "total_cost_usd": 0.1, "wall_s": 9}))
    use = {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "mcp__pool__search"}]}}
    msg = {"type": "user", "message": {"content": [{"type": "tool_result",
                                                    "content": json.dumps({"results": [{"id": p} for p in returned]})}]}}
    (d / "stream.jsonl").write_text(json.dumps(use) + "\n" + json.dumps(msg) + "\n")
    refs = "\n".join(f"[{i + 1}] Paper. PMID: {p}. https://pubmed.ncbi.nlm.nih.gov/{p}/" for i, p in enumerate(cited))
    (d / "report.md").write_text("# Intro\nBody [1] [2].\n\n## References\n" + refs + "\n")


def test_audit_claude_code_flags_citations_outside_the_retrieved_set(env, monkeypatch, capsys):
    write_claude_run(env, "fixinput", ["10001", "30003"], ["10001", "20002"])
    write_claude_run(env, "samepool", ["10001", "99999"], ["10001", "99999"])
    (env / "allowlists" / f"{TASK}.json").write_text(json.dumps(["10001", "20002"]))
    (env / "gold" / f"{TASK}.json").write_text(json.dumps({"review_pmid": 99999}))
    runpy.run_path(str(COMM / "audit_claude_code.py"), run_name="__main__")
    rows = {r["cond"]: r for r in json.loads((env / "out" / "audit.json").read_text())}
    assert rows["fixinput"]["outside_P"] == 1 and rows["fixinput"]["outside_allow"] == 1
    assert rows["fixinput"]["n_search"] == 1
    assert not rows["fixinput"]["review_cited"] and rows["samepool"]["review_cited"]
    assert "runs citing PMIDs outside own P: 1" in capsys.readouterr().out


def write_elicit_run(env, n_included):
    d = env / "out" / "runs" / TASK
    (d / "exports").mkdir(parents=True)
    (d / "final.json").write_text(json.dumps({"status": "completed", "_t": "2000-01-01T00:00:00+0000"}))
    (d / "request.json").write_text(json.dumps({"extraction": {"generate": True}}))
    (d / "exports" / "search.csv.final").write_text("Paper ID,Title\nPUBMED-1,A\nPUBMED-2,B\n")
    rows = "".join("PUBMED-1,Yes\n" for _ in range(n_included))
    (d / "exports" / "extract.csv.postreport").write_text("Paper ID,Included in report\n" + rows + "PUBMED-2,No\n")
    (d / "exports" / "report.txt.final").write_text("x")
    (d / "report_body.md").write_text("Result {ab12_1} and {cd34_2}.")


@pytest.mark.parametrize("n_included,over", [(80, []), (81, [TASK[-8:]])])
def test_audit_elicit_checks_entry_and_the_80_paper_cap(env, monkeypatch, capsys, n_included, over):
    write_elicit_run(env, n_included)
    runpy.run_path(str(COMM / "audit_elicit.py"), run_name="__main__")
    row = json.loads((env / "out" / "audit.json").read_text())[0]
    assert row["gathered_in_allow"] == 2 and row["allow_not_gathered"] == 0 and row["gathered_not_in_allow"] == 0
    assert row["included_in_report"] == n_included and row["included_not_gathered"] == 0
    assert row["cite_markers"] == 2 and "account" not in row and "replaces" not in row
    assert f"over 80: {over}" in capsys.readouterr().out
    assert sorted(p.name for p in (env / "out").iterdir()) == ["audit.json", "runs"]


def test_elicit_request_gathers_exactly_the_reference_list_without_screening(env, monkeypatch):
    monkeypatch.setenv("ELICIT_API_KEY", "el-test-not-a-key")
    allow = [str(10000 + i) for i in range(250)]
    (env / "allowlists" / f"{TASK}.json").write_text(json.dumps(allow))
    E = load("run_elicit_sr", monkeypatch)
    assert E.hdr() == {"Authorization": "Bearer el-test-not-a-key", "Content-Type": "application/json"}

    def stop(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(E.requests, "post", stop)
    with pytest.raises(KeyboardInterrupt):
        E.main(TASK)
    assert sorted(p.name for p in (env / "out").iterdir()) == ["runs"]
    req = json.loads((env / "out" / "runs" / TASK / "request.json").read_text())
    assert "abstractScreening" not in req and "fulltextScreening" not in req
    assert req["extraction"] == {"generate": True, "useFigures": False} and req["generateReport"] is True
    got = [q.split("[pmid]")[0].strip() for s in req["searches"] for q in s["query"].split(" OR ")]
    assert got == allow and len(req["searches"]) == 3 and all(len(s["query"]) <= 2000 for s in req["searches"])
    assert "el-test-not-a-key" not in json.dumps(req)


def test_audit_gemini_resolves_cited_pmids_in_the_pool(env, monkeypatch, capsys):
    monkeypatch.setenv("COMMERCIAL_POOL_URL", "http://full/")
    (env / "allowlists" / f"{TASK}.json").write_text(json.dumps(["10001", "20002"]))
    (env / "gold" / f"{TASK}.json").write_text(json.dumps({"review_pmid": 99999}))
    md = ("# Title\nBody text [1] [2].\n\n## References\n[1] Alpha beta gamma delta. https://pubmed.ncbi.nlm.nih.gov/10001/\n"
          "[2] Wrong words entirely here. PMID: 20002\n[3] The target review. PMID: 99999\n")
    for cond in ("samepool", "fixinput"):
        d = env / "out" / "runs" / cond
        d.mkdir(parents=True)
        (d / f"01_{TASK}.md").write_text(md)
        (d / f"01_{TASK}.sources.json").write_text(json.dumps({"sources_used": [{"url": "https://pubmed.ncbi.nlm.nih.gov/10001/", "domain": "pubmed"}]}))
    titles = {"10001": "Alpha beta gamma delta", "20002": "Completely unrelated study of mice"}

    def get(url, params=None, timeout=None):
        pmid = url.rsplit(":", 1)[1]
        return Resp(200, {"title": titles[pmid], "year": 2000}) if pmid in titles else Resp(404, {})
    import requests
    monkeypatch.setattr(requests, "get", get)
    runpy.run_path(str(COMM / "audit_gemini.py"), run_name="__main__")
    rows = {r["cond"]: r for r in json.loads((env / "out" / "audit.json").read_text())}
    r = rows["fixinput"]
    assert (r["n_pmid"], r["pmid_in_pool"], r["pmid_title_match"], r["pmid_title_mismatch"]) == (3, 2, 1, 1)
    assert r["pmid_not_in_pool_before_cutoff"] == 1 and r["in_allowlist"] == 2 and r["target_review_cited"]
    assert sorted(p.name for p in (env / "out").iterdir()) == ["audit.json", "runs"]


def make_cs_db(data, fid, link):
    org = data / "orgs" / "o1"
    (org / "workspaces" / fid).mkdir(parents=True)
    (org / "artifacts").mkdir()
    (org / "artifacts" / "v1.md").write_text("# Linked report\n\n## References\n[1] A. PMID: 1.\n")
    (org / "workspaces" / fid / "draft.md").write_text("# Draft\n")
    con = sqlite3.connect(org / "operon-cli.db")
    con.execute("create table frames (id, root_frame_id, agent_name, model, effort, input_tokens, output_tokens, total_cost, status)")
    con.execute("create table frame_messages (frame_id, idx, msg_json)")
    con.execute("create table artifact_versions (id, storage_path)")
    con.execute("create table artifacts (id, latest_version_id)")
    con.execute("insert into frames values (?,?,?,?,?,?,?,?,?)", (fid, fid, "MAIN", "claude-sonnet-5", "low", 10, 20, 0.5, "completed"))
    con.execute("insert into frames values (?,?,?,?,?,?,?,?,?)", ("kid", fid, "REVIEWER", "claude-sonnet-5", "low", 1, 2, 0.1, "completed"))
    aid = "0" * 8 + "-0000-0000-0000-" + "0" * 12
    con.execute("insert into artifact_versions values (?,?)", (aid, "v1.md"))
    long_msg = "## Review\n" + "word " * 900 + "\n## References\n[1] A. PMID: 1."
    msgs = [{"role": "user", "content": "prompt"},
            {"role": "assistant", "content": [{"type": "text", "text": long_msg}]},
            {"role": "assistant", "content": f"Done: {{{{artifact:{aid}}}}}" if link else "Done."}]
    for i, m in enumerate(msgs):
        con.execute("insert into frame_messages values (?,?,?)", (fid, i, json.dumps(m)))
    con.commit()
    con.close()


@pytest.mark.parametrize("link,source", [(True, "linked artifact"), (False, "message idx 1")])
def test_cs_extract_selects_the_final_delivery(env, monkeypatch, link, source):
    data = env / "csd"
    fid = "frm_1"
    make_cs_db(data, fid, link)
    monkeypatch.setenv("CS_DATA", str(data))
    X = load("cs_extract", monkeypatch)
    rec = X.extract("fixinput", f"01_{TASK}", f"http://localhost:38765/projects/p/frames/{fid}")
    d = env / "out" / "runs" / "fixinput" / f"01_{TASK}"
    assert rec["report_source"].startswith(source) and rec["children"] == [("REVIEWER", "claude-sonnet-5")]
    assert json.loads((d / "frame.json").read_text())["model"] == "claude-sonnet-5"
    assert len((d / "messages.jsonl").read_text().splitlines()) == 3
    assert (d / f"01_{TASK}.artifact.draft.md").exists()
    rep = (d / "report.md").read_text()
    assert rep.startswith("# Linked report") if link else rep.startswith("## Review")


def test_audit_claude_science_uses_the_run_window_of_the_mcp_log(env, monkeypatch, capsys):
    (env / "allowlists" / f"{TASK}.json").write_text(json.dumps(["10001", "20002"]))
    logs = []
    for cond, t0 in (("fixinput", 1000.0), ("samepool", 5000.0)):
        d = env / "out" / "runs" / cond / f"01_{TASK}"
        d.mkdir(parents=True)
        (d / "frame.json").write_text(json.dumps({"created_at": t0 * 1000, "completed_at": (t0 + 100) * 1000, "updated_at": None,
                                                  "model": "claude-sonnet-5", "effort": "low", "status": "completed",
                                                  "total_cost": 1.0, "input_tokens": 1, "output_tokens": 1, "_child_frames": []}))
        (d / "report.md").write_text("# A\nBody [1].\n\n## References\n[1] X. PMID: 10001.\n[2] Y. PMID: 30003.\n")
        (d / "cards.jsonl").write_text(json.dumps({"card": "Run Python code?", "decision": "allow"}) + "\n")
        logs.append({"t": t0 + 10, "task": TASK, "cond": cond, "tool": "search", "returned": ["10001"], "suppressed": [REVIEW]})
        logs.append({"t": t0 + 500, "task": TASK, "cond": cond, "tool": "search", "returned": ["30003"], "suppressed": []})
    (env / "out" / "mcp_call_logs").mkdir()
    (env / "out" / "mcp_call_logs" / "mcp_cs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in logs))
    runpy.run_path(str(COMM / "audit_claude_science.py"), run_name="__main__")
    rows = {r["cond"]: r for r in json.loads((env / "out" / "audit.json").read_text())}
    r = rows["fixinput"]
    assert r["P"] == 1 and r["cited_outside_P"] == ["30003"] and r["cited_outside_allow"] == ["30003"]
    assert r["suppressed"] == 1 and r["cards_allowed"] == 1 and r["cards_denied"] == 0 and not r["review_cited"]


class FakeLocator:
    def __init__(self, n=1, text="", on_click=None):
        self.n, self.text, self.on_click = n, text, on_click

    def count(self):
        return self.n

    @property
    def first(self):
        return self

    @property
    def last(self):
        return self

    def all(self):
        return [self] if self.n else []

    def click(self):
        self.on_click()

    def inner_text(self):
        return self.text


class FakePage:
    def __init__(self, title):
        self.title, self.clicked = title, None

    def get_by_role(self, role, name=None, exact=None):
        which = "deny" if name == "Deny" else "allow"
        return FakeLocator(1, on_click=lambda: setattr(self, "clicked", which))

    def get_by_text(self, rx):
        return FakeLocator(1, text=self.title)

    def locator(self, sel):
        return FakeLocator(1, text="print(1)")

    def wait_for_timeout(self, ms):
        pass


@pytest.mark.parametrize("title,decision", [("Run Python code?", "allow"), ("Run a shell command?", "allow"),
                                            ("Allow network access to pubmed.ncbi.nlm.nih.gov?", "deny"),
                                            ("Grant access to a folder?", "deny")])
def test_cs_approval_cards_allow_only_sandboxed_code(env, monkeypatch, title, decision):
    monkeypatch.setenv("CS_PROMPTS", str(env / "prompts"))
    R = load("cs_run", monkeypatch)
    log = []
    page = FakePage(title)
    assert R.handle_cards(page, log.append)
    assert page.clicked == decision and log[0]["decision"] == decision


def test_cs_run_reads_prompts_from_cs_prompts_and_never_nudges(env, monkeypatch):
    monkeypatch.setenv("CS_PROMPTS", str(env / "prompts"))
    R = load("cs_run", monkeypatch)
    assert R.PROMPTS == env / "prompts"
    assert "nudge" not in (COMM / "cs_run.py").read_text() and "best judgment" not in (COMM / "cs_run.py").read_text()


def test_cs_browser_and_worker_import(env, monkeypatch):
    monkeypatch.setenv("CS_DATA", str(env / "csd"))
    monkeypatch.setenv("CS_PROFILE", str(env / "profile"))
    B = load("cs_browser", monkeypatch)
    assert B.DATA == str(env / "csd") and callable(B.login)
    src = (COMM / "cs_worker.py").read_text()
    ast.parse(src)
    assert "exec(" not in src and "from cs_run import run_one" in src
    import tomllib
    cfg = tomllib.loads((COMM / "claude_science_config.toml").read_text())
    assert cfg == {"sandbox": {"network": {"enabled": False}}, "update": {"auto_update": False}}
