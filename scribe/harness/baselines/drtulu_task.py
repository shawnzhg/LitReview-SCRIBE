#!/usr/bin/env python3
"""Runs DR-Tulu on one task with its search routes pointed at the pool service and writes the
report, trace and search log. Usage: python drtulu_task.py --task-id <task> --task-input <json>
--out-dir <dir> --llm-base-url <url> --pool-port <port>."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

DRIVER_VERSION = "drtulu_task/1.0"
DATASET_NAME_DEFAULT = "sqav2"

EX_OK, EX_ARGS, EX_MCP, EX_WORKFLOW, EX_EMPTY, EX_ENV = 0, 2, 3, 4, 5, 90


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    print(f"[drtulu_task] {msg}", flush=True)


def die(code: int, msg: str) -> "None":
    sys.stderr.write(f"[drtulu_task] FATAL({code}): {msg}\n")
    sys.stderr.flush()
    sys.exit(code)


def port_listening(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def start_mcp_backend(python: str, agent_dir: str, port: int, cwd: str,
                      logfile: str, env: dict) -> subprocess.Popen:
    os.makedirs(os.path.dirname(os.path.abspath(logfile)) or ".", exist_ok=True)
    fh = open(logfile, "wb", buffering=0)
    cmd = [python, "-m", "dr_agent.mcp_backend.main", "--port", str(port)]
    log(f"starting MCP backend: {' '.join(cmd)}  (cwd={cwd})")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    proc._drtulu_logfh = fh
    return proc


def wait_mcp_ready(proc: subprocess.Popen, port: int, deadline_s: float,
                   logfile: str) -> None:
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        if proc.poll() is not None:
            tail = ""
            try:
                with open(logfile, errors="replace") as f:
                    tail = "".join(f.readlines()[-40:])
            except Exception:
                pass
            die(EX_MCP, f"MCP backend exited with rc={proc.returncode} before becoming "
                        f"ready.\n--- {logfile} (tail) ---\n{tail}")
        if port_listening(port):
            log(f"MCP backend ready on 127.0.0.1:{port} after {time.time() - t0:.1f}s")
            return
        time.sleep(1.0)
    die(EX_MCP, f"MCP backend did not listen on 127.0.0.1:{port} within {deadline_s:.0f}s")


def kill_process_group(proc: subprocess.Popen, name: str, grace: float = 5.0) -> None:
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            break
        try:
            if pgid is not None:
                os.killpg(pgid, sig)
            else:
                proc.send_signal(sig)
        except ProcessLookupError:
            break
        except Exception as e:
            log(f"warning: could not signal {name} with {sig}: {e}")
            break
        t0 = time.time()
        while time.time() - t0 < grace and proc.poll() is None:
            time.sleep(0.2)
    try:
        fh = getattr(proc, "_drtulu_logfh", None)
        if fh is not None:
            fh.close()
    except Exception:
        pass
    log(f"{name} reaped (rc={proc.returncode})")


def write_atomic_json(path: str, obj: dict) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(d, os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def jsonable(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return jsonable(fn())
            except Exception:
                pass
    return repr(obj)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    r = p.add_argument_group("required")
    r.add_argument("--task-id", required=True,
                   help="task id; the pool service picks the task's cutoff from it")
    r.add_argument("--task-input", required=True,
                   help="the task's refs.json; only its 'topic' is given to the agent")
    r.add_argument("--out-dir", required=True, help="per-task output directory")
    r.add_argument("--llm-base-url", required=True,
                   help="OpenAI-compatible base URL of the recording proxy in front of vLLM")
    r.add_argument("--pool-port", type=int, required=True,
                   help="port of the pool service on loopback")

    o = p.add_argument_group("deployment")
    o.add_argument("--agent-dir", default=None,
                   help="agent directory of the pinned dr-tulu clone (default: $DRTULU_AGENT_DIR)")
    o.add_argument("--model-name", default=os.environ.get("DRTULU_MODEL_NAME", "DR-Tulu-8B"),
                   help="vLLM's --served-model-name (default: DR-Tulu-8B)")
    o.add_argument("--tokenizer", default=os.environ.get("DRTULU_TOKENIZER"),
                   help="local DR-Tulu-8B directory; dr_agent applies its chat template")
    o.add_argument("--python", default=sys.executable,
                   help="interpreter of the MCP backend (default: this one)")
    p.add_argument("--pool-log", default=os.environ.get("POOL_LOG", ""),
                   help="the pool service log; this task's lines are copied to "
                        "<out-dir>/search_calls.jsonl (default: $POOL_LOG)")
    return p


def main() -> int:
    args = build_parser().parse_args()

    started = time.time()
    started_utc = utcnow()

    agent_dir = args.agent_dir or os.environ.get("DRTULU_AGENT_DIR")
    if not agent_dir:
        die(EX_ARGS, "--agent-dir (or $DRTULU_AGENT_DIR) is required")
    agent_dir = os.path.abspath(agent_dir)
    if not os.path.isdir(os.path.join(agent_dir, "dr_agent")):
        die(EX_ENV, f"--agent-dir {agent_dir} does not contain dr_agent/ -- this must be the "
                    f"agent directory of the pinned clone")
    config = os.path.join(agent_dir, "workflows", "auto_search_sft.yaml")
    if not os.path.isfile(config):
        die(EX_ENV, f"workflow config not found: {config}")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(args.task_input):
        die(EX_ARGS, f"--task-input not found: {args.task_input}")
    try:
        spec = json.load(open(args.task_input, encoding="utf-8"))
    except Exception as e:
        die(EX_ARGS, f"--task-input does not parse as JSON: {e}")
    topic = (spec.get("topic") or "").strip()
    if not topic:
        die(EX_ARGS, f"--task-input has no non-empty 'topic': {args.task_input}")
    spec_task = spec.get("task_id")
    if spec_task and spec_task != args.task_id:
        die(EX_ARGS, f"--task-id {args.task_id!r} disagrees with the input file's task_id "
                     f"{spec_task!r}")
    cutoff_year = spec.get("cutoff_year")

    if not args.tokenizer or not os.path.isdir(args.tokenizer):
        die(EX_ENV, f"--tokenizer must be a local model directory (got {args.tokenizer!r})")
    if not os.path.isfile(os.path.join(args.tokenizer, "tokenizer_config.json")):
        die(EX_ENV, f"{args.tokenizer} has no tokenizer_config.json")

    mcp_port = args.pool_port + 1000
    s2_base = f"http://127.0.0.1:{args.pool_port}/t/{args.task_id}/graph/v1"

    cache_dir = os.path.join(out_dir, "mcp_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    env = os.environ.copy()
    env["S2_GRAPH_API_URL"] = s2_base
    env["MCP_CACHE_DIR"] = cache_dir
    env["PYTHONPATH"] = os.pathsep.join(
        [agent_dir] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.pop("SERP_API_KEY", None)
    env.pop("SERPER_API_KEY", None)
    env.pop("JINA_API_KEY", None)
    os.environ["S2_GRAPH_API_URL"] = s2_base
    os.environ["MCP_CACHE_DIR"] = cache_dir

    log(f"task={args.task_id} cutoff={cutoff_year} topic={topic!r}")
    log(f"out_dir={out_dir}")
    log(f"S2_GRAPH_API_URL={s2_base}")
    log(f"MCP_CACHE_DIR={cache_dir}")
    log(f"llm_base_url={args.llm_base_url} model={args.model_name}")

    report_path = os.path.join(out_dir, "report.json")
    if os.path.exists(report_path):
        os.remove(report_path)

    mcp_log = os.path.join(out_dir, "mcp_backend.log")
    proc = None
    rc = EX_OK
    try:
        proc = start_mcp_backend(args.python, agent_dir, mcp_port, out_dir, mcp_log, env)
        wait_mcp_ready(proc, mcp_port, 180.0, mcp_log)

        if agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        try:
            from workflows.auto_search_sft import AutoReasonSearchWorkflow
        except Exception as e:
            traceback.print_exc()
            die(EX_ENV, f"could not import the shipped workflow from {agent_dir}: {e}")

        overrides = dict(
            search_tool_name="s2-only",
            browse_tool_name=None,
            use_browse_agent=False,
            search_agent_base_url=args.llm_base_url,
            search_agent_model_name=args.model_name,
            search_agent_tokenizer_name=args.tokenizer,
            mcp_port=mcp_port,
        )
        log("workflow overrides: " + json.dumps(
            {k: v for k, v in overrides.items()}, ensure_ascii=False))

        wf = AutoReasonSearchWorkflow(
            configuration=config,
            skip_service_check=True,
            **overrides,
        )
        effective = {}
        try:
            effective = wf.configuration_dict()
        except Exception:
            pass

        async def run():
            return await wf(problem=topic, dataset_name=DATASET_NAME_DEFAULT, verbose=False)

        try:
            res = asyncio.run(run())
        except Exception as e:
            traceback.print_exc()
            die(EX_WORKFLOW, f"workflow raised: {type(e).__name__}: {e}")

        answer = (res.get("final_response") or "").strip()
        traces = res.get("full_traces")

        trace_obj = {
            "schema": "drtulu_trace/1.0",
            "task_id": args.task_id,
            "topic": topic,
            "full_traces": jsonable(traces),
        }
        try:
            write_atomic_json(os.path.join(out_dir, "trace.json"), trace_obj)
        except Exception as e:
            log(f"warning: could not write trace.json: {e}")
        try:
            with open(os.path.join(out_dir, "answer.md"), "w", encoding="utf-8") as f:
                f.write(answer)
        except Exception as e:
            log(f"warning: could not write answer.md: {e}")

        if not answer:
            die(EX_EMPTY, "the agent returned an empty answer; no report.json is written")

        tstats = {}
        for k in ("total_tokens", "tool_call_count", "stopped_reason"):
            v = getattr(traces, k, None)
            if v is not None:
                tstats[k] = v

        report = {
            "schema": "drtulu_report/1.0",
            "driver": {
                "version": DRIVER_VERSION,
                "file": os.path.abspath(__file__),
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "host": os.uname().nodename,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
            "task_id": args.task_id,
            "topic": topic,
            "cutoff_year": cutoff_year,
            "dataset_name": DATASET_NAME_DEFAULT,
            "answer": answer,
            "answer_chars": len(answer),
            "answer_words": len(answer.split()),
            "total_tool_calls": res.get("total_tool_calls"),
            "total_failed_tool_calls": res.get("total_failed_tool_calls"),
            "failed_tool_call_errors": res.get("failed_tool_call_errors"),
            "searched_links": res.get("searched_links"),
            "browsed_links": res.get("browsed_links"),
            "trace_stats": tstats,
            "endpoints": {
                "llm_base_url": args.llm_base_url,
                "s2_graph_api_url": s2_base,
                "pool_port": args.pool_port,
                "mcp_port": mcp_port,
            },
            "mcp": {
                "cache_dir": cache_dir,
                "pid": proc.pid,
                "log": mcp_log,
                "flags": ["--port", str(mcp_port)],
            },
            "config": {
                "path": config,
                "overrides": {k: v for k, v in overrides.items()},
                "effective": jsonable(effective),
            },
            "timing": {
                "started_utc": started_utc,
                "finished_utc": utcnow(),
                "wall_s": round(time.time() - started, 2),
            },
        }
        split_path = os.path.join(out_dir, "search_calls.jsonl")
        split_n = 0
        if args.pool_log and os.path.exists(args.pool_log):
            tmp_split = split_path + ".tmp"
            with open(args.pool_log, "r", encoding="utf-8", errors="replace") as src, \
                 open(tmp_split, "w", encoding="utf-8") as dst:
                for line in src:
                    try:
                        if json.loads(line).get("task") == args.task_id:
                            dst.write(line if line.endswith("\n") else line + "\n")
                            split_n += 1
                    except Exception:
                        continue
            os.replace(tmp_split, split_path)
            log(f"search_calls split: {split_n} line(s) -> {split_path}")
        else:
            log(f"WARNING: pool log not found ({args.pool_log!r}); search_calls.jsonl not written")
        report["search_calls"] = {
            "shard_log": args.pool_log or None,
            "split_lines": split_n,
            "split_written": bool(args.pool_log and os.path.exists(split_path)),
        }
        write_atomic_json(report_path, report)
        log(f"OK {args.task_id}: {len(answer)} chars, "
            f"tool_calls={report['total_tool_calls']}, "
            f"failed={report['total_failed_tool_calls']}, "
            f"{report['timing']['wall_s']}s -> {report_path}")
        rc = EX_OK
    except SystemExit as e:
        rc = int(e.code or 0)
    except Exception as e:
        traceback.print_exc()
        sys.stderr.write(f"[drtulu_task] FATAL({EX_WORKFLOW}): unexpected: "
                         f"{type(e).__name__}: {e}\n")
        rc = EX_WORKFLOW
    finally:
        kill_process_group(proc, f"MCP backend (pid={getattr(proc, 'pid', None)})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
