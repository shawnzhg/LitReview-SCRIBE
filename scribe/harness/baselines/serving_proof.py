#!/usr/bin/env python3
"""Writes the serving and code-integrity record of a pipeline arm: served model and settings, model
and pool hashes, commits and installed packages. Usage: python serving_proof.py --out <json>
[--api-mode]."""

import argparse, hashlib, json, os, re, shlex, subprocess, sys, datetime

def sha256_file(p, full=False):
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        d = h.hexdigest()
        return d if full else d[:16]
    except Exception:
        return None

def git_sha(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15).stdout.strip() or None
    except Exception:
        return None

def _versions(python):
    code = ("import json\n"
            "o={}\n"
            "for m in ('openai','httpx','requests'):\n"
            "    try:\n"
            "        o[m]=__import__(m).__version__\n"
            "    except Exception as e:\n"
            "        o[m]='ERR:'+type(e).__name__\n"
            "print(json.dumps(o))")
    try:
        r = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=300)
        return json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}


def emit_api(a):
    import urllib.request

    missing = [n for n, v in (("--api-base", a.api_base), ("--probe-url", a.probe_url),
                              ("--api-key-file", a.api_key_file),
                              ("--force-model", a.force_model),
                              ("--cost-file", a.cost_file), ("--budget-usd", a.budget_usd))
               if not v]
    if missing:
        sys.exit("serving_proof --api-mode requires: " + ", ".join(missing))

    import stat as _stat
    try:
        m = os.stat(a.api_key_file).st_mode
        perms_ok = not (m & (_stat.S_IRGRP | _stat.S_IROTH))
        perms = oct(m & 0o777)
    except Exception as e:
        perms_ok, perms = False, "ERR:%s" % type(e).__name__

    sig, probe_err = None, None
    try:
        url = a.probe_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        with urllib.request.urlopen(url.rstrip("/") + "/__proxy_health", timeout=30) as r:
            sig = json.loads(r.read())
    except Exception as e:
        probe_err = "%s: %s" % (type(e).__name__, str(e)[:200])

    mism = []
    if sig is not None:
        if sig.get("mode") != "api":
            mism.append("mode=%r" % sig.get("mode"))
        for k, want in (("force_model", a.force_model), ("service_tier", a.service_tier),
                        ("reasoning_effort", a.reasoning_effort)):
            if want and sig.get(k) != want:
                mism.append("%s=%r declared %r" % (k, sig.get(k), want))
        if a.budget_usd and sig.get("budget_usd") != a.budget_usd:
            mism.append("budget_usd=%r declared %r" % (sig.get("budget_usd"), a.budget_usd))

    spent = None
    try:
        tot = 0.0
        with open(a.cost_file) as f:
            for line in f:
                try:
                    tot += json.loads(line).get("cost_usd", 0.0)
                except json.JSONDecodeError:
                    continue
        spent = round(tot, 6)
    except FileNotFoundError:
        spent = 0.0
    except Exception:
        pass

    try:
        freeze = subprocess.run([a.python, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=300).stdout
    except Exception:
        freeze = ""

    out = {
        "schema": "serving_proof/1.1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "serving": {
            "mode": "api",
            "engine": "openai-compatible vendor API behind scribe/harness/runners/llm_proxy.py",
            "api_base": a.api_base,
            "probe_url": a.probe_url,
            "model": a.force_model,
            "service_tier": a.service_tier,
            "reasoning_effort": a.reasoning_effort,
            "price_in_per_mtok_usd": a.price_in,
            "price_out_per_mtok_usd": a.price_out,
            "budget_usd": a.budget_usd,
            "cost_file": a.cost_file,
            "cost_file_total_usd_at_proof_time": spent,
            "key_file": a.api_key_file,
            "key_file_perms": perms,
            "key_file_perms_ok": perms_ok,
            "live_probe": {
                "endpoint": "/__proxy_health",
                "ok": sig is not None and not mism,
                "health_signature": sig,
                "error": probe_err,
                "mismatches": mism,
            },
        },
        "versions": _versions(a.python),
        "model_dir": None,
        "pool_manifest_sha256": sha256_file(a.pool_manifest, full=True) if a.pool_manifest else None,
        "git_sha": {r: git_sha(r) for r in a.repo},
        "pip_freeze_sha256_16": hashlib.sha256(freeze.encode()).hexdigest()[:16] if freeze else None,
        "pip_freeze_n_packages": len([l for l in freeze.splitlines() if l.strip()]),
        "deviations": [
            {"what": "serving", "detail":
             "Served by %s over the vendor API at service_tier %r. The recording proxy rewrites "
             "every requested model name to that model; the requested name is kept per call in "
             "_calls.jsonl as requested_model." % (a.force_model, a.service_tier)},
            {"what": "budget", "detail":
             "Hard cap $%s enforced at the proxy, which refuses with HTTP 402 once the shared "
             "cost file total reaches it. Accounting is per call in %s."
             % (a.budget_usd, a.cost_file)},
        ],
    }

    if probe_err or mism:
        sys.stderr.write(
            "serving_proof: FATAL: the live probe through the sanctioned lane %s.\n"
            "  error: %s\n  mismatches: %s\n"
            % ("failed" if probe_err else "disagreed with the declared configuration",
               probe_err, mism))
        sys.exit(3)
    if not perms_ok:
        sys.stderr.write("serving_proof: FATAL: %s is group/other-readable (%s); chmod 600 it.\n"
                         % (a.api_key_file, perms))
        sys.exit(3)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    with open(a.out + ".pipfreeze.txt", "w") as f:
        f.write(freeze)
    print("serving_proof written: %s  (api mode, model=%s tier=%s spent=$%s of $%s)"
          % (a.out, a.force_model, a.service_tier, spent, a.budget_usd))
    return 0


def serve_flags(cmd):
    argv = shlex.split(cmd)
    out = {}
    for k in ("--max-model-len", "--gpu-memory-utilization", "--dtype", "--max-num-seqs"):
        out[k[2:].replace("-", "_")] = argv[argv.index(k) + 1] if k in argv[:-1] else None
    return argv, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir", default=None, help="local mode: the served model directory")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--api-mode", action="store_true",
                    help="serving is a vendor API behind the recording proxy, not a local vLLM")
    ap.add_argument("--api-base", help="api mode: vendor base URL the proxy forwards to")
    ap.add_argument("--probe-url", help="api mode: base URL of the proxy as the app sees it")
    ap.add_argument("--api-key-file", help="api mode: the 600-perm key file (never read)")
    ap.add_argument("--force-model", help="api mode: model every request is rewritten to")
    ap.add_argument("--service-tier", help="api mode: injected when the request omits it")
    ap.add_argument("--reasoning-effort", help="api mode: injected when the request omits it")
    ap.add_argument("--price-in", type=float, help="api mode: $ per 1M input tokens")
    ap.add_argument("--price-out", type=float, help="api mode: $ per 1M output tokens")
    ap.add_argument("--budget-usd", type=float, help="api mode: hard cap enforced by the proxy")
    ap.add_argument("--cost-file", help="api mode: shared cumulative-cost jsonl")
    ap.add_argument("--pool-manifest", default=None, help="path to frozen pool manifest")
    ap.add_argument("--repo", action="append", default=[], help="repo path(s) for git SHAs")
    ap.add_argument("--vllm-log", default=None, help="local mode: the vLLM startup log")
    ap.add_argument("--serve-cmd", default=None, help="local mode: the vLLM command line as run")
    a = ap.parse_args()

    if a.api_mode:
        return emit_api(a)

    for req, name in ((a.model_dir, "--model-dir"), (a.serve_cmd, "--serve-cmd"),
                      (a.vllm_log, "--vllm-log")):
        if not req:
            sys.exit("serving_proof: %s is required in local mode" % name)
    argv, flags = serve_flags(a.serve_cmd)

    md = {}
    for f in ("config.json", "tokenizer_config.json", "chat_template.jinja",
              "generation_config.json", "model.safetensors.index.json"):
        p = os.path.join(a.model_dir, f)
        md[f] = sha256_file(p) if os.path.exists(p) else "ABSENT"

    vers = {}
    code = ("import json,sys\n"
            "o={}\n"
            "for m in ('vllm','torch','flashinfer','transformers'):\n"
            "    try:\n"
            "        o[m]=__import__(m).__version__\n"
            "    except Exception as e:\n"
            "        o[m]='ERR:'+type(e).__name__\n"
            "print(json.dumps(o))")
    try:
        r = subprocess.run([a.python, "-c", code], capture_output=True, text=True, timeout=300)
        vers = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])
    except Exception as e:
        vers = {"error": f"{type(e).__name__}: {e}"}

    try:
        freeze = subprocess.run([a.python, "-m", "pip", "freeze"],
                                capture_output=True, text=True, timeout=300).stdout
    except Exception:
        freeze = ""

    backends = {}
    if os.path.exists(a.vllm_log):
        pats = {
            "sampler":   r"Using (FlashInfer|PyTorch|flashinfer) for top-p & top-k sampling",
            "attention": r"Using ([A-Z_]+) attention backend",
        }
        txt = open(a.vllm_log, errors="replace").read()
        for k, pat in pats.items():
            m = re.findall(pat, txt)
            backends[k] = m[-1] if m else None
    missing = [k for k in ("sampler", "attention") if backends.get(k) is None]
    if missing:
        sys.stderr.write(f"serving_proof: FATAL: could not read {missing} from {a.vllm_log}\n")
        sys.exit(3)

    out = {
        "schema": "serving_proof/1.2",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "serving": {
            "engine": "vllm",
            "serve_cmd": argv,
            "backends_from_startup_log": backends,
            "backends_source": os.path.abspath(a.vllm_log),
            **flags,
        },
        "versions": vers,
        "model_dir": {"path": a.model_dir, "file_sha256_16": md},
        "pool_manifest_sha256": sha256_file(a.pool_manifest, full=True) if a.pool_manifest else None,
        "git_sha": {r: git_sha(r) for r in a.repo},
        "pip_freeze_sha256_16": hashlib.sha256(freeze.encode()).hexdigest()[:16] if freeze else None,
        "pip_freeze_n_packages": len([l for l in freeze.splitlines() if l.strip()]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    with open(a.out + ".pipfreeze.txt", "w") as f:
        f.write(freeze)
    print(f"serving_proof written: {a.out}")

if __name__ == "__main__":
    main()
