#!/usr/bin/env python3
"""HTTP proxy that forwards a system's chat calls to a vLLM port or a vendor API and logs one record
per call; in API mode it forces the model and settings and meters cost against a budget. Usage:
python llm_proxy.py --listen <port> (--upstream <port> | --upstream-url <url>) --log <file>."""

from __future__ import annotations
import argparse, json, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOCK = threading.Lock()
ARGS = None
SEQ = [0]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _pipe(self, method):
        if self.path.rstrip("/").endswith("/__proxy_health"):
            if not ARGS.upstream_url:
                out = json.dumps({"mode": "local"}).encode()
            else:
                out = json.dumps({
                    "mode": "api",
                    "force_model": ARGS.force_model,
                    "service_tier": ARGS.service_tier,
                    "reasoning_effort": ARGS.reasoning_effort,
                    "budget_usd": ARGS.budget_usd,
                    "warn_usd": ARGS.warn_usd,
                    "cost_file": ARGS.cost_file,
                    "log": ARGS.log,
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return
        length = int(self.headers.get("Content-Length") or 0)
        t_req = time.time()
        body = self.rfile.read(length) if length else b""
        requested_model = None
        if ARGS.upstream_url:
            path = self.path
            if path.startswith("/v1/"):
                path = path[3:]
            url = ARGS.upstream_url.rstrip("/") + path
            if body and self.path.endswith(("/chat/completions", "/completions", "/embeddings")):
                try:
                    j = json.loads(body)
                    requested_model = j.get("model")
                    if ARGS.force_model:
                        j["model"] = ARGS.force_model
                    if ARGS.service_tier and "service_tier" not in j:
                        j["service_tier"] = ARGS.service_tier
                    if ARGS.reasoning_effort and "reasoning_effort" not in j:
                        j["reasoning_effort"] = ARGS.reasoning_effort
                    if "max_tokens" in j and "max_completion_tokens" not in j:
                        j["max_completion_tokens"] = j.pop("max_tokens")
                    body = json.dumps(j).encode()
                except json.JSONDecodeError:
                    pass
                over = budget_state()
                if over is not None:
                    out = json.dumps({"error": {"message": over, "type": "budget_exceeded",
                                                "code": 402}}).encode()
                    self._record(body, out, 402, 0, requested_model, t_req)
                    self.send_response(402)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
        else:
            url = f"http://127.0.0.1:{ARGS.upstream}{self.path}"
        def _mk_req(b):
            r = urllib.request.Request(url, data=b if b else None, method=method)
            for h in ("Content-Type", "Accept"):
                if self.headers.get(h):
                    r.add_header(h, self.headers[h])
            if ARGS.upstream_url:
                r.add_header("Authorization", "Bearer " + API_KEY[0])
            elif self.headers.get("Authorization"):
                r.add_header("Authorization", self.headers["Authorization"])
            return r
        req = _mk_req(body)
        t0 = time.time()
        self._flex_retries = 0
        self._tier_escalated = False
        _FLEX_BACKOFF = (5, 15, 45, 90)
        while True:
            try:
                with urllib.request.urlopen(req, timeout=ARGS.timeout) as r:
                    out, status = r.read(), r.status
            except urllib.error.HTTPError as e:
                out, status = e.read(), e.code
            except Exception as e:
                out, status = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), 502
            transient = (ARGS.upstream_url and (
                (status in (400, 429, 500, 503) and b"sufficient resources" in out)
                or (status == 502 and b"RemoteDisconnected" in out)))
            if transient and self._flex_retries < len(_FLEX_BACKOFF):
                time.sleep(_FLEX_BACKOFF[self._flex_retries])
                self._flex_retries += 1
                req = _mk_req(body)
                continue
            if (transient and not self._tier_escalated and ARGS.service_tier == "flex" and body):
                try:
                    j2 = json.loads(body)
                    j2["service_tier"] = "default"
                    body = json.dumps(j2).encode()
                    self._tier_escalated = True
                    req = _mk_req(body)
                    continue
                except json.JSONDecodeError:
                    pass
            break
        ms = int((time.time() - t0) * 1000)

        if self.path.endswith(("/chat/completions", "/completions", "/embeddings")):
            self._record(body, out, status, ms, requested_model, t_req)

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _record(self, body, out, status, ms, requested_model=None, t_req=None):
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError:
            req = {"_unparsed_request": (body or b"")[:2000].decode("utf-8", "replace")}
        try:
            res = json.loads(out or b"{}")
        except json.JSONDecodeError:
            res = {"_unparsed_response": (out or b"")[:2000].decode("utf-8", "replace")}
        ch = (res.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        rec = {
            "path": self.path, "status": status, "wall_ms": ms,
            "flex_retries": getattr(self, "_flex_retries", 0) if ARGS.upstream_url else None,
            "tier_escalated": getattr(self, "_tier_escalated", False) if ARGS.upstream_url else None,
            "t": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "t_req": round(t_req, 3) if t_req is not None else None,
            "t_done": round(time.time(), 3),
            "model": req.get("model"),
            "requested_model": requested_model,
            "service_tier": req.get("service_tier"),
            "messages": req.get("messages"),
            "prompt": req.get("prompt"),
            "params": {k: req.get(k) for k in
                       ("temperature", "top_p", "max_tokens", "max_completion_tokens", "seed", "stop", "n")
                       if req.get(k) is not None},
            "completion": msg.get("content") if msg else ch.get("text"),
            "finish_reason": ch.get("finish_reason"),
            "usage": res.get("usage"),
            "system_fingerprint": res.get("system_fingerprint"),
            "error": res.get("error"),
        }
        if ARGS.upstream_url and ARGS.cost_file and status == 200:
            u = res.get("usage") or {}
            _pf = 2.0 if getattr(self, "_tier_escalated", False) else 1.0
            cin = (u.get("prompt_tokens") or 0) / 1e6 * ARGS.price_in * _pf
            cout = (u.get("completion_tokens") or 0) / 1e6 * ARGS.price_out * _pf
            add_cost(cin + cout, u, rec["t"], res.get("service_tier"))
            rec["cost_usd"] = round(cin + cout, 6)
        with LOCK:
            rec["seq"] = SEQ[0]
            SEQ[0] += 1
            with open(ARGS.log, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def do_POST(self):
        self._pipe("POST")

    def do_GET(self):
        self._pipe("GET")


API_KEY = [""]
_COST = {"total": None, "calls_since_refresh": 0}


def _read_cost_total():
    tot = 0.0
    try:
        with open(ARGS.cost_file) as f:
            for line in f:
                try:
                    tot += json.loads(line).get("cost_usd", 0.0)
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return tot


def _cap_usd():
    try:
        v = float(Path(ARGS.cost_file).with_suffix(".CAP").read_text().strip())
        if v > 0:
            return v
    except Exception:
        pass
    return ARGS.budget_usd


def budget_state():
    with LOCK:
        if _COST["total"] is None or _COST["calls_since_refresh"] >= 20:
            _COST["total"] = _read_cost_total()
            _COST["calls_since_refresh"] = 0
        _COST["calls_since_refresh"] += 1
        tot = _COST["total"]
    if ARGS.warn_usd and tot >= ARGS.warn_usd:
        warn = Path(ARGS.cost_file).with_suffix(".WARN")
        if not warn.exists():
            warn.write_text(f"cumulative OpenAI spend ${tot:.2f} crossed the ${ARGS.warn_usd} "
                            f"reporting threshold at {time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime())}\n")
            print(f"[proxy] WARN: cumulative spend ${tot:.2f} >= ${ARGS.warn_usd}", flush=True)
    cap = _cap_usd()
    if cap and tot >= cap:
        return (f"cumulative OpenAI spend ${tot:.2f} has reached the hard cap "
                f"${cap}; call refused.")
    return None


def add_cost(usd, usage, t, tier_applied=None):
    det = usage.get("prompt_tokens_details") or {}
    line = json.dumps({"t": t, "cost_usd": round(usd, 6),
                       "arm": ARGS.arm, "task": ARGS.task,
                       "prompt_tokens": usage.get("prompt_tokens"),
                       "completion_tokens": usage.get("completion_tokens"),
                       "cached_tokens": det.get("cached_tokens"),
                       "service_tier_applied": tier_applied,
                       "model": ARGS.force_model, "log": ARGS.log})
    with LOCK:
        with open(ARGS.cost_file, "a") as f:
            f.write(line + "\n")
        if _COST["total"] is not None:
            _COST["total"] += usd


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--upstream", type=int, help="local vLLM port")
    g.add_argument("--upstream-url", help="API base incl. /v1, e.g. https://api.openai.com/v1")
    ap.add_argument("--log", required=True)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--api-key-file", help="600-perm file holding the API key (API mode)")
    ap.add_argument("--force-model", help="replace every requested model with this")
    ap.add_argument("--service-tier", help="inject service_tier when absent (e.g. flex)")
    ap.add_argument("--reasoning-effort", help="inject reasoning_effort when absent")
    ap.add_argument("--price-in", type=float, default=0.0, help="$ per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=0.0, help="$ per 1M output tokens")
    ap.add_argument("--cost-file", help="shared cumulative-cost jsonl (API mode)")
    ap.add_argument("--budget-usd", type=float, default=0.0, help="hard refuse above this")
    ap.add_argument("--arm", default=None, help="arm name recorded in every cost line")
    ap.add_argument("--task", default=None, help="task id recorded in every cost line")
    ap.add_argument("--warn-usd", type=float, default=0.0, help="write .WARN above this")
    ARGS = ap.parse_args()
    if ARGS.upstream_url:
        assert ARGS.api_key_file, "--api-key-file is required in API mode"
        import os, stat
        st_ = os.stat(ARGS.api_key_file)
        assert not (st_.st_mode & (stat.S_IRGRP | stat.S_IROTH)), \
            f"{ARGS.api_key_file} is group/other-readable; chmod 600 it"
        API_KEY[0] = Path(ARGS.api_key_file).read_text().strip()
        assert API_KEY[0], "api key file is empty"
        assert ARGS.cost_file and ARGS.budget_usd > 0, \
            "API mode requires --cost-file and --budget-usd (accounting is not optional)"
    Path(ARGS.log).parent.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.listen), Handler)
    srv.daemon_threads = True
    print(f"[proxy] {ARGS.listen} -> {ARGS.upstream}, recording to {ARGS.log}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
