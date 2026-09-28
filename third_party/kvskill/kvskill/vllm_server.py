"""HTTP rollout server of a KV-Skill carrier on vLLM for GRPO, with payload reload and sleep and
wake. Usage: python -m kvskill.vllm_server --theta_payload <file> --model <dir> --port <port>."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROLLOUT_TEMPERATURE = 1.0


class SkillService:
    def __init__(self, args):
        from safetensors import safe_open

        with safe_open(args.theta_payload, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
        assert meta.get("cn") == "1", f"{args.theta_payload}: not a KV-Skill payload"
        os.environ["KVSKILL_CN_PATH"] = str(args.theta_payload)
        os.environ["KVSKILL_CN_SHAPE"] = ",".join(
            (meta["cn_ds"], meta["cn_dm"], meta["cn_r"], meta["cn_nlayers"]))
        root = os.environ.get("VLLM_CACHE_ROOT") or os.path.expanduser("~/.cache/vllm")
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(root, f"cn_ds{meta['cn_ds']}_r{meta['cn_r']}")
        from vllm import LLM
        self.theta_version = 0
        self.sleeping = False
        self.llm = LLM(model=args.model, dtype="bfloat16",
                       gpu_memory_utilization=args.gpu_memory_utilization,
                       max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                       trust_remote_code=False, seed=args.seed, enable_prefix_caching=False,
                       enable_sleep_mode=True, logprobs_mode="processed_logprobs",
                       worker_extension_cls="kvskill.vllm_patch.SkillWorkerExtension")
        self.lock = threading.Lock()
        print(f"[cn-server] engine up (depths={meta['cn_depths']})", flush=True)

    def _awake(self, ep):
        if self.sleeping:
            raise RuntimeError(f"{ep} while asleep: POST /wake first")

    def sleep(self, level=1):
        with self.lock:
            if self.sleeping:
                return {"ok": True, "sleeping": True, "noop": True}
            t0 = time.time(); self.llm.sleep(level=level); self.sleeping = True
            dt = time.time() - t0
        print(f"[cn-server] asleep (level={level}) in {dt:.1f}s", flush=True)
        return {"ok": True, "sleeping": True, "t_sleep": dt}

    def wake(self):
        with self.lock:
            if not self.sleeping:
                return {"ok": True, "sleeping": False, "noop": True}
            t0 = time.time(); self.llm.wake_up(); self.sleeping = False
            dt = time.time() - t0
        print(f"[cn-server] awake in {dt:.1f}s", flush=True)
        return {"ok": True, "sleeping": False, "t_wake": dt}

    def reload(self, path: str) -> int:
        with self.lock:
            self._awake("/reload")
            vs = self.llm.collective_rpc("kvskill_cn_reload", args=(str(path),))
        self.theta_version = int(vs[0]) if vs else self.theta_version + 1
        return self.theta_version

    def generate(self, prompts, *, max_new_tokens, stop_token_ids):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        reqs = [TokensPrompt(prompt_token_ids=list(ids)) for ids in prompts]
        sp = SamplingParams(temperature=ROLLOUT_TEMPERATURE, top_p=1.0, top_k=-1,
                            max_tokens=max_new_tokens, logprobs=0,
                            stop_token_ids=list(stop_token_ids), detokenize=False)
        with self.lock:
            self._awake("/generate")
            outs = self.llm.generate(reqs, sp)
        stop_set = set(stop_token_ids)
        results = []
        for o in outs:
            seq = o.outputs[0]
            tids = list(seq.token_ids)
            row_lp = []
            for t, entry in zip(tids, seq.logprobs or []):
                assert entry is not None and t in entry, f"missing logprob for {t}"
                row_lp.append(float(entry[t].logprob))
            while tids and tids[-1] in stop_set:
                tids.pop()
                if row_lp:
                    row_lp.pop()
            assert len(row_lp) == len(tids), (len(row_lp), len(tids))
            results.append({"token_ids": tids, "logprobs": row_lp,
                            "finish_reason": seq.finish_reason,
                            "n_prompt_tokens": len(o.prompt_token_ids or [])})
        return results


def _handler(svc: SkillService):
    class H(BaseHTTPRequestHandler):
        def _json(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                import kvskill.vllm_patch as vp
                self._json(200, {"ok": True, "version": svc.theta_version,
                                 "sleeping": svc.sleeping, "v_norm": float(vp._STAGED_V_NORM)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", "0") or 0)
                d = json.loads(self.rfile.read(n).decode() or "{}")
                if self.path == "/reload":
                    self._json(200, {"ok": True, "version": svc.reload(d["path"])})
                elif self.path == "/sleep":
                    self._json(200, svc.sleep(level=int(d.get("level", 1))))
                elif self.path == "/wake":
                    self._json(200, svc.wake())
                elif self.path == "/generate":
                    self._json(200, {"results": svc.generate(
                        d["prompts"], max_new_tokens=int(d["max_new_tokens"]),
                        stop_token_ids=list(d.get("stop_token_ids", [])))})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self._json(500, {"error": f"{type(exc).__name__}: {exc}",
                                 "traceback": traceback.format_exc()})

        def log_message(self, *a):
            return
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta_payload", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--max_model_len", type=int, default=65536)
    ap.add_argument("--max_num_seqs", type=int, default=48)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    svc = SkillService(a)
    httpd = ThreadingHTTPServer((a.host, a.port), _handler(svc))
    print(f"[cn-server] READY on http://{a.host}:{a.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
