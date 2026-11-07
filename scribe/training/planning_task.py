#!/usr/bin/env python3
"""Planning-window training task for kvskill: each item is one training review's outline prompt over
the synthesis graph of its base run with the initial carriers, scored by the planning-exit reward."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
RUNNERS = HERE.parent / "harness" / "runners"
if str(RUNNERS) not in sys.path:
    sys.path.insert(0, str(RUNNERS))

import windows as W
from calllog import CallLog
from llm import Draw

BASE_CACHE = os.environ.get("SCRIBE_BASE_CACHE", "")
TASKS_FILE = os.environ.get("SCRIBE_PLAN_TASKS_FILE", "")
RUNS = Path(os.environ.get("SCRIBE_RUNS_ROOT") or "/nonexistent/SCRIBE_RUNS_ROOT")
SPEC_ROOT = RUNS / "taskspecs" / "train"
EVAL_TASKS_PATH = os.environ.get("SCRIBE_EVAL_TASKS") or "/nonexistent/SCRIBE_EVAL_TASKS"
EVAL_EXCLUDED_PATH = os.environ.get(
    "SCRIBE_EVAL_EXCLUDED", str(HERE.parent.parent / "configs" / "train_tasks" / "eval_excluded.txt"))
PLANNING_EXIT = "planning_exit"
FAILURE_REWARD = 0.0
REWARD_THREADS = 8
BASE_FILES = ("synthesis_graph.json", "outline_plan.json")


class ReplayClient:

    def __init__(self, text: str, finish_reason: str = "stop", n_completion_tokens: int = 0):
        self._text = text
        self._finish_reason = finish_reason
        self._n_completion_tokens = int(n_completion_tokens or 0)

    def chat(self, messages, max_tokens=0, seed=0, **_):
        prompt = "\n\n".join(m.get("content", "") for m in messages)
        return Draw(prompt=prompt, completion=self._text, finish_reason=self._finish_reason,
                    params={"max_tokens": int(max_tokens), "seed": int(seed or 0), "replay": True},
                    n_prompt_tokens=0, n_completion_tokens=self._n_completion_tokens,
                    wall_ms=0, model="replay",
                    role_sequence=[m.get("role", "") for m in messages], error=None)


def plan_messages(spec, graph):
    seen = {}

    class _Capture:
        def chat(self, messages, **kw):
            seen["messages"] = list(messages)
            return Draw(prompt="", completion="", finish_reason="error", params={},
                        error="capture")

    try:
        W.planning(spec, graph, _Capture(), CallLog(Path("/dev/null"), "capture", "capture"), seed=0)
    except Exception:
        pass
    if "messages" not in seen:
        raise RuntimeError("windows.planning made no call")
    return seen["messages"]


def base_dir(cache_dir, task_id: str) -> Path:
    return Path(cache_dir) / task_id / "seed0"


def has_base(cache_dir, task_id: str) -> bool:
    d = base_dir(cache_dir, task_id)
    return all((d / f).is_file() for f in BASE_FILES)


def load_base(cache_dir, task_id: str) -> dict:
    d = base_dir(cache_dir, task_id)
    if not has_base(cache_dir, task_id):
        raise FileNotFoundError(f"no base run for {task_id}: {d} must hold {list(BASE_FILES)}")
    spec_p = SPEC_ROOT / f"{task_id}.json"
    if not spec_p.is_file():
        raise FileNotFoundError(f"no training TaskSpec {spec_p}")
    return {"spec": json.loads(spec_p.read_text()),
            "graph": json.loads((d / "synthesis_graph.json").read_text()),
            "outline_plan": json.loads((d / "outline_plan.json").read_text()),
            "run_dir": str(d)}


def eval_task_ids() -> set:
    try:
        return set(json.loads(Path(EVAL_TASKS_PATH).read_text())["tasks"])
    except (OSError, KeyError, json.JSONDecodeError):
        return set()


def eval_excluded_ids() -> set:
    try:
        return {l.strip() for l in open(EVAL_EXCLUDED_PATH) if l.strip()}
    except OSError:
        return set()


def refuse_eval(task_ids, where: str) -> None:
    ev = eval_task_ids()
    if not ev:
        raise ValueError(f"{where}: the evaluation task list {EVAL_TASKS_PATH} could not be read. "
                         f"Set $SCRIBE_EVAL_TASKS.")
    bad = sorted(set(task_ids) & ev)
    if bad:
        raise ValueError(f"{where}: {len(bad)} of the {len(set(task_ids))} tasks are evaluation "
                         f"tasks: {bad[:5]}")
    ex = eval_excluded_ids()
    if not ex:
        raise ValueError(f"{where}: the exclusion list {EVAL_EXCLUDED_PATH} could not be read. Build "
                         f"it with scribe/training/build_train_tasks.py or set $SCRIBE_EVAL_EXCLUDED.")
    badx = sorted(set(task_ids) & ex)
    if badx:
        raise ValueError(f"{where}: {len(badx)} of the {len(set(task_ids))} tasks are peers or "
                         f"references of an evaluation task: {badx[:5]}")


class PlanningTask:
    name = "planning"

    def __init__(self):
        if not BASE_CACHE:
            raise ValueError("planning needs SCRIBE_BASE_CACHE (the base runs of the training tasks)")
        if not TASKS_FILE:
            raise ValueError("planning needs SCRIBE_PLAN_TASKS_FILE (the training task list)")
        tasks = [l.strip() for l in open(TASKS_FILE) if l.strip()]
        refuse_eval(tasks, f"planning task list {TASKS_FILE}")
        missing = [t for t in tasks if not has_base(BASE_CACHE, t)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} of {len(tasks)} tasks have no base run under "
                                    f"{BASE_CACHE}: {missing[:5]}")
        sha = hashlib.sha256(Path(TASKS_FILE).read_bytes()).hexdigest()
        self._bases: dict = {}
        self.train_items = [{"id": f"{t}#plan", "task": t} for t in tasks]
        print(f"[planning] items: train {len(self.train_items)} (list {Path(TASKS_FILE).name} "
              f"sha256 {sha[:16]}); base cache {BASE_CACHE}", flush=True)

    def base(self, task_id: str) -> dict:
        b = self._bases.get(task_id)
        if b is None:
            b = self._bases[task_id] = load_base(BASE_CACHE, task_id)
        return b

    def build_messages(self, item: dict) -> list[dict]:
        b = self.base(item["task"])
        return plan_messages(b["spec"], b["graph"])


EVAL_ROOT = str(HERE.parent.parent / "evaluation")
_CC_LOCK = threading.Lock()
_CC_READY = False


def _import_ccbench() -> None:
    global _CC_READY
    with _CC_LOCK:
        if _CC_READY:
            return
        if EVAL_ROOT not in sys.path:
            sys.path.insert(0, EVAL_ROOT)
        from ccbench.config import prereg
        from ccbench.readouts import outline as ccoutline
        import rapidfuzz
        _CC_READY = True


def parse_candidate_plan(spec: dict, graph: dict, text: str, *, finish_reason: str = "stop",
                         n_completion_tokens: int = 0):
    replay = ReplayClient(text, finish_reason=finish_reason, n_completion_tokens=n_completion_tokens)
    log = CallLog(Path("/dev/null"), PLANNING_EXIT, str(spec.get("task_id") or ""))
    plan, _errs = W.planning(spec, graph, replay, log, seed=0)
    return plan


@dataclasses.dataclass
class RewardResult:
    reward: float
    components: dict = dataclasses.field(default_factory=dict)
    status: str = "ok"


class PlanningExitReward:

    def __init__(self, base_cache, embed=None, bands=None, targets=None):
        self.base_cache = base_cache
        self._bases: dict = {}
        self._lock = threading.Lock()
        self.stats: dict = {}
        _import_ccbench()
        import planning_exit_readouts as PE
        import readout_reward as RR
        self.PE, self.RR = PE, RR
        if bands is None:
            path = os.environ.get("SCRIBE_PLAN_BANDS", "")
            if not path:
                raise ValueError("planning_exit needs $SCRIBE_PLAN_BANDS (precompute_train_bands.py)")
            bands = PE.load_bands(path)
        if targets is None:
            path = os.environ.get("SCRIBE_READOUT_TARGETS", "")
            if not path:
                raise ValueError("planning_exit needs $SCRIBE_READOUT_TARGETS (reward_weights.py)")
            targets = RR.load_targets(path)
        if PE.WINDOW not in (targets.get("window_sets") or {}):
            raise ValueError(f"targets {targets.get('_path')} have no {PE.WINDOW!r} window set")
        self.bands, self.targets = bands, targets
        self.embed = embed if embed is not None else PE.NomicHTTPEmbed(os.environ.get("SCRIBE_EMBED_URL", ""))
        self.cc = PE.make_cc(self.embed)
        for b in (bands.get("tasks") or {}).values():
            if b.get("hum_top_emb_b64") and b.get("emb_dim") and hasattr(self.embed, "seed"):
                self.embed.seed(b["hum_top"], PE.from_b64_f32(b["hum_top_emb_b64"], int(b["emb_dim"])))
        self._base_z: dict = {}

    def base(self, task_id: str) -> dict:
        with self._lock:
            b = self._bases.get(task_id)
        if b is None:
            b = load_base(self.base_cache, task_id)
            with self._lock:
                b = self._bases.setdefault(task_id, b)
        return b

    def __call__(self, items) -> list:
        t0 = time.time()
        out = [None] * len(items)
        with ThreadPoolExecutor(max_workers=REWARD_THREADS, thread_name_prefix="plan") as ex:
            futs = {ex.submit(self._one, it): k for k, it in enumerate(items)}
            for f, k in futs.items():
                try:
                    out[k] = f.result()
                except Exception as e:
                    out[k] = RewardResult(FAILURE_REWARD, {"error": f"{type(e).__name__}: {e}"[:500]},
                                          "failed")
        self.stats = {"n_items": len(items),
                      "n_tasks": len({it["item"]["task"] for it in items}),
                      "n_failed_rewards": sum(1 for r in out if r.status != "ok"),
                      "n_parse_failed": sum(1 for r in out if r.status == "parse_failed"),
                      "t_reward_wall": time.time() - t0}
        return out

    def check_tasks(self, tasks) -> None:
        bt = self.bands.get("tasks") or {}
        missing = sorted(t for t in set(tasks) if t not in bt)
        partial = sorted(t for t in set(tasks) if t in bt and not bt[t].get("complete"))
        if missing or partial:
            raise ValueError(f"planning_exit: {len(missing)} tasks have no train band and "
                             f"{len(partial)} an incomplete one in {self.bands.get('_path')} "
                             f"(missing {missing[:5]}, incomplete {partial[:5]}); rerun "
                             f"precompute_train_bands.py on this task list")

    def probe_encoder(self) -> str:
        mid = "injected"
        if isinstance(self.embed, self.PE.NomicHTTPEmbed):
            mid = str(self.embed.health().get("model") or "")
            if "nomic" not in mid.lower():
                raise ValueError(f"planning_exit: the embedding service serves {mid!r}; "
                                 f"outline_title_f1_emb is defined on nomic-embed-text-v1")
        v = np.asarray(self.embed._raw(["planning exit encoder probe"]))
        dims = {int(b["emb_dim"]) for b in (self.bands.get("tasks") or {}).values() if b.get("emb_dim")}
        if dims and int(v.shape[-1]) not in dims:
            raise ValueError(f"planning_exit: encoder dim {v.shape[-1]} != bands emb_dim {dims}")
        return mid

    def score_plan(self, plan: dict, task_id: str) -> tuple:
        PE = self.PE
        b = self.bands["tasks"][task_id]
        sys_top, n_sys = PE.sys_top_titles(PE.nodes_from_plan(plan), self.cc["cco"])
        raw = PE.readouts(sys_top, n_sys, b["hum_top"], int(b["n_hum_top"]), self.cc)
        return raw, PE.z_all(raw, b["bands"]), sys_top

    def base_z(self, task_id: str) -> dict:
        with self._lock:
            v = self._base_z.get(task_id)
        if v is None:
            raw, z, _ = self.score_plan(self.base(task_id)["outline_plan"], task_id)
            with self._lock:
                v = self._base_z.setdefault(task_id, {"raw": raw, "z": z})
        return v

    def _one(self, it) -> RewardResult:
        t0 = time.time()
        tid = it["item"]["task"]
        comp = {"task": tid, "prompt_index": it.get("prompt_index"),
                "sample_index": it.get("sample_index"), "reward_form": PLANNING_EXIT}
        try:
            base = self.base(tid)
            bz = self.base_z(tid)
            plan = parse_candidate_plan(
                base["spec"], base["graph"], it["text"],
                finish_reason=("length" if it.get("truncated") else "stop"),
                n_completion_tokens=int(it.get("n_completion_tokens") or 0))
        except Exception as e:
            comp["error"] = f"{type(e).__name__}: {e}"[:500]
            comp["traceback"] = traceback.format_exc()[-1200:]
            return RewardResult(FAILURE_REWARD, comp, "failed")
        comp["n_plan_sections"] = len(plan.get("sections") or [])
        comp["n_claims_unassigned"] = len(plan.get("unassigned_claims") or [])
        comp["plan_coverage"] = (plan.get("coverage_map") or {}).get("coverage")
        if not comp["n_plan_sections"]:
            return RewardResult(FAILURE_REWARD, comp, "parse_failed")
        try:
            raw, z, sys_top = self.score_plan(plan, tid)
        except Exception as e:
            comp["error"] = f"{type(e).__name__}: {e}"[:500]
            comp["traceback"] = traceback.format_exc()[-1200:]
            return RewardResult(FAILURE_REWARD, comp, "failed")
        comp.update({"pe_raw": raw, "pe_z": z, "pe_base_raw": bz["raw"], "pe_base_z": bz["z"],
                     "top_titles": sys_top[:12], "n_sys_top": len(sys_top)})
        if all(v is None for v in raw.values()):
            comp["error"] = "the candidate outline has no titled top-level section"
            return RewardResult(FAILURE_REWARD, comp, "parse_failed")
        rr = self.RR.compute(z, bz["z"], self.targets, window=self.PE.WINDOW)
        if rr is None:
            comp["error"] = "no planning-exit readout is scored on both sides"
            return RewardResult(FAILURE_REWARD, comp, "failed")
        comp.update({"R": rr["R"], "parts": rr["parts"], "n_scored": rr["n_scored"],
                     "targets_sha256_16": rr["targets_sha256_16"],
                     "bands_sha256_16": self.bands.get("_sha256_16"),
                     "reward": rr["reward01"], "t_local": time.time() - t0})
        for r in self.PE.READOUTS:
            comp[r] = raw.get(r)
            comp[f"{r}_z"] = z.get(r)
        return RewardResult(float(rr["reward01"]), comp, "ok")


def classify_failure(components):
    st = (components or {}).get("status")
    if st is None or st == "ok":
        return None
    return "candidate" if st == "parse_failed" else "infra"


def _configure_reward_batch(self, base_cache: str):
    if not base_cache:
        raise ValueError("planning needs the base cache (--base_cache)")
    rc = PlanningExitReward(base_cache)
    rc.check_tasks([it["task"] for it in self.train_items])
    emb_id = rc.probe_encoder()
    self.planning_reward = rc
    tg = rc.targets
    print(f"[planning] planning-exit reward armed: base cache {base_cache}; readouts "
          f"{list(rc.PE.READOUTS)} (title_fuzzy={rc.cc['TITLE_FUZZ']} sec_cos={rc.cc['SEC_COS']}; "
          f"encoder {emb_id}); bands {rc.bands.get('_path')} sha256_16 {rc.bands.get('_sha256_16')}; "
          f"targets {tg.get('_path')} sha256_16 {tg.get('_sha256_16')} weights "
          f"{ {r: v.get('w') for r, v in tg['window_readouts'][rc.PE.WINDOW].items()} }; "
          f"threads {REWARD_THREADS}", flush=True)


def _reward_batch(self, items) -> list:
    out = self.planning_reward(items)
    st = dict(self.planning_reward.stats)

    def _mean(key):
        v = [float(r.components[key]) for r in out if r.components.get(key) is not None]
        return (sum(v) / len(v)) if v else None
    st.update({"plan_coverage_mean": _mean("plan_coverage"),
               "n_claims_unassigned_mean": _mean("n_claims_unassigned"),
               **{f"{r}_mean": _mean(r) for r in self.planning_reward.PE.READOUTS},
               **{f"{r}_z_mean": _mean(f"{r}_z") for r in self.planning_reward.PE.READOUTS},
               "pe_R_mean": _mean("R")})
    self.last_reward_batch_stats = st
    return out


PlanningTask.configure_reward_batch = _configure_reward_batch
PlanningTask.reward_batch = _reward_batch
PlanningTask.classify_failure = staticmethod(classify_failure)
PlanningTask.last_reward_batch_stats = {}
PlanningTask.planning_reward = None


def register():
    from kvskill.rl import tasks as T
    T.TASKS["planning"] = PlanningTask
    return T
