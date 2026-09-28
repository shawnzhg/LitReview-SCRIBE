"""Loads the score table into one Data object with the draft, planning-exit and planning-axis rows,
the recorded failures and the peer-band calibration of each view."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import roster as R

DOWNSTREAM = {"retrieval": ["retrieval", "synthesis", "planning", "writing", "draft", "draft_trunc", "system", "system_trunc"],
              "synthesis": ["synthesis", "planning", "writing", "draft", "draft_trunc", "system", "system_trunc"],
              "planning": ["planning", "writing", "draft", "draft_trunc", "system", "system_trunc"],
              "writing": ["writing", "draft", "draft_trunc", "system", "system_trunc"],
              "draft": ["draft", "draft_trunc", "system", "system_trunc"],
              "system": ["system", "system_trunc"]}
FINAL_WINDOWS = ("writing", "draft", "draft_trunc", "system", "system_trunc")
TRUNC_WINDOWS = ("system_trunc", "draft_trunc")
DRAFT_SUFFIX = ".draft"


def _ccbench():
    C.bootstrap_env()
    from ccbench.ingest import gold
    from ccbench.rankability import normalise as NRM
    from ccbench.rankability.weighted import group_of
    from ccbench.readouts import window_views as WV
    return NRM, gold, WV, group_of


HARNESS_FAILURE_STATUS = ("proxy_fail", "no_input", "no_lane", "guard_fail")
MISSING_REASONS = ("task_dir_missing", "no_input", "adapter_error")


def run_dir(arm: str, task: str, fixed_input: bool) -> Path | None:
    for root in ((C.RUN_ROOT_FIXED_INPUT,) if fixed_input else C.RUN_ROOTS_SAME_POOL):
        d = root / arm / task
        if d.is_dir():
            return d
    return None


def status_files(arm: str, fixed_input: bool) -> list[Path]:
    roots = (C.RUN_ROOT_FIXED_INPUT,) if fixed_input else C.RUN_ROOTS_SAME_POOL
    return [r / arm / "_arm/task_status.jsonl" for r in roots]


def _system_acted(arm: str, task: str, fixed_input: bool) -> bool | None:
    d = run_dir(arm, task, fixed_input)
    if d is None:
        return None
    f = d / "_calls.jsonl"
    try:
        return f.exists() and f.stat().st_size > 0
    except OSError:
        return None


def _runner_status(files: list) -> dict:
    last: dict = {}
    for p in reversed([x for x in files if x.exists()]):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = d.get("task") or d.get("task_id")
                if t:
                    last[t] = str(d.get("status") or d.get("event") or "")
    return last


def rollout_failures(systems) -> list[tuple[str, str, str]]:
    root = C.CCB_OUT / "rollouts"
    out = []
    names = sorted({k for s in systems for k in (s, s + DRAFT_SUFFIX)})
    for d in [root / n for n in names if (root / n).is_dir()]:
        for fp in sorted(d.glob("*.json")):
            try:
                with open(fp) as f:
                    head = f.read(4000)
            except OSError:
                continue
            m = re.search(r'"status":\s*"([a-z_]+)"', head)
            if m is None:
                try:
                    rec = json.load(open(fp))
                except (OSError, json.JSONDecodeError):
                    continue
                st, reason = str(rec.get("status", "ok")), str(rec.get("bot_reason") or "")
            else:
                st = m.group(1)
                rm = re.search(r'"bot_reason":\s*"([^"]*)"', head)
                reason = rm.group(1) if rm else ""
            if st != "ok":
                out.append((d.name, fp.stem, reason))
    return out


def load_bot_map(scores: pd.DataFrame) -> dict:
    out: dict = {}
    dropped: list = []

    def add(sys_key, window, task):
        out.setdefault((sys_key, window), set()).add(task)

    baselines = {k for k, v in R.SYSTEMS.items() if v["role"] == "baseline"}
    status = {k: _runner_status(status_files(k.replace(".ref", ""), k.endswith(".ref"))) for k in baselines}
    for k, st in status.items():
        for t, s_ in st.items():
            if s_.lower().startswith("fail"):
                for w in FINAL_WINDOWS:
                    add(k, w, t)
    for system, task, reason in rollout_failures(R.SYSTEMS):
        if system.endswith(DRAFT_SUFFIX):
            if not reason.startswith("draft_unavailable"):
                for w in ("draft", "draft_trunc"):
                    add(system[: -len(DRAFT_SUFFIX)], w, task)
            continue
        if system in baselines:
            base, fixed = system.replace(".ref", ""), system.endswith(".ref")
            st = status[system].get(task)
            if st in HARNESS_FAILURE_STATUS or any(reason.startswith(x) for x in MISSING_REASONS):
                dropped.append((system, task, f"status={st or '-'} reason={reason or '-'}"))
                continue
            if _system_acted(base, task, fixed) is False:
                dropped.append((system, task, f"{st or reason}:no_model_call"))
                continue
            for w in FINAL_WINDOWS:
                add(system, w, task)
        elif reason.startswith("window_failed:"):
            first = reason.split(":", 1)[1].split(",")[0].strip()
            for w in DOWNSTREAM.get(first, list(FINAL_WINDOWS)):
                add(system, w, task)
        elif any(reason.startswith(x) for x in MISSING_REASONS):
            dropped.append((system, task, f"status=- reason={reason}"))
        else:
            for w in FINAL_WINDOWS:
                add(system, w, task)
    if "note" in scores.columns:
        for r in scores[scores.note == "no_exit_artifact"].itertuples():
            for w in DOWNSTREAM.get(r.window, [r.window]):
                add(r.system, w, r.task)
    out["_dropped_not_system_failure"] = dropped
    return out


def load_outline_window(stage: str = "final") -> pd.DataFrame:
    p = C.SOURCES["outline_arms"]
    rows = []
    if not p.exists():
        return pd.DataFrame(columns=["system", "task", "window", "readout", "value", "source"])
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            if d.get("stage") != stage or not d.get("admitted", True) or d.get("control", "none") != "none":
                continue
            key = d["key"]
            if key.endswith(".self"):
                key = key[:-5]
            ours_arm = R.SYSTEMS.get(key, {}).get("role") == "ours"
            for r, v in d["readouts"].items():
                if ours_arm and r.startswith("org_"):
                    continue
                val = v.get("value") if isinstance(v, dict) else v
                rows.append({"system": key, "task": d["task"], "window": "planning", "readout": r,
                             "value": np.nan if val is None else float(val), "source": "outline_window",
                             "n_sections": d.get("n_sections")})
    return pd.DataFrame(rows)


class Data:
    def __init__(self):
        NRM, gold, WV, group_of = _ccbench()
        self.NRM, self.gold, self.WV, self.group_of = NRM, gold, WV, group_of
        self.tasks = sorted(json.load(open(C.SOURCES["campaign50"]))["tasks"])
        self.clusters = gold.campaign50_clusters(C.REGISTERED["cluster_level"])
        table = pd.read_parquet(C.SOURCES["window_scores"])
        self.human = table[table.panel == "H"].copy()
        self.scores = table[table.panel != "H"].copy()
        self.scores["source"] = "E13"
        self._ingest_draft_window()
        self.stage_of = self.scores.drop_duplicates("readout").set_index("readout").stage.to_dict()
        self.direction_of = self.scores.drop_duplicates("readout").set_index("readout").direction.to_dict()
        self.cal_e10 = pd.read_csv(C.SOURCES["calibration_e10"])
        self._cal: dict = {}
        mem = json.load(open(C.SOURCES["membership"]))
        self.non_discriminating = set(mem["non_discriminating"])
        self.humans_incomparable = set(mem["humans_incomparable"])
        self.eps_fair = {r["readout"]: r["eps_fair"] for r in mem["discriminability_table"]}
        self.no_topical_signal = set(mem["planning_no_topical_signal"])
        self.saturated = set(mem["planning_saturated"])
        self.bot = load_bot_map(self.scores)
        self.bot_dropped = self.bot.pop("_dropped_not_system_failure", [])
        self.radii = pd.read_parquet(C.SOURCES["radii"]) if C.SOURCES["radii"].exists() else None
        ow = load_outline_window()
        if len(ow):
            calibrated = ow[ow.readout.str.startswith("org_")].copy()
            calibrated["panel"] = "A"
            calibrated["mode"] = "campaign"
            calibrated["stage"] = calibrated.readout.map(self.stage_of).fillna("organisation")
            calibrated["direction"] = calibrated.readout.map(self.direction_of).fillna("quality")
            calibrated["family"] = "outline"
            calibrated["is_bot"] = calibrated.value.isna()
            calibrated["note"] = ""
            calibrated["source_review"] = None
            calibrated["n_units"] = np.nan
            calibrated["stratum"] = "all"
            calibrated["criterion"] = "organisation"
            self.scores = pd.concat([self.scores, calibrated[self.scores.columns.intersection(calibrated.columns)]], ignore_index=True)
            ow = ow[~ow.readout.str.startswith("org_")]
        self.extra = ow.reset_index(drop=True)
        self._ingest_outline_axis()
        self.systems_present = sorted(set(self.scores.system) | set(self.extra.system))

    def _ingest_draft_window(self) -> None:
        is_draft = self.scores.system.str.endswith(DRAFT_SUFFIX)
        d = self.scores[is_draft & self.scores.window.isin(["system", "system_trunc"])].copy()
        d["system"] = d.system.str[: -len(DRAFT_SUFFIX)]
        d["window"] = d.window.map({"system": "draft", "system_trunc": "draft_trunc"})
        d["source"] = "E13_draft"
        self.draft_systems = sorted(set(d.system))
        self.scores = self.scores[~is_draft]
        dup = [k for k, v in R.SYSTEMS.items() if v.get("draft_is_final") and k not in self.draft_systems]
        f = self.scores[self.scores.system.isin(dup) & self.scores.window.isin(["system", "system_trunc"])].copy()
        f["window"] = f.window.map({"system": "draft", "system_trunc": "draft_trunc"})
        f["source"] = "final_as_draft"
        self.scores = pd.concat([self.scores, d, f], ignore_index=True)

    def _ingest_outline_axis(self) -> None:
        rp, hp = C.SOURCES["outline_axis_rows"], C.SOURCES["outline_axis_human"]
        self.outline_axis_human = pd.DataFrame(columns=["task", "readout", "value", "source_review"])
        if not (rp.exists() and hp.exists()):
            self.outline_axis_loaded = False
            return
        rows = [json.loads(l) for l in open(rp)]
        hum = [json.loads(l) for l in open(hp)]
        reads = C.REGISTERED["planning_axis_readouts"]
        recs = []
        for d in rows:
            if d["system"] not in R.SYSTEMS:
                continue
            windows = ["planning"] if d["obs"] == "planning" else ["system", "system_trunc"]
            if d["obs"] != "planning" and R.SYSTEMS[d["system"]].get("draft_is_final"):
                windows += ["draft", "draft_trunc"]
            for w in windows:
                for r in reads:
                    v = d.get(r)
                    recs.append({"system": d["system"], "task": d["task"], "panel": "A" if R.SYSTEMS[d["system"]]["role"] == "baseline" else "B",
                                 "mode": "campaign", "readout": r, "stage": "organisation", "family": "outline", "stratum": "all",
                                 "criterion": "organisation", "direction": "quality", "value": np.nan if v is None else float(v),
                                 "is_bot": v is None, "n_units": np.nan, "note": "no_outline" if d.get("no_outline") else "",
                                 "source_review": None, "window": w, "source": "outline_axis"})
        if recs:
            df = pd.DataFrame(recs)
            self.scores = pd.concat([self.scores, df[self.scores.columns.intersection(df.columns)]], ignore_index=True)
            for r in reads:
                self.stage_of[r] = "organisation"
                self.direction_of[r] = "quality"
        H = pd.DataFrame([{"task": d["task"], "readout": r, "value": d.get(r), "source_review": d["source_review"]}
                          for d in hum for r in reads if d.get(r) is not None])
        if len(H):
            self.outline_axis_human = H
            cal = self.NRM.calibration(H)
            self.cal_e10 = pd.concat([self.cal_e10[~self.cal_e10.readout.isin(reads)], cal[self.cal_e10.columns.intersection(cal.columns)]], ignore_index=True)
        self.outline_axis_loaded = True

    def calibration(self, window: str) -> pd.DataFrame:
        if window not in TRUNC_WINDOWS:
            return self.cal_e10
        if "trunc" not in self._cal:
            H = self.human[self.human.window == "system_trunc"]
            if H.empty:
                raise ValueError("the score table holds no peer rows at window system_trunc; "
                                 "run ccbench.score_windows --peers --truncate")
            cal = self.NRM.calibration(H[["task", "readout", "value", "source_review"]])
            if len(self.outline_axis_human):
                reads = C.REGISTERED["planning_axis_readouts"]
                cal = pd.concat([cal[~cal.readout.isin(reads)], self.NRM.calibration(self.outline_axis_human)], ignore_index=True)
            self._cal["trunc"] = cal
        return self._cal["trunc"]

    def z_matrix(self, window: str, systems: list[str]) -> pd.DataFrame:
        t = self.scores[(self.scores.window == window) & self.scores.system.isin(systems)]
        if t.empty:
            return pd.DataFrame()
        z = self.NRM.normalise(t, self.calibration(window))
        return z.pivot_table(index=["system", "task"], columns="readout", values="z", aggfunc="mean")

    def raw_matrix(self, window: str, systems: list[str]) -> pd.DataFrame:
        t = self.extra[(self.extra.window == window) & self.extra.system.isin(systems)]
        if t.empty:
            return pd.DataFrame()
        return t.pivot_table(index=["system", "task"], columns="readout", values="value", aggfunc="mean")

    def readouts_at(self, window: str) -> list[str]:
        w = "system" if window in ("system_trunc", "draft", "draft_trunc") else window
        present = set(self.scores[self.scores.window == window].readout.unique())
        return sorted(r for r in present if self.WV.readouts_observable_at(w, r))

    def extra_readouts_at(self, window: str) -> list[str]:
        return sorted(self.extra[self.extra.window == window].readout.unique()) if len(self.extra) else []

    def n_finished(self, system: str, window: str) -> int:
        w = "system" if window == "system_trunc" else window
        if window in ("draft", "draft_trunc"):
            w = "draft"
        if not hasattr(self, "_fin"):
            c = self.scores[self.scores.readout == "completion"]
            self._fin = c[c.value >= 0.999].groupby(["system", "window"]).task.nunique().to_dict()
            self._fin_any = self.scores[self.scores.value.notna()].groupby(["system", "window"]).task.nunique().to_dict()
        if (system, w) in self._fin:
            return int(self._fin[(system, w)])
        return int(self._fin_any.get((system, w), 0))

    def flags(self, readout: str, window: str) -> list[str]:
        f = []
        if readout in C.REGISTERED["channel_defects"]:
            f.append("channel_defect")
        if readout in self.non_discriminating:
            f.append("non_discriminating")
        if readout in self.humans_incomparable:
            f.append("humans_incomparable")
        if readout in C.REGISTERED["guardrails"]:
            f.append("guardrail")
        if readout in self.no_topical_signal:
            f.append("no_topical_signal")
        if readout in self.saturated:
            f.append("saturated")
        return f

    def group(self, readout: str) -> str:
        if readout.startswith("outline_"):
            return "organisation"
        st = self.stage_of.get(readout, "")
        try:
            return self.group_of(readout, st)
        except Exception:
            return st or "-"
