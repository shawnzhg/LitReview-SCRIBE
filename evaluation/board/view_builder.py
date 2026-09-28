"""Builds a private merged view for build_board.py --extra: the shared score tables plus the isolated
rows of one external agent or one of our same-pool keys. Usage: python view_builder.py view --agent
<agent> --iso KEY=DIR ... | native --key <key> --iso <dir>."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from view_agents import AGENTS, EXT_KEYS

EVAL = HERE.parent
SHARED_OUT = Path(os.environ.get("CCBENCH_ROOT") or "/nonexistent/CCBENCH_ROOT") / "out"
RUN_VIEW = HERE / "run_view.py"
CODE_FILES = ("view_builder.py", "view_agents.py", "run_view.py")
PY = os.environ.get("PY") or sys.executable
VIEWS_ROOT = Path(os.environ.get("SCRIBE_WORK_ROOT") or "/nonexistent/SCRIBE_WORK_ROOT") / "board" / "views"
VIEW_MARKER = ".merged_view"
NEW_KEYS: dict[str, dict] = {}
SNAP_PORT = ["E13/window_scores.parquet", "E13/radii.parquet", "E13/distances.parquet", "E1/conformance.csv", "E1/summary.csv"]
SNAP_WB = {"outline_axis_rows.jsonl": Path(os.environ.get("WINDOWBENCH_OUT") or "/nonexistent/WINDOWBENCH_OUT") / "outline_axis/rows.jsonl",
           "roster_extra.json": Path(os.environ.get("WINDOWBENCH_ROSTER_EXTRA") or "/nonexistent/WINDOWBENCH_ROSTER_EXTRA")}
INJECT_FIELDS = ("template", "label", "interface", "temperature", "notes", "vendor_model", "model_group", "backbone")
REF_OF = {"fixed_input": "autosurvey.ref", "same_pool": "autosurvey"}
ISO_LOG = "score_iso.log"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fail(msg: str):
    raise SystemExit(f"VIEW FAILED: {msg}")


def sha256(p: Path, n: int = 64) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_record(base: Path = HERE) -> dict:
    files = {n: {"md5": md5(base / n), "sha256_16": sha256(base / n, 16)} for n in CODE_FILES if (base / n).exists()}
    return {"dir": str(base), "files": files}


def is_view(vdir: Path) -> bool:
    return (Path(vdir) / VIEW_MARKER).exists()


def bundle_entry_root() -> Path:
    if str(EVAL) not in sys.path:
        sys.path.insert(0, str(EVAL))
    from windowbench import config as WC
    return WC.SOURCES["bundle_entry_root"]


def shared_state() -> dict:
    st = {}
    for rel in SNAP_PORT + ["E10/unit_scores.parquet"]:
        p = SHARED_OUT / rel
        st[f"out/{rel}"] = md5(p) if p.exists() else None
    for k, p in SNAP_WB.items():
        st[k] = md5(p) if p.exists() else None
    emb = sorted(os.listdir(SHARED_OUT / "_embed_cache"))
    st["embed_cache_files"] = len(emb)
    st["embed_cache_listing_sha"] = hashlib.sha256("\n".join(emb).encode()).hexdigest()[:16]
    st["rollout_keys"] = sorted(os.listdir(SHARED_OUT / "rollouts"))
    lv = bundle_entry_root()
    st["bundle_entry_listing_sha"] = hashlib.sha256("\n".join(sorted(os.listdir(lv))).encode()).hexdigest()[:16] if lv.is_dir() else None
    return st


def diff_state(a: dict, b: dict) -> list[str]:
    return [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]


def _scoring_log_problems(logp: Path) -> list[str]:
    txt = logp.read_text(errors="replace") if logp.exists() else ""
    why = []
    if not re.search(r"^### end=.* rc=0", txt, re.M):
        why.append(f"log {logp.name} has no '### end=... rc=0'")
    if "shared tables and the shared embed cache unchanged" not in txt:
        why.append("log lacks the shared-tables-unchanged line")
    return why


def find_iso(key: str, level: int, override: str | None = None) -> dict:
    want_budget = NEW_KEYS[key]["budget"]
    cands = [Path(override)] if override else []
    tried = []
    for d in cands:
        ws = d / "out/E13/window_scores.parquet"
        if not ws.exists():
            tried.append(f"{d.name}: no out/E13/window_scores.parquet")
            continue
        job = d.name.rsplit("_", 1)[-1]
        logp = d / ISO_LOG
        why = _scoring_log_problems(logp)
        try:
            gate = json.load(open(d / "level_gate.json"))
        except Exception as e:
            gate = {}
            why.append(f"no level_gate.json ({type(e).__name__})")
        rec = gate.get("record") or {}
        if gate and (gate.get("problems") or int(rec.get("level", -1)) != int(level)):
            why.append(f"level gate problems={gate.get('problems')} level={rec.get('level')} (want {level})")
        if gate and (gate.get("retrieval_budget") != want_budget or rec.get("retrieval_budget") != want_budget):
            why.append(f"retrieval budget gate={gate.get('retrieval_budget')!r} record={rec.get('retrieval_budget')!r} (want {want_budget!r})")
        if why:
            tried.append(f"{d.name}: " + "; ".join(why))
            continue
        return {"key": key, "dir": str(d), "job": job, "log": str(logp), "window_scores": str(ws),
                "window_scores_sha256": sha256(ws), "conformance": str(d / "out/E1/conformance.csv"),
                "conformance_sha256": sha256(d / "out/E1/conformance.csv"), "rollouts": str(d / "out/rollouts" / key),
                "level_gate": {"record": gate.get("record"), "n_units_linked": gate.get("n_units_linked"),
                               "retrieval_budget": gate.get("retrieval_budget"), "snapshot": gate.get("snapshot"),
                               "units": gate.get("units"), "allow_partial": gate.get("allow_partial")}}
    return {"key": key, "dir": None, "tried": tried or ["no isolated scoring directory given (--iso KEY=DIR)"]}


def find_ext(key: str, override: str | None = None) -> dict:
    cfg = EXT_KEYS[key]
    cands = [Path(override)] if override else []
    tried = []
    for d in cands:
        ws = d / "out/E13/window_scores.parquet"
        if not ws.exists():
            tried.append(f"{d.name}: no out/E13/window_scores.parquet")
            continue
        job = d.name.rsplit("_", 1)[-1]
        logp = d / ISO_LOG
        why = _scoring_log_problems(logp)
        n_ro = len(list((d / "out/rollouts" / key).glob("pmcid_*.json")))
        want_n = int(cfg.get("n_tasks", 50))
        if n_ro != want_n:
            why.append(f"{n_ro} rollouts for {key} (want {want_n})")
        if why:
            tried.append(f"{d.name}: " + "; ".join(why))
            continue
        gate = ({**cfg["level_gate"], "n_units_linked": n_ro} if cfg.get("level_gate") else
                {"record": {"level": "n/a (external baseline; no run level)"}, "n_units_linked": n_ro,
                 "retrieval_budget": cfg.get("retrieval_budget", "the pool MCP server, search top-20"), "units": "report",
                 "snapshot": cfg.get("snapshot_note", "the frozen pool"), "allow_partial": False})
        return {"key": key, "dir": str(d), "job": job, "log": str(logp), "source_window_scores": str(ws),
                "source_window_scores_sha256": sha256(ws), "source_conformance": str(d / "out/E1/conformance.csv"),
                "source_conformance_sha256": sha256(d / "out/E1/conformance.csv"),
                "rollouts": str(d / "out/rollouts" / key), "audit": str(d / "audit.json"), "level_gate": gate}
    return {"key": key, "dir": None, "tried": tried or ["no isolated scoring directory given (--iso KEY=DIR)"]}


def split_ext(f: dict, key: str, vdir: Path) -> dict:
    dst = vdir / "iso_split" / key
    dst.mkdir(parents=True)
    t = pd.read_parquet(f["source_window_scores"])
    same_src = {k for k, v in EXT_KEYS.items() if v["agent"] == EXT_KEYS[key]["agent"]}
    extra = sorted(set(t.system.unique()) - same_src)
    if extra:
        fail(f"{f['source_window_scores']} carries systems of another agent {extra}")
    k = t[t.system == key].reset_index(drop=True)
    if k.empty:
        fail(f"no rows of {key} in {f['source_window_scores']}")
    k.to_parquet(dst / "window_scores.parquet", index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(dst / "window_scores.parquet"), k, check_exact=True)
    raw = Path(f["source_conformance"]).read_bytes().split(b"\n")
    head, rows = raw[0], [l for l in raw[1:] if l.strip()]
    if [l for l in rows if not any(l.startswith(c.encode() + b",") for c in same_src)]:
        fail(f"{f['source_conformance']} carries rows of another agent")
    mine = [l for l in rows if l.startswith(key.encode() + b",")]
    (dst / "conformance.csv").write_bytes(head + b"\n" + b"\n".join(mine) + b"\n")
    f = dict(f)
    f.update(window_scores=str(dst / "window_scores.parquet"), window_scores_sha256=sha256(dst / "window_scores.parquet"),
             conformance=str(dst / "conformance.csv"), conformance_sha256=sha256(dst / "conformance.csv"))
    return f


def _times(p: Path) -> dict:
    st = p.stat()
    return {"mtime": st.st_mtime, "ctime": st.st_ctime, "mtime_s": time.ctime(st.st_mtime), "ctime_s": time.ctime(st.st_ctime)}


def snapshot(dst: Path) -> dict:
    srcs = {f"out/{rel}": SHARED_OUT / rel for rel in SNAP_PORT}
    srcs.update(SNAP_WB)
    dst.mkdir(parents=True)
    before = {k: md5(p) for k, p in srcs.items()}
    for k, p in srcs.items():
        shutil.copy2(p, dst / k.replace("/", "__"))
    copies = {k: md5(dst / k.replace("/", "__")) for k in srcs}
    if before != copies or before != {k: md5(p) for k, p in srcs.items()}:
        fail("a shared table changed while the snapshot was taken")
    return {k: {"src": str(p), "copy": str(dst / k.replace("/", "__")), "md5": before[k],
                "sha256": sha256(dst / k.replace("/", "__")), **{f"src_{a}": b for a, b in _times(p).items()}}
            for k, p in srcs.items()}


def build_out(vdir: Path, files: dict[str, Path], rollouts_extra: dict[str, Path], external: bool) -> None:
    out = vdir / "out"
    out.mkdir()
    for e in SHARED_OUT.iterdir():
        if e.name not in {"E1", "E13", "rollouts"}:
            (out / e.name).symlink_to(e)
    for sub in ("E1", "E13"):
        (out / sub).mkdir()
        mine = {Path(r).name: src for r, src in files.items() if r.startswith(sub + "/")}
        for e in (SHARED_OUT / sub).iterdir():
            if e.name not in mine:
                (out / sub / e.name).symlink_to(e)
        for name, src in mine.items():
            shutil.copy2(src, out / sub / name)
    ro = out / "rollouts"
    ro.mkdir()
    for e in (SHARED_OUT / "rollouts").iterdir():
        (ro / e.name).symlink_to(e)
    for key, src in rollouts_extra.items():
        if (ro / key).exists():
            fail(f"rollouts/{key} already exists in the shared rollouts")
        (ro / key).symlink_to(src)
    if external:
        (vdir / VIEW_MARKER).write_text("private merged view with external-agent rows; never the shared tables\n")
    else:
        (vdir / VIEW_MARKER).write_text("private merged view; never the shared tables\n")


def base_env() -> dict:
    env = dict(os.environ)
    keep = [p for p in env.get("PYTHONPATH", "").split(":") if p and not (Path(p) / "ccbench").is_dir()]
    env.update({"CCBENCH_OUT": str(SHARED_OUT), "PYTHONPATH": ":".join([str(EVAL)] + keep),
                "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONNOUSERSITE": "1",
                "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONWARNINGS": "ignore"})
    for k in ("WINDOWBENCH_ROSTER_EXTRA", "WINDOWBENCH_OUTLINE_AXIS_ROWS", "CCBENCH_DEVICE",
              "WINDOWBENCH_ROSTER_INJECT"):
        env.pop(k, None)
    return env


def view_env(vdir: Path) -> dict:
    env = base_env()
    env.update({"CCBENCH_OUT": str(vdir / "out"), "WINDOWBENCH_ROSTER_EXTRA": str(vdir / "roster_extra.json"),
                "WINDOWBENCH_OUTLINE_AXIS_ROWS": str(vdir / "outline_axis/rows.jsonl")})
    if (vdir / "roster_inject.json").exists():
        env["WINDOWBENCH_ROSTER_INJECT"] = str(vdir / "roster_inject.json")
    return env


def declarations(vdir: Path, keys: list[str]) -> dict:
    r = subprocess.run([PY, str(RUN_VIEW), "declarations", *keys], env=view_env(vdir), capture_output=True, text=True,
                       cwd=str(EVAL))
    if r.returncode:
        fail(f"the view roster does not load: {r.stderr[-1500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def entry_problems(d: dict, base: dict, base_key: str) -> list[str]:
    return [f"entry {w} {d['entries'][w]!r} != {base_key}'s {base['entries'][w]!r}"
            for w in ("system", "system_trunc", "retrieval") if d["entries"][w] != base["entries"][w]]


def run(cmd: list[str], env: dict, logp: Path, cwd: Path = EVAL) -> int:
    logp.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(logp, "w") as f:
        f.write(f"### cmd: {' '.join(cmd)}\n### CCBENCH_OUT={env.get('CCBENCH_OUT')} WINDOWBENCH_ROSTER_EXTRA={env.get('WINDOWBENCH_ROSTER_EXTRA')} "
                f"WINDOWBENCH_OUTLINE_AXIS_ROWS={env.get('WINDOWBENCH_OUTLINE_AXIS_ROWS')}\n")
        f.flush()
        rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd))
        f.write(f"### rc={rc} seconds={time.time() - t0:.1f}\n")
    return rc


def check_external_declarations(vdir: Path, ext: list[str]) -> dict:
    want = sorted({x for k in ext for x in (k, REF_OF[EXT_KEYS[k]["dataset"]])})
    d = declarations(vdir, want)
    for k in ext:
        b = REF_OF[EXT_KEYS[k]["dataset"]]
        e = d[k]["entries"]
        probs = entry_problems(d[k], d[b], b)
        if d[k]["role"] != "baseline" or d[k]["model_group"] != EXT_KEYS[k].get("model_group", "gpt56") \
                or d[k]["provenance"] != d[b]["provenance"]:
            probs.append(f"role/model_group/provenance {d[k]['role']}/{d[k]['model_group']}/{d[k]['provenance']}")
        if EXT_KEYS[k].get("backbone") and d[k]["backbone"] != EXT_KEYS[k]["backbone"]:
            probs.append(f"backbone {d[k]['backbone']!r}")
        if d[k]["in_datasets"]:
            probs.append(f"declared on dataset panels {d[k]['in_datasets']}")
        if not d[k]["draft_is_final"] or e["draft"] != e["system"] or e["planning"] is not None:
            probs.append("draft/planning declaration")
        if probs:
            fail(f"{k} is not declared like {b}: {probs}")
    return {k: d[k] for k in want}


def planning_rows(vdir: Path, k: str) -> dict:
    cmd = ([PY, str(RUN_VIEW), "windowbench.outline_axis_add"] if k in EXT_KEYS else [PY, "-m", "windowbench.outline_axis_add"]) + [
           "--systems", k, "--canon", str(vdir / "outline_axis/rows.jsonl"), "--scratch", str(vdir / f"outline_axis/_add_{k}")]
    env = view_env(vdir)
    env["CCBENCH_DEVICE"] = "cuda"
    logp = vdir / f"logs/outline_axis_add_{k}.log"
    log(f"planning axis rows for {k}")
    rc = run(cmd, env, logp)
    if rc != 0:
        (vdir / "FAILED_VIEW").write_text(f"{k}: planning-axis rows rc={rc}: {logp}\n")
        fail(f"{k}: planning-axis rows rc={rc}; see {logp}")
    return {"rc": rc, "log": str(logp)}


def declare_native(key: str, iso_dir: str) -> None:
    gate = json.load(open(Path(iso_dir) / "level_gate.json"))
    NEW_KEYS[key] = dict(level=int(gate["record"]["level"]), budget=gate.get("retrieval_budget"),
                         label=key)


def build_view(keys: list[str], iso_override: dict[str, str], views_dir: Path, suffix: str = "") -> Path:
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    vdir = Path(views_dir) / (stamp + suffix)
    vdir.mkdir(parents=True)
    log(f"view {vdir}")
    code = code_record()
    state0 = shared_state()
    snap = snapshot(vdir / "snapshot")
    sh = pd.read_parquet(vdir / "snapshot/out__E13__window_scores.parquet")
    shared_systems = set(sh.system.unique())
    iso, avail, unavailable = {}, [], {}
    for k in keys:
        if k in shared_systems:
            unavailable[k] = "already in the shared tables: this builder appends isolated rows only"
            continue
        if k in EXT_KEYS:
            f = find_ext(k, iso_override.get(k))
            if f.get("dir") is not None:
                f = split_ext(f, k, vdir)
        else:
            f = find_iso(k, NEW_KEYS[k]["level"], iso_override.get(k))
        if f.get("dir") is None:
            unavailable[k] = f["tried"]
            continue
        iso[k] = f
        avail.append(k)
    log(f"isolated rows available: {avail}; unavailable: {list(unavailable)}")
    parts = [sh]
    for k in avail:
        t = pd.read_parquet(iso[k]["window_scores"])
        if set(t.system.unique()) != {k}:
            fail(f"{iso[k]['window_scores']} carries systems {sorted(t.system.unique())}, not only {k}")
        if list(t.columns) != list(sh.columns) or not (t.dtypes == sh.dtypes).all():
            fail(f"schema of {iso[k]['window_scores']} differs from the shared table")
        iso[k]["tasks_by_window"] = t.groupby("window").task.nunique().to_dict()
        iso[k]["rows"] = int(len(t))
        parts.append(t)
    tabs = vdir / "tables"
    tabs.mkdir()
    merged = pd.concat(parts, ignore_index=True)
    merged.to_parquet(tabs / "window_scores.parquet")
    back = pd.read_parquet(tabs / "window_scores.parquet")
    pd.testing.assert_frame_equal(back.iloc[: len(sh)].reset_index(drop=True), sh.reset_index(drop=True), check_exact=True)
    off = len(sh)
    for t in parts[1:]:
        pd.testing.assert_frame_equal(back.iloc[off: off + len(t)].reset_index(drop=True), t.reset_index(drop=True), check_exact=True)
        off += len(t)
    if back.duplicated(["system", "task", "readout", "window"]).any():
        fail("the merged window_scores has duplicate (system, task, readout, window) keys")
    del back, merged
    conf_src = (vdir / "snapshot/out__E1__conformance.csv").read_bytes()
    if not conf_src.endswith(b"\n"):
        conf_src += b"\n"
    head = conf_src.split(b"\n", 1)[0]
    body = conf_src
    for k in avail:
        lines = [l.rstrip(b"\r") for l in Path(iso[k]["conformance"]).read_bytes().split(b"\n")]
        if lines[0] != head:
            fail(f"conformance header of {k} differs from the shared one")
        rows = [l for l in lines[1:] if l.strip()]
        if not rows or any(not l.startswith(k.encode() + b",") for l in rows):
            fail(f"{iso[k]['conformance']} carries rows that are not {k}'s")
        body += b"\n".join(rows) + b"\n"
        iso[k]["conformance_rows"] = len(rows)
    (tabs / "conformance.csv").write_bytes(body)
    cm = pd.read_csv(tabs / "conformance.csv")
    cs = pd.read_csv(vdir / "snapshot/out__E1__conformance.csv")
    pd.testing.assert_frame_equal(cm.iloc[: len(cs)].reset_index(drop=True), cs, check_exact=True)
    if avail:
        ci = pd.concat([pd.read_csv(iso[k]["conformance"]) for k in avail], ignore_index=True)
        pd.testing.assert_frame_equal(cm.iloc[len(cs):].reset_index(drop=True), ci, check_exact=True, check_dtype=False)
    files = {"E13/window_scores.parquet": tabs / "window_scores.parquet",
             "E13/radii.parquet": vdir / "snapshot/out__E13__radii.parquet",
             "E13/distances.parquet": vdir / "snapshot/out__E13__distances.parquet",
             "E1/conformance.csv": tabs / "conformance.csv",
             "E1/summary.csv": vdir / "snapshot/out__E1__summary.csv"}
    ext = [x for x in avail if x in EXT_KEYS]
    native = [x for x in avail if x in NEW_KEYS]
    build_out(vdir, files, {k: Path(iso[k]["rollouts"]) for k in avail}, external=bool(ext))
    ros = json.load(open(vdir / "snapshot/roster_extra.json"))
    for k in native:
        ros["families"].append({"tag": k, "template": "native", "keys": [k], "labels": [NEW_KEYS[k]["label"]],
                                "notes": f"same-pool run, level {NEW_KEYS[k]['level']}, retrieval budget {NEW_KEYS[k]['budget']}",
                                "level": NEW_KEYS[k]["level"]})
    (vdir / "roster_extra.json").write_text(json.dumps(ros, indent=1))
    if ext:
        inj = {"schema": "roster_inject/1",
               "systems": {k: {f: EXT_KEYS[k][f] for f in INJECT_FIELDS if f in EXT_KEYS[k]} for k in ext}}
        (vdir / "roster_inject.json").write_text(json.dumps(inj, indent=1))
    base = REF_OF["same_pool"]
    decl = declarations(vdir, native + [base])
    for k in native:
        probs = entry_problems(decl[k], decl[base], base)
        if decl[k]["role"] != "ours" or decl[k]["provenance"] != decl[base]["provenance"]:
            probs.append(f"role/provenance {decl[k]['role']}/{decl[k]['provenance']}")
        if probs:
            fail(f"{k} is not declared like {base}: {probs}")
    (vdir / "outline_axis").mkdir(exist_ok=True)
    shutil.copy2(vdir / "snapshot/outline_axis_rows.jsonl", vdir / "outline_axis/rows.jsonl")
    if ext:
        decl.update(check_external_declarations(vdir, ext))
    planning = {k: planning_rows(vdir, k) for k in avail}
    rows = [json.loads(l) for l in open(vdir / "outline_axis/rows.jsonl")]
    snap_rows = open(vdir / "snapshot/outline_axis_rows.jsonl").read().splitlines()
    now_rows = open(vdir / "outline_axis/rows.jsonl").read().splitlines()
    if [l for l in now_rows if json.loads(l)["system"] not in set(avail)] != snap_rows:
        fail("the private planning rows changed rows of systems other than the new keys")
    cov = {}
    for k in avail:
        rr = [d for d in rows if d["system"] == k and d["obs"] == "system"]
        cov[k] = {"n": len(rr), "lex": sum(d.get("outline_title_f1_lex") is not None for d in rr),
                  "emb": sum(d.get("outline_title_f1_emb") is not None for d in rr),
                  "planning_obs_rows": sum(1 for d in rows if d["system"] == k and d["obs"] == "planning")}
    state1 = shared_state()
    man = {"schema": "board_view/1", "view": str(vdir), "created_utc": stamp, "ccbench_out": str(vdir / "out"),
           "roster_inject": str(vdir / "roster_inject.json") if ext else None,
           "roster_extra": str(vdir / "roster_extra.json"), "outline_axis_rows": str(vdir / "outline_axis/rows.jsonl"),
           "outline_axis_rows_sha256_16": sha256(vdir / "outline_axis/rows.jsonl", 16),
           "window_scores_sha256_16": sha256(tabs / "window_scores.parquet", 16),
           "conformance_sha256_16": sha256(tabs / "conformance.csv", 16),
           "code": code,
           "snapshot": snap, "keys_requested": keys, "keys_available": avail, "keys_unavailable": unavailable,
           "isolated": iso, "roster_declarations": decl, "planning_rows": planning,
           "planning_rows_coverage": cov, "shared_state_before": state0, "shared_state_after": state1,
           "shared_state_changed": diff_state(state0, state1)}
    (vdir / "view_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    log(f"view ready: keys {avail}; planning coverage {cov}; shared state changed during build: {man['shared_state_changed'] or 'nothing'}")
    return vdir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("view")
    v.add_argument("--agent", required=True, choices=sorted(AGENTS))
    v.add_argument("--keys", default=None, help="default: every key of the agent")
    v.add_argument("--iso", action="append", help="KEY=DIR: the isolated scoring directory of KEY")
    n = sub.add_parser("native")
    n.add_argument("--key", required=True)
    n.add_argument("--iso", required=True, help="the key's isolated scoring directory (with level_gate.json)")
    a = ap.parse_args(argv)
    if a.cmd == "native":
        declare_native(a.key, a.iso)
        print(build_view([a.key], {a.key: a.iso}, VIEWS_ROOT / "native"))
        return 0
    cfg = AGENTS[a.agent]
    keys = [k for k in (a.keys.split(",") if a.keys else list(cfg["keys"])) if k]
    other = [k for k in keys if k not in cfg["keys"]]
    if other:
        fail(f"{other} are not keys of agent {a.agent}")
    print(build_view(keys, dict(x.split("=", 1) for x in (a.iso or [])), VIEWS_ROOT / a.agent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
