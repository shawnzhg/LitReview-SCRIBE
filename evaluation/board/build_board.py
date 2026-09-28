"""Builds the merged leaderboard (composite, certified pairs and rank intervals per tier and view)
from the shared score tables and the views given with --extra. Usage: python build_board.py build
[--ref S,...] [--self S,...] [--extra NAME=VIEW,...] [--out <dir>]."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PY = os.environ.get("PY") or sys.executable
WORK_ROOT = Path(os.environ.get("SCRIBE_WORK_ROOT") or "/nonexistent/SCRIBE_WORK_ROOT") / "board" / "runs"
HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
VIEWS_DRIVER = EVAL / "board" / "view_builder.py"
VIEW_WRAPPER = EVAL / "board" / "run_view.py"

BOARD_OBS = "system_trunc,system"
FILES_BY_SYSTEM = ("per_task_axis_z.csv", "per_readout_mean_z.csv", "recorded_failures.csv")
VIEW_MARKER = ".merged_view"


def is_view(p: Path) -> bool:
    return (Path(p) / VIEW_MARKER).exists()


def live_out_is_a_view() -> bool:
    return is_view(Path(os.environ.get("CCBENCH_OUT") or "/nonexistent/CCBENCH_OUT").parent)


def die(msg: str):
    raise SystemExit(f"BOARD FAILED: {msg}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_views():
    return load_module("board_view_builder", VIEWS_DRIVER)


def cmd_export(a) -> int:
    from windowbench import config as C
    view = Path(a.view) if a.view else None
    if view:
        if C.CCB_OUT.resolve() != (view / "out").resolve() or not is_view(view):
            die(f"export: windowbench.config.CCB_OUT={C.CCB_OUT} is not the view {view}/out (or no view marker)")
        if os.environ.get("WINDOWBENCH_ROSTER_EXTRA") != str(view / "roster_extra.json"):
            die("export: WINDOWBENCH_ROSTER_EXTRA is not the view's roster copy")
        C.SOURCES["outline_axis_rows"] = view / "outline_axis/rows.jsonl"
    elif live_out_is_a_view():
        die("export: the live export must run on the shared tables (CCBENCH_OUT is a view)")
    from windowbench import axes as A
    from windowbench import merged_board as MB
    from windowbench import roster as R
    from windowbench.load import Data
    inj_rec = None
    if view and (view / "roster_inject.json").exists():
        wmod = load_module("board_run_view", VIEW_WRAPPER)
        inj_rec = {"wrapper": str(VIEW_WRAPPER), "wrapper_md5": md5(VIEW_WRAPPER),
                   **wmod.inject(R, str(view / "roster_inject.json"))}
        print(f"external-agent declarations: {inj_rec}")
    systems = [s for s in a.systems.split(",") if s]
    unk = [s for s in systems if s not in R.FAMILIES and s not in R.SYSTEMS]
    if unk:
        die(f"export: {unk} not declared in this root's roster")
    out = Path(a.out)
    if out.exists():
        die(f"export: {out} exists")
    D = Data()
    MB.write_inputs(D, systems, list(MB.EXPORT_OBS), out)
    bots = [{"obs": o, "system": s, "n_bot": len(A.bot_tasks(D, s, o))} for o in a.obs.split(",") for s in systems]
    pd.DataFrame(bots).sort_values(["obs", "system"]).to_csv(out / "recorded_failures.csv", index=False)
    json.dump({o: {s: sorted(A.bot_tasks(D, s, o)) for s in systems} for o in a.obs.split(",")},
              open(out / "recorded_failures_tasks.json", "w"), indent=1)
    decl = {}
    for s in systems:
        ks = A.members(s)
        S = R.SYSTEMS[ks[0]]
        decl[s] = {"members": ks, "label": A.label(s), "role": S.get("role"), "provenance": S.get("provenance"),
                   "model_group": S.get("model_group"), "backbone": S.get("backbone"),
                   "retrieval_entry": (S.get("entries") or {}).get("retrieval")}
    json.dump({"ccbench_out": str(C.CCB_OUT), "view": str(view) if view else None,
               "roster_extra": os.environ.get("WINDOWBENCH_ROSTER_EXTRA"),
               "outline_axis_rows": str(C.SOURCES["outline_axis_rows"]), "merged_board": str(MB.__file__),
               "merged_board_md5": md5(Path(MB.__file__)), "injection": inj_rec, "systems": decl},
              open(out / "systems.json", "w"), indent=1, default=str)
    print(f"export done: {len(systems)} systems -> {out}")
    return 0


def run_export(V, tag: str, systems: list[str], work: Path, view: Path | None) -> Path:
    out = work / f"export_{tag}"
    env = V.view_env(view) if view else V.base_env()
    if not view and os.environ.get("WINDOWBENCH_ROSTER_EXTRA"):
        env["WINDOWBENCH_ROSTER_EXTRA"] = os.environ["WINDOWBENCH_ROSTER_EXTRA"]
    cmd = [PY, "-u", str(HERE / "build_board.py"), "_export", "--systems", ",".join(systems), "--obs", BOARD_OBS,
           "--out", str(out)] + (["--view", str(view)] if view else [])
    logp = work / f"export_{tag}.log"
    log(f"export {tag}: {len(systems)} systems{' on view ' + str(view) if view else ' on the LIVE tables'} (log {logp})")
    with open(logp, "w") as f:
        rc = subprocess.run(cmd, env=env, cwd=str(EVAL), stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        die(f"export {tag} rc={rc}; tail of {logp}:\n" + "\n".join(logp.read_text().splitlines()[-15:]))
    return out


def view_problems(V, vdir: Path, keys: list[str]) -> list[str]:
    need = ["roster_extra.json", "outline_axis/rows.jsonl", "view_manifest.json", "tables/window_scores.parquet"]
    why = [f"missing {n}" for n in need if not (vdir / n).exists()]
    if not is_view(vdir):
        why.append(f"no view marker ({VIEW_MARKER})")
    if (vdir / "FAILED_VIEW").exists():
        why.append("FAILED_VIEW present")
    if why:
        return why
    man = json.load(open(vdir / "view_manifest.json"))
    miss = [k for k in keys if k not in (man.get("keys_available") or [])]
    if miss:
        return [f"keys {miss} not in the view"]
    for rel, rec in (man.get("snapshot") or {}).items():
        if not Path(rec["src"]).exists() or md5(Path(rec["src"])) != rec["md5"]:
            why.append(f"stale: live {rel} changed since the view's snapshot")
    port_out = vdir / "out"
    for rel, key in (("E13/window_scores.parquet", "window_scores_sha256_16"), ("E1/conformance.csv", "conformance_sha256_16")):
        if V.sha256(port_out / rel, 16) != man.get(key):
            why.append(f"view {rel} sha256 differs from its manifest")
    if V.sha256(vdir / "outline_axis/rows.jsonl", 16) != man.get("outline_axis_rows_sha256_16"):
        why.append("view planning rows sha256 differ from the manifest")
    ros = {f["tag"] for f in json.load(open(vdir / "roster_extra.json")).get("families", [])}
    if (vdir / "roster_inject.json").exists():
        ros |= set(json.load(open(vdir / "roster_inject.json")).get("systems", {}))
    for k in keys:
        iso = (man.get("isolated") or {}).get(k) or {}
        if k not in ros:
            why.append(f"{k} not declared in the view roster")
        if not iso.get("window_scores") or V.sha256(Path(iso["window_scores"])) != iso.get("window_scores_sha256"):
            why.append(f"{k}: isolated window_scores changed since the view was built")
    return why


def read_csv_lines(p: Path) -> tuple[str, list[str]]:
    lines = p.read_text().splitlines()
    return lines[0], [l for l in lines[1:] if l]


def keep_lines(p: Path, keep: set) -> tuple[str, list[str]]:
    hdr, body = read_csv_lines(p)
    ci = hdr.split(",").index("system")
    return hdr, [l for l in body if next(csv.reader([l]))[ci] in keep]


def members_of(decl: dict, systems) -> set:
    return {k for s in systems for k in decl[s]["members"]}


def same_lines(a: Path, b: Path, keep_a: set, keep_b: set | None = None) -> tuple[bool, str]:
    ha, la = keep_lines(a, keep_a)
    hb, lb = keep_lines(b, keep_a if keep_b is None else keep_b)
    ok = ha == hb and sorted(la) == sorted(lb)
    return ok, f"{a.name}: {len(la)} vs {len(lb)} lines, header {'same' if ha == hb else 'DIFFERENT'}, " \
               f"{len(set(la) ^ set(lb))} lines differ"


def identity_checks(base: Path, other: Path, common: list[str], decl: dict, tag: str) -> list[tuple[str, bool, str]]:
    res = []
    ok1, d1 = same_lines(base / "per_task_axis_z.csv", other / "per_task_axis_z.csv", set(common))
    ok2, d2 = same_lines(base / "per_readout_mean_z.csv", other / "per_readout_mean_z.csv", members_of(decl, common))
    ok3, d3 = same_lines(base / "recorded_failures.csv", other / "recorded_failures.csv", set(common))
    ok4 = all((base / f).read_bytes() == (other / f).read_bytes() for f in ("readout_admission.csv", "task_clusters.csv"))
    res.append((f"{tag} export == live export on the {len(common)} systems both carry (line-identical)",
                ok1 and ok2 and ok3 and ok4,
                f"{d1}; {d2}; {d3}; readout_admission+task_clusters byte-identical={ok4}"))
    return res


def sort_key_factory(name: str, obs_order: list[str], axes_order: list[str], hdr: str):
    cols = hdr.split(",")
    oi = {o: i for i, o in enumerate(obs_order)}
    ai = {x: i for i, x in enumerate(axes_order)}
    if name == "per_task_axis_z.csv":
        c = [cols.index(x) for x in ("obs", "axis", "system", "task")]
        return lambda r: (oi[r[c[0]]], ai[r[c[1]]], r[c[2]], r[c[3]])
    if name == "per_readout_mean_z.csv":
        c = [cols.index(x) for x in ("obs", "system", "readout")]
        return lambda r: (oi[r[c[0]]], r[c[1]], r[c[2]])
    c = [cols.index(x) for x in ("obs", "system")]
    return lambda r: (r[c[0]], r[c[1]])


def assemble(parts: list[tuple[Path, set, set]], dst: Path, obs_order: list[str], axes_order: list[str]) -> dict:
    dst.mkdir(parents=True)
    rec = {}
    for name in FILES_BY_SYSTEM:
        hdr0, rows = None, []
        for d, systems, keys in parts:
            if not (d / name).exists():
                if name == "per_task_axis_z.csv":
                    die(f"{d}/{name} missing")
                continue
            hdr, lines = keep_lines(d / name, keys if name == "per_readout_mean_z.csv" else systems)
            if hdr0 not in (None, hdr):
                die(f"{name}: header of {d} differs ({hdr} vs {hdr0})")
            hdr0 = hdr
            rows += lines
        parsed = [(next(csv.reader([l])), l) for l in rows]
        key = sort_key_factory(name, obs_order, axes_order, hdr0)
        parsed.sort(key=lambda t: key(t[0]))
        ks = [key(r) for r, _ in parsed]
        if len(set(ks)) != len(ks):
            die(f"{name}: duplicate keys after assembly")
        (dst / name).write_text(hdr0 + "\n" + "".join(l + "\n" for _, l in parsed))
        rec[name] = {"rows": len(parsed), "sha256_16": hashlib.sha256((dst / name).read_bytes()).hexdigest()[:16]}
    bt: dict = {}
    for d, systems, _ in parts:
        if (d / "recorded_failures_tasks.json").exists():
            for o, m in json.load(open(d / "recorded_failures_tasks.json")).items():
                bt.setdefault(o, {}).update({k: v for k, v in m.items() if k in systems})
    (dst / "recorded_failures_tasks.json").write_text(json.dumps(bt, indent=1, sort_keys=True))
    base = parts[0][0]
    for name in ("task_clusters.csv", "readout_admission.csv"):
        shutil.copy2(base / name, dst / name)
        rec[name] = {"rows": len(read_csv_lines(dst / name)[1]), "sha256_16": hashlib.sha256((dst / name).read_bytes()).hexdigest()[:16],
                     "from": str(base / name)}
    return rec


def parse_extra(spec: str | None) -> list[dict]:
    out = []
    for item in [x for x in (spec or "").split(",") if x.strip()]:
        tier = None
        name, path = item.split("=", 1)
        if ":" in name:
            tier, name = name.split(":", 1)
        tier = tier or ("reference_fed" if name.endswith(".ref") else "self_retrieving")
        if tier not in ("reference_fed", "self_retrieving"):
            die(f"--extra {item}: unknown tier {tier}")
        p = Path(path)
        if not p.exists() or not is_view(p):
            die(f"--extra {name}: {p} is not a view built by view_builder.py")
        out.append({"name": name, "tier": tier, "path": p})
    return out


def cmd_build(a) -> int:
    t0 = time.time()
    if live_out_is_a_view():
        die("the parent must run on the shared tables (CCBENCH_OUT is a view)")
    from windowbench import axes as A
    from windowbench import merged_board as MB
    from windowbench import roster as R
    mb_md5 = md5(Path(MB.__file__))
    V = load_views()
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    if not a.out and not os.environ.get("WINDOWBENCH_OUT"):
        die("give --out or set WINDOWBENCH_OUT")
    out = Path(a.out) if a.out else Path(os.environ["WINDOWBENCH_OUT"]) / f"merged_board_multi_{stamp}"
    if out.exists():
        die(f"{out} exists; a board is never written over an existing directory")
    work = WORK_ROOT / stamp
    work.mkdir(parents=True)
    log(f"work {work}; board {out}")
    state0 = V.shared_state()
    checks: list[tuple[str, bool, str]] = []

    ref_add = [s for s in a.ref.split(",") if s]
    self_add = [s for s in a.self.split(",") if s]
    extras = parse_extra(a.extra)
    for e in extras:
        (ref_add if e["tier"] == "reference_fed" else self_add).append(e["name"])
    ours = list(dict.fromkeys(ref_add + self_add))
    extra_names = {e["name"] for e in extras}
    live_declared = [s for s in ours if (s in R.FAMILIES or s in R.SYSTEMS) and s not in extra_names]
    undeclared = [s for s in ours if s not in live_declared and s not in extra_names]
    if undeclared:
        die(f"{undeclared} are neither declared in the roster nor given as --extra NAME=<view>")
    live_systems = list(dict.fromkeys(list(MB.EXPORT_SYSTEMS) + live_declared))
    log(f"reference_fed + {ref_add}; self_retrieving + {self_add}; live {live_declared}; extras {sorted(extra_names)}")

    live = run_export(V, "live", live_systems, work, None)
    decl = json.load(open(live / "systems.json"))["systems"]
    parts = [(live, set(live_systems), members_of(decl, live_systems))]
    prov = {"live": {"export": str(live), "systems": live_systems}}
    common = list(dict.fromkeys(list(MB.EXPORT_SYSTEMS) + live_declared))
    by_view: dict[Path, list[str]] = {}
    for e in extras:
        by_view.setdefault(e["path"], []).append(e["name"])
    for vd, names in by_view.items():
        why = view_problems(V, vd, names)
        if why:
            die(f"--extra {names}: view {vd}: " + "; ".join(why))
        ex = run_export(V, f"extra_{vd.name}", common + names, work, vd)
        xs = json.load(open(ex / "systems.json"))
        checks += identity_checks(live, ex, common, xs["systems"], f"extra view {vd.name}")
        for n in names:
            decl[n] = xs["systems"][n]
        parts.append((ex, set(names), members_of(xs["systems"], names)))
        man = json.load(open(vd / "view_manifest.json"))
        prov[f"extra_view:{vd.name}"] = {"view": str(vd), "export": str(ex), "keys": names, "injection": xs.get("injection"),
                                         "isolated": {n: {x: (man.get("isolated") or {}).get(n, {}).get(x)
                                                          for x in ("dir", "job", "window_scores_sha256", "rows")} for n in names}}

    inp = out / "inputs"
    out.mkdir(parents=True)
    files = assemble(parts, inp, list(MB.EXPORT_OBS), list(A.AXES))
    rf = pd.read_csv(inp / "recorded_failures.csv")
    bot_tasks = json.load(open(inp / "recorded_failures_tasks.json"))
    bots = {(r.obs, r.system): int(r.n_bot) for r in rf.itertuples()}

    merge = MB.import_merge_windowbench()
    z, clusters = merge.load(out)
    tiers = {k: list(v) for k, v in merge.TIERS.items()}
    tiers["reference_fed"] += [s for s in ref_add if s not in tiers["reference_fed"]]
    tiers["self_retrieving"] += [s for s in self_add if s not in tiers["self_retrieving"]]
    obs_list = [o for o in a.obs.split(",") if o]
    boards, pairs, metas, text = MB.run_board(merge, z, clusters, tiers, obs_list, highlight=ours, bots=bots)
    got = {(m["obs"], m["tier"]) for m in metas}
    lost = [(o, t) for o in obs_list for t in tiers if (o, t) not in got]
    absent = [(m["obs"], m["tier"], m["missing"]) for m in metas if m["missing"]]
    checks.append(("every (obs, tier) board is built with every declared system", not lost and not absent,
                   f"boards {sorted(got)}; not built {lost}; absent systems {absent}"))

    B_ = pd.concat(boards, ignore_index=True)
    P_ = pd.concat(pairs, ignore_index=True)
    B_.to_csv(out / "merged_leaderboard.csv", index=False)
    P_.to_csv(out / "merged_pairs.csv", index=False)
    (out / "merged_meta.json").write_text(json.dumps(metas, indent=1, default=str))
    rows = []
    for m in metas:
        axes = m["axes"].split(",")
        for r in B_[(B_.obs == m["obs"]) & (B_.tier == m["tier"])].itertuples():
            s = z[(z.obs == m["obs"]) & (z.system == r.system) & z.axis.isin(axes)]
            U = s.pivot_table(index="system", columns="task", values="z", aggfunc="mean").loc[r.system]
            if abs(float(U.mean()) - r.composite) > 1e-12:
                die(f"summary: recomputed C of {r.system} differs from the board's")
            d = decl.get(r.system, {})
            nb = None if pd.isna(r.n_bot) else int(r.n_bot)
            row = {"tier": m["tier"], "obs": m["obs"], "system": r.system,
                   "role": d.get("role"), "backbone": d.get("backbone"),
                   "label": d.get("label"), "display_rank": int(r.display_rank),
                   "composite": float(r.composite), "cert_lo": int(r.cert_lo), "cert_hi": int(r.cert_hi),
                   "cert_interval": f"{r.cert_lo}" if r.cert_lo == r.cert_hi else f"{r.cert_lo}-{r.cert_hi}",
                   "n_tasks": int(r.n_tasks), "n_bot": nb, "n_in_tier": len(B_[(B_.obs == m["obs"]) & (B_.tier == m["tier"])]),
                   "pairs_in_family": m["pairs"], "certified_pairs": m["certified"], "axes": m["axes"]}
            for ax in axes:
                row[f"z_{ax}"] = float(s[s.axis == ax].z.mean())
            fail = set(bot_tasks.get(m["obs"], {}).get(r.system, []))
            n_ax = s.groupby("task")["axis"].nunique()
            row["partial_axis_tasks"] = int((n_ax < len(axes)).sum())
            row["failed_tasks_in_C"] = int(len(fail & set(U.dropna().index)))
            rows.append(row)
    S = pd.DataFrame(rows)
    leaked = S[S.failed_tasks_in_C > 0] if len(S) else S
    checks.append(("no recorded failure enters a composite (a task either system failed is dropped from that pair)",
                   leaked.empty, "; ".join(f"{r.system} @ {r.obs}: {r.failed_tasks_in_C}" for r in leaked.itertuples()) or "none"))
    S.to_csv(out / "summary.csv", index=False)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    state1 = V.shared_state()
    changed = V.diff_state(state0, state1)
    mb_ok = md5(Path(MB.__file__)) == mb_md5
    checks.append(("shared state unchanged (tables, rollouts, embed cache, roster_extra, planning rows, merged_board.py)",
                   not changed and mb_ok, f"changed: {changed or 'nothing'}; merged_board md5 ok={mb_ok}"))
    ok_all = all(ok for _, ok, _ in checks)
    (out / "checks.json").write_text(json.dumps([{"claim": c, "pass": bool(o), "detail": d} for c, o, d in checks], indent=1))
    man = {"builder": "evaluation/board/build_board.py", "code_md5": md5(Path(__file__)), "view_builder_md5": md5(VIEWS_DRIVER),
           "merged_board_md5": mb_md5, "ccbench": str(Path(merge.__file__).parent), "argv": sys.argv, "stamp": stamp, "work": str(work),
           "tiers": tiers, "obs": obs_list, "inputs": files, "provenance": prov, "declarations": decl,
           "shared_state_changed": changed, "all_checks_pass": ok_all, "elapsed_s": round(time.time() - t0, 1)}
    (out / "run_manifest.json").write_text(json.dumps(man, indent=1, default=str))
    shutil.copy2(out / "summary.csv", work / "summary.csv")
    print(text)
    print("\nCHECKS")
    for c, o, d in checks:
        print(f"  [{'PASS' if o else 'FAIL'}] {c}\n         {d}")
    print(f"\nwrote {out}/ in {time.time() - t0:.0f}s; all checks pass: {ok_all}")
    return 0 if ok_all else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--ref", default="", help="reference-fed rows declared in the live roster")
    b.add_argument("--self", default="", help="self-retrieving rows declared in the live roster")
    b.add_argument("--extra", default=None, help="NAME=VIEW[,NAME=VIEW]: rows of a view built by view_builder.py")
    b.add_argument("--obs", default=BOARD_OBS)
    b.add_argument("--out", default=None)
    e = sub.add_parser("_export")
    e.add_argument("--systems", required=True)
    e.add_argument("--obs", default=BOARD_OBS)
    e.add_argument("--out", required=True)
    e.add_argument("--view", default=None)
    a = ap.parse_args(argv)
    return cmd_build(a) if a.cmd == "build" else cmd_export(a)

if __name__ == "__main__":
    raise SystemExit(main())
