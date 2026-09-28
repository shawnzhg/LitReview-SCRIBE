"""Runs the twenty-nine rebuild assertions and prints PASS or FAIL for each. Usage: python -m
windowbench.validate --previous <cells.csv> --independent <window_scores.parquet> --replicates <key>."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import decide as DEC
from . import merged_board as MB
from . import roster as R
from .load import HARNESS_FAILURE_STATUS, MISSING_REASONS, Data, _runner_status, _system_acted, status_files

FAILS: list[str] = []
RUN: list[str] = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    RUN.append(name)
    if not ok:
        FAILS.append(name)


CELL_KEY = ["window", "ours_family", "opponent", "readout", "kind"]
CELL_VALUES = ["ours", "opp", "diff", "q"]


def _flag(s: pd.Series) -> np.ndarray:
    return s.astype(str).str.lower().eq("true").to_numpy()


def reproduce_cells(prev: pd.DataFrame, now: pd.DataFrame, tol: float = 1e-9) -> dict:
    a, b = now.set_index(CELL_KEY).sort_index(), prev.set_index(CELL_KEY).sort_index()
    r = {"cells": 0, "only_now": len(a.index.difference(b.index)), "only_prev": len(b.index.difference(a.index)),
         "max_abs": float("nan"), "nan_mismatch": 0, "flips": 0, "unique": bool(a.index.is_unique and b.index.is_unique)}
    if not r["unique"]:
        return {**r, "ok": False}
    common = a.index.intersection(b.index)
    x, y = a.loc[common, CELL_VALUES].to_numpy(float), b.loc[common, CELL_VALUES].to_numpy(float)
    d = np.abs(x - y)
    r.update(cells=len(common), max_abs=float(np.nanmax(d)) if np.isfinite(d).any() else 0.0,
             nan_mismatch=int((np.isnan(x) != np.isnan(y)).sum()),
             flips=int((_flag(a.loc[common, "certified"]) != _flag(b.loc[common, "certified"])).sum()))
    r["ok"] = (r["cells"] > 0 and not r["only_now"] and not r["only_prev"] and not r["nan_mismatch"]
               and r["max_abs"] < tol and not r["flips"])
    return r


def rebuild_cells(D: Data, prev: pd.DataFrame) -> pd.DataFrame:
    from .run import build_cells
    parts = [build_cells(D, fam.split("+"), list(dict.fromkeys(g.opponent)), list(dict.fromkeys(g.window)))
             for fam, g in prev.groupby("ours_family", sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else prev.iloc[:0]


def table_reproduces_previous_analysis(D: Data, previous: str | None):
    name = "the table reproduces the previous analysis cell by cell"
    if not previous or not Path(previous).is_file():
        check(name, False, f"no previous analysis table at {previous}")
        return
    prev = pd.read_csv(previous)
    r = reproduce_cells(prev, rebuild_cells(D, prev))
    check(f"{name} ({r['cells']} cells, max|Δ| {r['max_abs']:.3g})", r["ok"],
          f"{previous}: {r['only_now']} cells only in the rebuild, {r['only_prev']} only in the previous analysis, "
          f"{r['nan_mismatch']} NaN mismatches, {r['flips']} certification flips, unique keys {r['unique']}")


def replicate_null(Z: pd.DataFrame, seeds: list[str], reads: list[str], tasks: list[str], clusters: dict,
                   failed: dict, alpha: float) -> tuple[int, int]:
    d_eff = alpha / DEC.table_pairs(len(seeds))
    n_cells = n_cert = 0
    have = set(Z.index.get_level_values(0)) if not Z.empty else set()
    for a_, b_ in itertools.combinations(seeds, 2):
        if a_ not in have or b_ not in have:
            continue
        for r in reads:
            if r not in Z.columns:
                continue
            sa = DEC.drop_failed(Z.loc[a_][r].reindex(tasks), failed.get(a_, set()))
            sb = DEC.drop_failed(Z.loc[b_][r].reindex(tasks), failed.get(b_, set()))
            res = DEC.contrast(sa, sb, clusters, d_eff, eps=0.0)
            if res["status"] == "too_few_tasks":
                continue
            n_cells += 1
            n_cert += int(res["certified"])
    return n_cells, n_cert


def replicate_null_certifies_nothing(D: Data, key: str):
    from .axes import AXES, axis_readouts
    obs = C.REGISTERED["axis_primary_observation"]
    seeds = R.replicate_keys(key)
    undeclared = [k for k in seeds if k not in R.SYSTEMS]
    if undeclared:
        check("replicate null", False, f"{undeclared} not declared: give the family of {key} \"seeds\": "
                                        f"{list(R.REPLICATE_SEEDS)} in $WINDOWBENCH_ROSTER_EXTRA")
        return
    reads = sorted({r for ax in AXES for r in axis_readouts(D, ax, obs)})
    Z = D.z_matrix(obs, seeds)
    n_cells, n_cert = replicate_null(Z, seeds, reads, D.tasks, D.clusters,
                                     {k: D.bot.get((k, obs), set()) for k in seeds}, C.REGISTERED["alpha"])
    check(f"replicate null at {obs}: {n_cert}/{n_cells} cells certified among {len(seeds)} seeds of {key}",
          n_cells > 0 and n_cert == 0)


def entry_byte_equality(tasks: list[str], entries: dict[str, dict[str, Path]], canonical_dir: Path,
                        allowlist_dir: Path) -> dict[str, list[str]]:
    out = {}
    for t in tasks:
        bad = []
        files = {arm: m.get(t) for arm, m in entries.items()}
        raw = {arm: f.read_bytes() for arm, f in files.items() if f is not None and f.exists()}
        if not raw:
            bad.append("no fixed-input entry")
        if len(set(raw.values())) > 1:
            bad.append(f"entry bytes differ across {sorted(raw)}")
        cp, ap = canonical_dir / f"{t}.json", allowlist_dir / f"{t}.json"
        canon = json.loads(cp.read_text()) if cp.exists() else None
        allow = [str(x) for x in json.loads(ap.read_text())] if ap.exists() else None
        if canon is None or allow is None:
            bad.append("canonical bundle or shared allowlist missing")
        else:
            for arm, b in raw.items():
                if json.loads(b).get("content_hash") != canon.get("content_hash"):
                    bad.append(f"{arm}: entry is not the canonical bundle")
            val = canon.get("validation") or {}
            excl = {str(x) for x in [val.get("review_pmid_excluded"), *(val.get("post_cutoff_excluded_pmids") or [])] if x}
            papers = {str(p.get("paper_id")) for p in canon.get("papers") or []}
            if papers != set(allow) - excl:
                bad.append("canonical bundle papers != shared allowlist minus declared exclusions")
        out[t] = bad
    return out


def entry_equality(D: Data):
    root = C.SOURCES["bundle_entry_root"]
    arms = sorted({k.split(".")[0] for k in R.SYSTEMS if ".bundle_entry" in k})
    exposed = [a for a in arms if (root / a / "bundle_entry").is_dir()]
    if not exposed:
        check("entry byte-equality", False, f"no fixed-input run of ours under {root}")
        return
    entries = {}
    for a in exposed:
        d = root / a / "bundle_entry"
        for sd in sorted(d.glob("*/seed*")):
            entries.setdefault(f"{a}/{sd.name}", {})[sd.parent.name] = sd / "entry_evidence_bundle.json"
    res = entry_byte_equality(D.tasks, entries, C.SOURCES["canonical_bundles"], C.SOURCES["allowlists"])
    bad = {t: p for t, p in res.items() if p}
    check(f"entry byte-equality across {len(entries)} fixed-input runs: {len(res) - len(bad)}/{len(res)} tasks",
          not bad, f"{sorted(bad.items())[:2]}")


def independent_rescoring(table: pd.DataFrame, other: pd.DataFrame, tol: float = 1e-9) -> tuple[int, pd.DataFrame, bool]:
    key = ["system", "task", "window", "readout"]
    a = table.set_index(key).value
    b = other[other.system.isin(set(table.system))].set_index(key).value
    if not (a.index.is_unique and b.index.is_unique):
        return 0, pd.DataFrame(columns=key), False
    common = a.index.intersection(b.index)
    x, y = a.loc[common].astype(float).to_numpy(), b.loc[common].astype(float).to_numpy()
    differ = (np.abs(x - y) > tol) | (np.isnan(x) != np.isnan(y))
    return len(common), common[differ].to_frame(index=False), True


def reference_fed_reports(D) -> pd.DataFrame:
    s = D.scores
    return s[(s.source == "E13") & s.system.isin(R.DATASETS["fixed_input"])]


def rescoring_differs_only_on_excluded_readouts(D: Data, independent: str | None):
    name = "rescoring the reference-fed reports on an independent implementation differs only on excluded readouts"
    if not independent or not Path(independent).is_file():
        check(name, False, f"no independent score table at {independent}")
        return
    p = Path(independent)
    other = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    n, diff, unique = independent_rescoring(reference_fed_reports(D), other)
    adm = MB.export_readout_admission(D, sorted(set(diff.window))) if len(diff) else pd.DataFrame(columns=["obs", "readout", "admitted"])
    admitted = set(map(tuple, adm[adm.admitted.astype(bool)][["obs", "readout"]].to_numpy()))
    on_admitted = sorted({(w, r) for w, r in zip(diff.window, diff.readout) if (w, r) in admitted})
    check(f"{name} ({len(diff)} of {n} cells differ, on {sorted(set(diff.readout))})", unique and n > 0 and not on_admitted,
          f"{p}; unique keys {unique}; differing on admitted readouts: {on_admitted[:5]}")


def fairness_regimes(D: Data, key: str):
    from . import fairness as F
    if key not in R.SYSTEMS:
        for name in ("ours vs a fixed-input pipeline is exact", "LiRA vs a self-retrieving pipeline is refused",
                     "a backbone change is refused in a scaffold family", "fixed input vs same pool is not exact",
                     "a fixed-input arm is not observable at retrieval"):
            check(name, False, f"{key} is not declared")
        return
    r1 = F.regime(key, "autosurvey.ref", "system", "syn_required_cov", D.radii, D.eps_fair)
    r2 = F.regime("lira", "autosurvey", "system", "syn_required_cov", D.radii, D.eps_fair)
    r3 = F.regime("drtulu", "autosurvey", "system", "syn_required_cov", D.radii, D.eps_fair, scaffold=True)
    r4 = F.regime("autosurvey.ref", "autosurvey", "system", "syn_required_cov", D.radii, D.eps_fair)
    r5 = F.regime(key, "autosurvey", "retrieval", "ret_hub_recall", D.radii, D.eps_fair)
    check("ours vs a fixed-input pipeline at the system window is exact, with the backbone difference flagged",
          r1["regime"] == "exact" and r1.get("cross_model") is True and r1["eps_outside"] == 0.0, f"{r1['regime']}:{r1['reason']}")
    check("LiRA vs a self-retrieving pipeline is refused by the admissibility test", r2["regime"] == "unfair"
          and r2["reason"] in ("eps_out_exceeds_eps_fair", "no_finite_radius", "no_admissible_tolerance"), f"{r2['regime']}:{r2['reason']}")
    check("a backbone change has no finite radius in a scaffold family", r3["regime"] == "unfair"
          and r3["reason"] == "backbone_outside_scaffold_window", f"{r3['regime']}:{r3['reason']}")
    check("a fixed-input run and a same-pool run do not share an entry", r4["regime"] != "exact", f"{r4['regime']}:{r4['reason']}")
    check("a fixed-input arm is not observable at the retrieval window", r5["regime"] == "not_observable", r5["regime"])


def decision_rule(D: Data):
    tasks = [f"t{i:02d}" for i in range(40)]
    clusters = {t: f"c{i % 8}" for i, t in enumerate(tasks)}
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0.5, 0.05, len(tasks)), index=tasks)
    few = DEC.contrast(base.iloc[: C.REGISTERED["min_tasks"] - 1] + 0.5, base.iloc[: C.REGISTERED["min_tasks"] - 1], clusters, 0.05, eps=0.0)
    check(f"a pair with fewer than {C.REGISTERED['min_tasks']} common tasks is never certified",
          few["status"] == "too_few_tasks" and not few["certified"], few["status"])
    failed = DEC.contrast(DEC.drop_failed(base + 0.3, {tasks[0], tasks[1]}), base, clusters, 0.05, eps=0.0)
    check("a task either system failed is dropped from the pair, not scored", failed["n"] == len(tasks) - 2, f"n={failed['n']}")
    shifted = base + 0.02 + rng.normal(0, 0.002, len(tasks))
    gap = DEC.contrast(shifted, base, clusters, 0.05, eps=0.0)
    wide = DEC.contrast(shifted, base, clusters, 0.05, eps=0.05)
    check("a pair is certified only when the gap exceeds the sampling radius plus the nuisance radius",
          gap["certified"] and not wide["certified"] and wide["sampling_decided"], f"q={gap['q']:.4f}")


def table_family(D: Data, key: str):
    from . import axes as A
    names = ("the Bonferroni level divides alpha over all pairs of the table and its axes",
             "the composite weights the axes equally",
             "every certified relation holds in the rank intervals of the transitive closure")
    obs = C.REGISTERED["axis_primary_observation"]
    systems = list(R.DATASETS["fixed_input"]) + ([key] if key in R.SYSTEMS else [])
    conf = A.axis_confirmatory(D, systems, obs)
    z = A.axis_z_rows(D, systems, obs)
    if conf.empty or z.empty:
        for name in names:
            check(name, False, f"no axis scores at {obs}")
        return
    want = len(systems) * (len(systems) - 1) // 2
    ok = set(conf.n_pairs_in_family) == {want} and np.allclose(
        conf.d_eff, C.REGISTERED["alpha"] / (conf.n_pairs_in_family * conf.n_axes_in_family))
    check(names[0], ok, f"{want} pairs; family sizes {sorted(set(conf.n_pairs_in_family))}")
    try:
        U, axes = A.composite(D, z, obs, systems)
    except ValueError as e:
        check(names[1], False, str(e))
    else:
        per = z[z.axis.isin(axes)].groupby(["system", "task"]).z.mean()
        diffs = [abs(float(U.loc[s, t]) - float(v)) for (s, t), v in per.items()
                 if s in U.index and t in U.columns and U.loc[s, t] == U.loc[s, t]]
        check(names[1], bool(diffs) and max(diffs) < 1e-9, f"{len(diffs)} cells")
    scored = conf[conf.status != "not_scored"]
    ri = A.rank_intervals(scored, systems)
    bad = []
    for r in scored[scored.certified.fillna(False).astype(bool) & scored.winner.isin(["a", "b"])].itertuples():
        g = ri[(ri.axis == r.axis) & (ri.obs == r.obs)].set_index("system")
        w, l = (r.a, r.b) if r.winner == "a" else (r.b, r.a)
        if not (g.loc[w, "lo"] < g.loc[l, "lo"] and g.loc[w, "hi"] < g.loc[l, "hi"]):
            bad.append((r.axis, w, l))
    check(names[2], not bad, f"{bad[:3]}")


def _rollout_delivered(sys_key: str, task: str) -> bool:
    fp = C.CCB_OUT / "rollouts" / sys_key / f"{task}.json"
    if not fp.exists():
        return False
    try:
        d = json.load(open(fp))
    except (OSError, json.JSONDecodeError):
        return False
    return str(d.get("status", "ok")) == "ok" and bool((d.get("report") or {}).get("sentences"))


def recorded_failures(D: Data):
    dropped = getattr(D, "bot_dropped", [])
    undeclared = []
    for sys_key, task, why in dropped:
        why = str(why)
        if any(f"status={h}" in why for h in HARNESS_FAILURE_STATUS) or any(f"reason={m}" in why for m in MISSING_REASONS) \
           or why.endswith("no_model_call"):
            continue
        undeclared.append((sys_key, task, why))
    check(f"every failure candidate set aside as not a system failure has a declared reason ({len(dropped)} set aside)",
          not undeclared, f"{undeclared[:4]}")
    harness, never_acted, delivered = [], [], []
    for (sys_key, window), tasks in D.bot.items():
        if window != "system_trunc" or sys_key not in R.SYSTEMS:
            continue
        for t in tasks:
            if _rollout_delivered(sys_key, t):
                delivered.append((sys_key, t))
        if R.SYSTEMS[sys_key]["role"] != "baseline":
            continue
        base, fixed = sys_key.replace(".ref", ""), sys_key.endswith(".ref")
        st = _runner_status(status_files(base, fixed))
        for t in tasks:
            if st.get(t) in HARNESS_FAILURE_STATUS:
                harness.append((sys_key, t, st.get(t)))
            if _system_acted(base, t, fixed) is False:
                never_acted.append((sys_key, t))
    check("no harness failure is recorded as a system failure", not harness, f"{harness[:4]}")
    check("every recorded failure of a published pipeline has a recorded model call", not never_acted, f"{never_acted[:4]}")
    check("no task is a recorded failure while its rollout delivered a report", not delivered, f"{delivered[:4]}")


def draft_window(D: Data):
    leaked = [s for s in D.systems_present if s.endswith(".draft")]
    check("no `.draft` key survives as a system", not leaked, str(leaked[:5]))
    have = set(D.scores[D.scores.window == "draft"].system)
    ds = set(getattr(D, "draft_systems", []))
    check(f"draft rows present for every draft-exit system ({len(ds)} scored)", ds <= have, str(sorted(ds - have)[:5]))
    scored = set(D.scores[D.scores.window == "system_trunc"].system)
    refiners = {k for k in scored if k in R.SYSTEMS and R.SYSTEMS[k]["entries"].get("draft")
                and not R.SYSTEMS[k].get("draft_is_final")}
    check("every scored pipeline with a refinement stage is scored at its draft exit", refiners <= ds,
          str(sorted(refiners - ds)[:5]))


def allocation_is_reproducible(D: Data):
    names = ("every scored arm reaches the calibrated allocation frame",
             "no allocation pair is certified outside the fair regimes",
             "only a chance-corrected readout certifies the allocation exit")
    d = C.OUT_DEFAULT / "allocation"
    need = [d / f for f in ("allocation_scores.csv", "allocation_calibrated.csv", "allocation_pairs.csv")]
    if not all(p.exists() for p in need):
        for name in names:
            check(name, False, "run windowbench.allocation_pairs")
        return
    S, Z, P = (pd.read_csv(p) for p in need)
    missing = sorted(set(S.system.unique()) - set(Z.system.unique()))
    check(names[0], not missing, f"missing: {missing}")
    bad = P[P.certified.fillna(False).astype(bool) & ~P.regime.isin(["exact", "tolerance"])]
    check(names[1], bad.empty, f"{len(bad)} cells: {sorted(set(zip(bad.a, bad.b)))[:4]}")
    from .allocation_pairs import ALLOC_READS
    not_cc = sorted(set(P.readout.unique()) - set(ALLOC_READS))
    check(names[2], not not_cc, f"certifying readouts: {sorted(set(P.readout.unique()))}")


def channel_and_registry(D: Data):
    trunc = set(D.human[D.human.window == "system_trunc"].task)
    check("peers are scored at the matched-length view on every task", set(D.tasks) <= trunc,
          f"{len(set(D.tasks) - trunc)} tasks without matched-length peers")
    extra = sorted(set(D.scores.window) - set(C.REGISTERED["windows"]))
    check("every score row lies at a registered window", not extra, str(extra))
    reg = C.registry()
    unhashed = [k for k, v in reg["inputs"].items() if v["exists"] and not v["sha256_16"]]
    check("the analysis constants are recorded with input hashes", bool(reg.get("constants_sha256_16")) and not unhashed,
          str(unhashed))
    check("the planning axis is loaded with its peer band", bool(getattr(D, "outline_axis_loaded", False))
          and len(D.outline_axis_human) > 0, "run windowbench.outline_axis")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--previous", required=True, help="frozen cells.csv of the previous analysis")
    ap.add_argument("--independent", required=True,
                    help="score table (parquet or csv) of the reference-fed reports from an independent implementation")
    ap.add_argument("--replicates", required=True,
                    help="fixed-input key of SCRIBE (untrained); its seeds 1 and 2 are <key>.seed1 and <key>.seed2")
    a = ap.parse_args(argv)
    D = Data()
    print(f"loaded: {len(D.scores)} score rows, {len(D.systems_present)} systems")
    table_reproduces_previous_analysis(D, a.previous)
    replicate_null_certifies_nothing(D, a.replicates)
    entry_equality(D)
    rescoring_differs_only_on_excluded_readouts(D, a.independent)
    fairness_regimes(D, a.replicates)
    decision_rule(D)
    table_family(D, a.replicates)
    recorded_failures(D)
    draft_window(D)
    allocation_is_reproducible(D)
    channel_and_registry(D)
    print(f"\n{len(RUN)} assertions: " + ("all pass" if not FAILS else f"{len(FAILS)} failed: {FAILS}"))
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
