#!/bin/bash
# Scores one added system into a private output root under $ISO_ROOT without writing the shared
# tables: an external agent's converted runs, or our same-pool runs through the budget-stop overlay.
# Usage: score_iso.sh agent <agent> <runs dir> | score_iso.sh native <key> <level>; LOCAL=1 runs in
# the current shell.
set -uo pipefail
shopt -s nullglob
export PYTHONDONTWRITEBYTECODE=1
die() { echo "FATAL: $*"; exit 2; }
USAGE="usage: score_iso.sh agent <agent> <converted runs dir> | score_iso.sh native <key> <level>"
LITREVIEW_ROOT=${LITREVIEW_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export LITREVIEW_ROOT
BOARD=$LITREVIEW_ROOT/evaluation/board
MODE=${1:-}
case "$MODE" in
  agent) [ $# -eq 3 ] || die "$USAGE"; AGENT=$2; RUNS=$3; NAME=$AGENT;;
  native) [ $# -eq 3 ] || die "$USAGE"; KEY=$2; LEVEL=$3; NAME=$KEY;;
  *) die "$USAGE";;
esac
{ [ -f "$BOARD/view_builder.py" ] && [ -d "$LITREVIEW_ROOT/evaluation/ccbench" ]; } || die "LITREVIEW_ROOT=$LITREVIEW_ROOT is not the repository root"
{ [ -n "${CCBENCH_ENV:-}" ] && [ -f "$CCBENCH_ENV" ]; } || die "set CCBENCH_ENV to the site environment script"
source "$CCBENCH_ENV"
{ [ -n "${PY:-}" ] && [ -n "${CCBENCH_ROOT:-}" ] && [ -n "${CCBENCH_PARENT:-}" ]; } || die "$CCBENCH_ENV must define PY, CCBENCH_ROOT and CCBENCH_PARENT"
SHARED_OUT=$CCBENCH_ROOT/out
[ -f "$SHARED_OUT/E1/conformance.csv" ] || die "$SHARED_OUT holds no shared tables"
[ -n "${ISO_ROOT:-}" ] || die "set ISO_ROOT to the directory that receives the private output roots"
ISO_ROOT=$(realpath -m "$ISO_ROOT")
for t in "$CCBENCH_ROOT" "$CCBENCH_PARENT"; do
  r=$(realpath -e "$t") || die "$t missing"
  case "$ISO_ROOT/" in "$r"/*) die "ISO_ROOT=$ISO_ROOT lies inside $t";; esac
done
POOL_SNAPSHOT=$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import pool_retrieval_tool as P; print(P.POOL_SNAPSHOT_ID)' "$LITREVIEW_ROOT/scribe/harness/tools") \
  || die "cannot read the frozen pool snapshot id from scribe/harness/tools/pool_retrieval_tool.py"
export CCBENCH_ENV ISO_ROOT

if [ "$MODE" = agent ]; then
  [[ "$AGENT" =~ ^[a-z][a-z_]*$ ]] || die "unknown agent $AGENT"
  ING=$BOARD/ingest/$AGENT.py
  SPEC=$(cd "$BOARD" && "$PY" -c "import sys; from view_agents import AGENTS; ks = AGENTS[sys.argv[1]]['keys']; print(','.join(ks)); print(' '.join(sorted({dict(fixed_input='fixinput', same_pool='samepool')[v['dataset']] for v in ks.values()})))" "$AGENT" 2>/dev/null) \
    || die "unknown agent $AGENT (not in view_agents.py)"
  [ -f "$ING" ] || die "$ING missing"
  KEYS=$(sed -n 1p <<<"$SPEC"); CONDS=$(sed -n 2p <<<"$SPEC")
  { [ -n "${COMMERCIAL_OUT:-}" ] && [ -d "$COMMERCIAL_OUT" ]; } || die "set COMMERCIAL_OUT to the agent's run bundle"
  export COMMERCIAL_OUT
  [ -d "$RUNS" ] || die "$RUNS is not a directory"
  RUNS=$(realpath -e "$RUNS")
  for c in fixinput samepool; do
    if [[ " $CONDS " == *" $c "* ]]; then
      [ -d "$RUNS/$c" ] || die "$RUNS has no $c/; it is not a converted runs dir of $AGENT"
    else
      [ ! -e "$RUNS/$c" ] || die "$RUNS has $c/, which $AGENT does not run"
    fi
  done
  for k in ${KEYS//,/ }; do [ ! -e "$SHARED_OUT/rollouts/$k" ] || die "$k already has rollouts in the shared tables"; done
else
  [[ "$KEY" =~ ^[A-Za-z0-9_]+$ ]] || die "KEY must be a bare native label"
  [[ "$LEVEL" =~ ^[0-9]+$ ]] || die "LEVEL must be a run level number"
  [ -n "${SCRIBE_RUNS_ROOT:-}" ] || die "set SCRIBE_RUNS_ROOT to the run tree"
  [ -d "$SCRIBE_RUNS_ROOT/level$LEVEL" ] || die "$SCRIBE_RUNS_ROOT/level$LEVEL missing"
  RUNS_ROOT=$(realpath -e "$SCRIBE_RUNS_ROOT")
  PARENT_R=$(realpath -e "$CCBENCH_PARENT")
  case "$RUNS_ROOT/" in "$PARENT_R"/*) ;; *) die "SCRIBE_RUNS_ROOT=$RUNS_ROOT does not lie inside CCBENCH_PARENT=$PARENT_R (the scorer reads runs there)";; esac
  RUNS_REL=${RUNS_ROOT#"$PARENT_R"/}
  case "$ISO_ROOT/" in "$RUNS_ROOT"/*) die "ISO_ROOT=$ISO_ROOT lies inside SCRIBE_RUNS_ROOT=$RUNS_ROOT";; esac
  export RUNS_ROOT PARENT_R RUNS_REL
  CH=("$RUNS_ROOT/level$LEVEL"/*/native_chain)
  [ ${#CH[@]} -eq 1 ] || die "level$LEVEL holds ${#CH[@]} <system>/native_chain trees; exactly one is scored"
  SYSTEM=$(basename "$(dirname "${CH[0]}")")
  [ ! -e "$SHARED_OUT/rollouts/$KEY" ] || die "$KEY already has rollouts in the shared tables"
  KEYS=$KEY
fi

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${LOCAL:-0}" != 1 ]; then
  mkdir -p "$ISO_ROOT" || die "cannot create $ISO_ROOT"
  exec sbatch --parsable --job-name=score_iso --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=04:00:00 \
    --export=ALL,LITREVIEW_ROOT="$LITREVIEW_ROOT" --output="$ISO_ROOT/score_iso_%j.log" "$0" "$@"
fi

WORK=$ISO_ROOT/${NAME}_iso_${SLURM_JOB_ID:-$$}
[ ! -e "$WORK" ] || die "$WORK exists; it must be fresh"
mkdir -p "$WORK" || die "cannot create $WORK"

step() {
  local label=$1; shift
  echo "[$(date -u +%H:%M:%S)] === $label: $*"
  "$@" 2>&1 | tail -"${TAIL:-25}"
  local rc=${PIPESTATUS[0]}
  echo "[$(date -u +%H:%M:%S)] === $label rc=$rc"
  return "$rc"
}

native_entry() {
  local ov=$WORK/overlay led nref
  step "overlay level$LEVEL" "$PY" "$BOARD/budget_stop_overlay.py" --level "$LEVEL" --root "$ov" --runs-root "$RUNS_ROOT" \
    || { echo "FATAL: the overlay refused level$LEVEL"; exit 2; }
  led=$ov/overlay_ledger_level$LEVEL.json
  cp -p "$led" "$ov/overlay_ledger_level$LEVEL.tsv" "$WORK/"
  nref=$("$PY" - "$led" <<'PYEOF'
import json, os, sys
r = [u for u in json.load(open(sys.argv[1]))["units"]
     if u["verdict"] == "refused" and str(u["seed"]) == "0" and os.path.exists(os.path.join(u["unit"], "report_artifact.json"))]
for u in r:
    print(f"  refused: {u['task']}: {u['reason'][:200]}", file=sys.stderr)
print(len(r))
PYEOF
) || { echo "FATAL: cannot read the overlay ledger"; exit 2; }
  [ "$nref" = 0 ] || { echo "FATAL: $nref unit(s) the overlay refused would be scored"; exit 2; }
  "$PY" - "$ov/level$LEVEL" "$SYSTEM" "$POOL_SNAPSHOT" "$WORK/level_gate.json" <<'PYEOF' || { echo "FATAL: level$LEVEL is not a passed pool level (see $WORK/level_gate.json)"; exit 2; }
import glob, json, os, sys
lvl, system, snap, out = sys.argv[1:5]
L = int(os.path.basename(lvl)[5:])
bad, info = [], {}
marks = sorted(glob.glob(os.path.join(lvl, "_level_infra_failure_*.json")) + glob.glob(os.path.join(lvl, "_level_*failed*.json")))
if marks:
    bad.append(f"failure markers: {marks[:3]}")
okp = os.path.join(lvl, "_level_acq_audit_ok.json")
try:
    ok = json.load(open(okp))
except Exception as e:
    ok = None
    bad.append(f"no acquisition audit record {okp} ({type(e).__name__})")
if ok is not None and ok.get("ok") is not True:
    bad.append(f"audit record ok={ok.get('ok')}")
rec = None
if ok is not None:
    rdir = os.path.dirname(ok.get("audit") or "")
    cands = [os.path.join(rdir, "level_record.json")] + glob.glob(os.path.join(os.path.dirname(rdir), f"eval_{ok.get('arm')}_*", "level_record.json"))
    recs = []
    for rp in set(cands):
        try:
            r = json.load(open(rp))
        except Exception:
            continue
        if int(r.get("level", -1)) == L and r.get("arm") == ok.get("arm"):
            recs.append((os.path.getmtime(rp), rp, r))
    if recs:
        _, rp, rec = max(recs)
        info["record_path"] = rp
    else:
        bad.append(f"no level_record.json of arm {ok.get('arm')} at this level near {rdir}")
if rec is not None:
    info["record"] = {k: rec.get(k) for k in ("rc", "level", "retrieval_budget", "n_units", "n_bundles", "n_reports",
                                                "n_tasks_expected", "acq_audit_ok", "corpus_snapshot_id", "job", "arm")}
    if int(rec.get("rc", -1)) != 0:
        bad.append(f"record rc {rec.get('rc')} != 0")
    if int(rec.get("level", -1)) != L:
        bad.append("record is for another level")
    if rec.get("corpus_snapshot_id") != snap:
        bad.append(f"record snapshot {rec.get('corpus_snapshot_id')} != {snap}")
    if rec.get("acq_audit_ok") is not True:
        bad.append("record says the acquisition audit did not pass")
    n = rec.get("n_tasks_expected")
    if n is None or rec.get("n_bundles") != n or rec.get("n_reports") != n:
        bad.append(f"record bundles {rec.get('n_bundles')} / reports {rec.get('n_reports')} != tasks x seeds {n}")
    info["retrieval_budget"] = rec.get("retrieval_budget")
nu, ubad = 0, []
for u in sorted(glob.glob(os.path.join(lvl, system, "native_chain", "*", "seed0"))):
    if not os.path.exists(os.path.join(u, "report_artifact.json")):
        continue
    nu += 1
    t = os.path.basename(os.path.dirname(u))
    try:
        mans = [json.loads(l) for l in open(os.path.join(u, "manifests.jsonl")) if l.strip()]
    except Exception:
        mans = []
    acq = [m for m in mans if m.get("window") == "acquisition"]
    if not acq or any(m.get("corpus_snapshot_id") != snap for m in acq):
        ubad.append(f"{t}: acquisition manifest missing or not {snap}")
    try:
        if json.load(open(os.path.join(u, "pool_provenance.json"))).get("ok") is not True:
            ubad.append(f"{t}: pool_provenance not ok")
    except Exception:
        ubad.append(f"{t}: no pool_provenance.json")
    if not os.path.exists(os.path.join(u, "retrieval_budget.json")):
        ubad.append(f"{t}: no retrieval_budget.json")
bad += ubad[:20]
if nu == 0:
    bad.append("no unit to score")
info.update(n_units_linked=nu, problems=bad, units="report", allow_partial=False, snapshot=snap, system=system)
json.dump(info, open(out, "w"), indent=1)
print(f"### level gate: {nu} units, {len(bad)} problems" + (f": {bad[:3]}" if bad else ""))
sys.exit(1 if bad else 0)
PYEOF
  local p=$WORK/parent src=$PARENT_R dst rel=$RUNS_REL head e n u t nu=0
  dst=$p; mkdir -p "$dst"
  while :; do
    head=${rel%%/*}
    for e in "$src"/*; do n=$(basename "$e"); [ "$n" = "$head" ] || ln -s "$e" "$dst/$n"; done
    mkdir -p "$dst/$head"; src=$src/$head; dst=$dst/$head
    [ "$rel" = "$head" ] && break
    rel=${rel#*/}
  done
  for e in "$src"/*; do n=$(basename "$e"); [ "$n" = "level$LEVEL" ] || ln -s "$e" "$dst/$n"; done
  local d=$dst/level$LEVEL/$KEY/native_chain
  mkdir -p "$d"
  for u in "$ov/level$LEVEL/$SYSTEM/native_chain"/*/seed0; do
    [ -f "$u/report_artifact.json" ] || continue
    t=$(basename "$(dirname "$u")"); mkdir -p "$d/$t"; ln -s "$u" "$d/$t/seed0"; nu=$((nu+1))
  done
  echo "### shadow parent $p: $KEY -> level$LEVEL/$SYSTEM seed0, $nu units linked"
  [ "$nu" -gt 0 ] || { echo "FATAL: no unit to score"; exit 2; }
  export CCBENCH_PARENT=$p CCBENCH_PANEL_B_EXTRA=$KEY CCBENCH_MODE_LEVELS="native_chain=level$LEVEL"
  TAIL=12 step "build" "$PY" -u -m ccbench.build --systems "$KEY" --redo || exit 3
}

score() {
  echo "### start=$(date) node=$(hostname) mode=$MODE name=$NAME keys=$KEYS work=$WORK"
  echo "### launcher md5 $(md5sum "$BOARD/score_iso.sh" | awk '{print $1}')"
  export CCBENCH_DEVICE=${CCBENCH_DEVICE:-cuda}
  [ "$CCBENCH_DEVICE" != cuda ] || [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: CCBENCH_DEVICE=cuda but no GPU is visible (request one, or set CCBENCH_DEVICE=cpu)"; exit 2; }
  local o=$WORK/out f n
  mkdir -p "$o/E1" "$o/E10" "$o/E13" "$o/rollouts" "$o/_embed_cache"
  for f in "$SHARED_OUT"/E10/*; do
    n=$(basename "$f"); [ -f "$f" ] || continue
    case "$n" in unit_scores*|*.lock) continue;; esac
    cp -p "$f" "$o/E10/"
  done
  find "$SHARED_OUT/_embed_cache" -maxdepth 1 -type f -name '*.npy' -printf '%f\n' | while read -r n; do ln -s "$SHARED_OUT/_embed_cache/$n" "$o/_embed_cache/$n"; done
  cmp -s "$SHARED_OUT/E10/null_tau.json" "$o/E10/null_tau.json" || { echo "FATAL: the private null_tau.json is not the shared one"; exit 2; }
  echo "### private out $o: E10 files $(ls "$o/E10" | wc -l), embed-cache links $(ls "$o/_embed_cache" | wc -l)"
  md5sum "$SHARED_OUT"/E1/conformance.csv "$SHARED_OUT"/E1/summary.csv "$SHARED_OUT"/E10/unit_scores.parquet \
    "$SHARED_OUT"/E13/window_scores.parquet "$SHARED_OUT"/E13/radii.parquet "$SHARED_OUT"/E13/distances.parquet \
    "$SHARED_OUT"/E10/null_tau.json 2>/dev/null > "$WORK/shared_before.md5"
  ls "$SHARED_OUT/_embed_cache" > "$WORK/shared_embed_before.txt"
  export PYTHONPATH=$LITREVIEW_ROOT/evaluation:${PYTHONPATH:-}
  export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} PYTHONWARNINGS=ignore PYTHONHASHSEED=0
  export CCBENCH_OUT=$o
  cd "$LITREVIEW_ROOT/evaluation" || exit 2
  if [ "$MODE" = agent ]; then
    TAIL=4 step "build ($AGENT ingest)" "$PY" -u "$ING" build --runs "$RUNS" || exit 3
  else
    native_entry
  fi
  TAIL=12 step "score_units" "$PY" -u -m ccbench.score_units --systems "$KEYS" --no-cache || exit 3
  TAIL=20 step "score_windows" "$PY" -u -m ccbench.score_windows --systems "$KEYS" --no-cache --truncate || exit 3
  md5sum -c --quiet "$WORK/shared_before.md5" || { echo "FATAL: a shared table changed during this job"; exit 7; }
  ls "$SHARED_OUT/_embed_cache" > "$WORK/shared_embed_after.txt"
  cmp -s "$WORK/shared_embed_before.txt" "$WORK/shared_embed_after.txt" || { echo "FATAL: the shared embedding cache changed during this job"; exit 7; }
  echo "### shared tables and the shared embed cache unchanged ($(wc -l < "$WORK/shared_embed_before.txt") files); private cache misses: $(find "$o/_embed_cache" -maxdepth 1 -type f | wc -l)"
  echo "### end=$(date) rc=0 (rows: $o/E13/window_scores.parquet)"
}

score 2>&1 | tee "$WORK/score_iso.log"
exit "${PIPESTATUS[0]}"
