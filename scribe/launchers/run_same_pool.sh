#!/bin/bash
# Submits same-pool SCRIBE generation from the login node: checks the carriers, backbone, levers and
# pinned acquisition units, then submits the prepare, shard and finalize jobs. Usage: LEVEL=<n>
# TAG=<name> SRC_PINS=<pins> bash run_same_pool.sh.
set -uo pipefail
die(){ echo "REFUSED: $*"; exit 2; }
LITREVIEW_ROOT=${LITREVIEW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
[ -f "$LITREVIEW_ROOT/scribe/harness/runners/runner.py" ] || die "set LITREVIEW_ROOT to the repo (configs/site.env.example)"
for v in SCRIBE_WORK_ROOT SCRIBE_RUN_PY SCRIBE_MODEL_DIR SCRIBE_THETA0 SCRIBE_RUNS_ROOT; do
  [ -n "${!v:-}" ] || die "$v is not set (configs/site.env.example)"
done
L=$LITREVIEW_ROOT/scribe/launchers
RUN_PY=$SCRIBE_RUN_PY; MODEL0=$SCRIBE_MODEL_DIR; THETA0=$SCRIBE_THETA0; RUNS=$SCRIBE_RUNS_ROOT
LOGS=$SCRIBE_WORK_ROOT/logs
LEVEL=${LEVEL:?LEVEL required}; TAG=${TAG:?TAG required}
[[ "$LEVEL" =~ ^[0-9]+$ ]] && [ "$LEVEL" -ge 1 ] || die "LEVEL=$LEVEL must be a positive integer"
[[ "$TAG" =~ ^[A-Za-z0-9_]{6,}$ ]] || die "TAG must be [A-Za-z0-9_]{6,}"
PINS=${SRC_PINS:-}
[ -f "$PINS" ] || die "SRC_PINS (a same_pool_src_pins/1 file written by same_pool_prepare.py pin) is required"
read -r SRC_ROOT SEED < <($RUN_PY -c "import json,sys; p=json.load(open(sys.argv[1])); print(p['src_root'], p['seed'])" "$PINS") || die "SRC_PINS unreadable"
SRC_RUN=$($RUN_PY -c "import glob,json,sys; r={json.load(open(f)).get('run') for f in glob.glob(sys.argv[1] + '/_level_claim_*.json')}; print(r.pop() if len(r) == 1 else '')" "$SRC_ROOT")
[ -d "$SRC_RUN" ] && [ -f "$SRC_RUN/code.md5" ] && [ -f "$SRC_RUN/search_calls.jsonl" ] \
  || die "the source level $SRC_ROOT names no single acquisition run dir with code.md5 and search_calls.jsonl (its _level_claim_*.json)"
PINS_SHA=$(sha256sum < "$PINS" | cut -c1-64); SLOG_SHA=$(sha256sum < "$SRC_RUN/search_calls.jsonl" | cut -c1-64)
SRC_AUDIT_OK=$SRC_ROOT/_level_acq_audit_ok.json
[ -f "$SRC_AUDIT_OK" ] || die "the source level has no $SRC_AUDIT_OK"
SRC=level; SRC_DESC="same-pool acquisition $(basename $SRC_RUN), pins $(basename $PINS)"
SYSTEM=${SYSTEM:-SCRIBE}
[ "$SYSTEM" = SCRIBE ] || die "SYSTEM=$SYSTEM: the pipeline has one system, SCRIBE"
LEVERS=${LEVERS:-$LITREVIEW_ROOT/scribe/levers/writing_levers.json}
[ -f "$LEVERS" ] || die "LEVERS=$LEVERS is not a file (the pipeline always runs the lever writer)"
NSHARDS=${NSHARDS:-5}; TIME=${TIME:-03:00:00}; GEN_MAXLEN=65536
[[ "$NSHARDS" =~ ^[0-9]+$ ]] && [ "$NSHARDS" -ge 1 ] && [ "$NSHARDS" -le 25 ] || die "NSHARDS=$NSHARDS"
[[ "$TIME" =~ ^[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]] || die "TIME=$TIME must be HH:MM:SS"
LVL_DIR=$RUNS/level$LEVEL
[ -e "$LVL_DIR" ] && die "level$LEVEL exists ($(ls $LVL_DIR | head -3 | tr '\n' ' ')); levels are add-only"
IDX_HIT=$(grep -lE "\"level\": $LEVEL[,}]" $RUNS/_index/*.jsonl 2>/dev/null | head -1)
[ -n "$IDX_HIT" ] && die "a campaign index records level $LEVEL ($IDX_HIT)"
md5sum -c --quiet $SRC_RUN/code.md5 >/dev/null 2>&1 || die "a file the source acquisition ran has changed since (md5sum -c $SRC_RUN/code.md5)"
TH_P=${THETA_PLANNING:-$THETA0}
[ "$(readlink -f ${THETA:-$THETA0})" = "$(readlink -f $THETA0)" ] || die "THETA must be the initial carrier SCRIBE_THETA0"
for th in "$THETA0" "$TH_P"; do [ -f "$th/meta.json" ] || die "carrier $th has no meta.json"; done
[ -f "$MODEL0/config.json" ] || die "model $MODEL0 has no config.json"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 $RUN_PY - "$LEVERS" "$LITREVIEW_ROOT" <<'PYEOF' || die "lever file refused"
import sys
lf, root = sys.argv[1:3]
sys.path[:0] = [f"{root}/scribe/harness/runners", f"{root}/scribe/levers"]
import writing_levers as LV
print(f"### levers {lf}: {LV.load(lf)}")
PYEOF
TASKS_SRC=$($RUN_PY -c "import json; print('\n'.join(sorted(json.load(open('$PINS'))['tasks'])))")
if [ -n "${TASKS_FILE:-}" ]; then
  TASKS_LIST=$(sort -u "$TASKS_FILE"); BADT=$(comm -23 <(echo "$TASKS_LIST") <(echo "$TASKS_SRC") | head -3)
  [ -z "$BADT" ] || die "TASKS_FILE has tasks outside the source: $BADT"
else TASKS_LIST=$TASKS_SRC; fi
ARM=same_pool_${TAG}_cap
mkdir -p "$RUNS" "$LOGS"
mkdir "$LVL_DIR" 2>/dev/null || die "level$LEVEL was created by someone else"
RUN=$SCRIBE_WORK_ROOT/runs/eval_${ARM}_$(date +%Y%m%dT%H%M%S)_$$
mkdir -p $RUN
echo "{\"arm\": \"$ARM\", \"tag\": \"$TAG\", \"run\": \"$RUN\", \"utc\": \"$(date -u +%FT%TZ)\", \"by\": \"run_same_pool.sh\"}" > $LVL_DIR/_level_claim_$(date +%s).json
cp -p "$LEVERS" $RUN/levers.json
{ cat $SRC_RUN/code.md5
  md5sum $L/same_pool_prepare.slurm $L/same_pool_shard.slurm $L/same_pool_prepare.py $L/generate.py $L/same_pool_finalize.py \
    $LITREVIEW_ROOT/scribe/levers/writing_levers.py $LITREVIEW_ROOT/scribe/serving/*.py $LITREVIEW_ROOT/scribe/retrieval/*.py \
    $LITREVIEW_ROOT/scribe/harness/runners/*.py $LITREVIEW_ROOT/scribe/harness/runners/egress_guard.sh \
    $LITREVIEW_ROOT/third_party/kvskill/kvskill/*.py $PINS $RUN/levers.json 2>/dev/null; } | awk '!seen[$2]++' > $RUN/code.md5
$RUN_PY - "$RUN" <<PYEOF || { echo "FATAL: cannot write config.json (level$LEVEL claimed; remove it only if nothing was submitted)"; exit 2; }
import hashlib, json, os, sys
run = sys.argv[1]
tasks = """$TASKS_LIST""".split()
n = min($NSHARDS, len(tasks))
shards = {str(i): tasks[i::n] for i in range(n)}
th = {"synthesis": os.path.realpath("$THETA0"), "planning": os.path.realpath("$TH_P"), "writing": os.path.realpath("$THETA0")}
md = {s: os.path.realpath("$MODEL0") for s in th}
servers = []
for s in ("synthesis", "planning", "writing"):
    for sv in servers:
        if sv["theta"] == th[s] and sv["model"] == md[s]:
            sv["stages"].append(s); break
    else:
        ident = th[s] == os.path.realpath("$THETA0")
        servers.append({"theta": th[s], "stages": [s], "model": md[s], "expect_identity": ident,
                        "gpu_mem_util": 0.90 if ident else float("${GPU_UTIL_REG:-0.72}")})
lf = f"{run}/levers.json"
cfg = {"schema": "same_pool_config/1", "level": $LEVEL, "tag": "$TAG", "arm": "$ARM", "src": "$SRC", "src_desc": "$SRC_DESC",
       "system": "$SYSTEM", "seed": $SEED, "src_audit_ok": "$SRC_AUDIT_OK", "src_run": "$SRC_RUN",
       "pins": "$PINS", "pins_sha256": "$PINS_SHA", "runs_root": "$RUNS", "tasks": tasks, "shards": shards,
       "workers": {k: min(int(os.environ.get("SCRIBE_MAX_WORKERS", "25")), len(v)) for k, v in shards.items()}, "carriers": th, "models": md,
       "servers": servers, "harness_code_from": "$LITREVIEW_ROOT",
       "levers_src": "$LEVERS", "levers": lf, "levers_md5": hashlib.md5(open(lf, "rb").read()).hexdigest(),
       "gen_maxlen": $GEN_MAXLEN, "time": "$TIME",
       "service_log": "$SRC_RUN/search_calls.jsonl", "service_log_sha256": "$SLOG_SHA",
       "user": os.environ.get("USER"), "submit_host": os.uname().nodename}
json.dump(cfg, open(f"{run}/config.json", "w"), indent=1)
print(f"### config: {len(tasks)} tasks in {n} shards {[len(v) for v in shards.values()]}; servers {[(s['stages'], s['expect_identity']) for s in servers]}")
PYEOF
NSRV=$($RUN_PY -c "import json; print(len(json.load(open('$RUN/config.json'))['servers']))")
NSH=$($RUN_PY -c "import json; print(len(json.load(open('$RUN/config.json'))['shards']))")
PREP=$(sbatch --parsable --time=00:45:00 --job-name=same_pool_prepare_${TAG:0:10} --output=$LOGS/same_pool_prepare_%j.log \
      --export=ALL,LITREVIEW_ROOT=$LITREVIEW_ROOT,SCRIBE_RUN_DIR=$RUN $L/same_pool_prepare.slurm) || { echo "FATAL: prepare submit failed; level$LEVEL claimed with nothing submitted"; exit 2; }
$RUN_PY -c "import json; p='$RUN/config.json'; c=json.load(open(p)); c['prepare_job']='$PREP'; json.dump(c, open(p,'w'), indent=1)"
md5sum $RUN/config.json > $RUN/config.md5
SJ=()
for i in $(seq 0 $((NSH-1))); do
  j=$(sbatch --parsable --dependency=afterok:$PREP --kill-on-invalid-dep=yes --gres=gpu:$NSRV --time=$TIME \
      --job-name=same_pool_${TAG:0:12}_s$i --output=$LOGS/same_pool_shard_%j.log \
      --export=ALL,LITREVIEW_ROOT=$LITREVIEW_ROOT,SCRIBE_RUN_DIR=$RUN,SCRIBE_SHARD=$i $L/same_pool_shard.slurm) || { echo "FATAL: shard $i submit failed"; exit 2; }
  SJ+=($j)
done
DEP=$(IFS=:; echo "${SJ[*]}")
FIN=$(sbatch --parsable --dependency=afterany:$DEP --kill-on-invalid-dep=yes --cpus-per-task=2 --mem=8G \
      --time=00:20:00 --job-name=same_pool_finalize_${TAG:0:12} --output=$LOGS/same_pool_finalize_%j.log \
      --export=ALL,LITREVIEW_ROOT=$LITREVIEW_ROOT,SCRIBE_RUN_DIR=$RUN \
      --wrap="md5sum -c --quiet $RUN/code.md5 && PYTHONNOUSERSITE=1 $RUN_PY $L/same_pool_finalize.py $RUN")
echo "{\"prepare\": \"$PREP\", \"shards\": \"${SJ[*]}\", \"finalize\": \"$FIN\"}" > $RUN/jobs.json
echo "### level$LEVEL claimed; run $RUN"
echo "### jobs: prepare $PREP, shards ${SJ[*]} (gres gpu:$NSRV each, time $TIME), finalize $FIN"
