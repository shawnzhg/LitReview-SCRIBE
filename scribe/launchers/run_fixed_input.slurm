#!/bin/bash
# Slurm GPU launcher that runs one SCRIBE arm under fixed input, serving each stage's carrier inside the
# egress guard. Usage: THETA=<carrier> TAG=<name> LEVEL=<n> [SEEDS=0|0,1,2] sbatch --gres=gpu:<servers>
# --export=ALL run_fixed_input.slurm.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=160G --gres=gpu:1
#SBATCH --time=06:00:00
set -uo pipefail
case "${1:-}" in -h|--help) awk 'NR>1 && /^#SBATCH/ {next} NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0;; esac
LITREVIEW_ROOT=${LITREVIEW_ROOT:-$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)}
[ -f "$LITREVIEW_ROOT/scribe/harness/runners/runner.py" ] && [ -f "$LITREVIEW_ROOT/scribe/levers/writing_levers.json" ] \
  || { echo "FATAL: set LITREVIEW_ROOT to the repo (configs/site.env.example)"; exit 2; }
for v in SCRIBE_WORK_ROOT SCRIBE_SERVE_PY SCRIBE_TRAIN_PY SCRIBE_RUN_PY SCRIBE_HF_HOME SCRIBE_MODEL_DIR SCRIBE_THETA0 SCRIBE_RUNS_ROOT SCRIBE_EVAL_TASKS; do
  [ -n "${!v:-}" ] || { echo "FATAL: $v is not set (configs/site.env.example)"; exit 2; }
done
SERVE_PY=$SCRIBE_SERVE_PY; LEARN_PY=$SCRIBE_TRAIN_PY; RUN_PY=$SCRIBE_RUN_PY; MODEL=$SCRIBE_MODEL_DIR; THETA0=$SCRIBE_THETA0
RUNNERS=$LITREVIEW_ROOT/scribe/harness/runners; SERVING=$LITREVIEW_ROOT/scribe/serving; RUNS=$SCRIBE_RUNS_ROOT
DRIVER=$LITREVIEW_ROOT/scribe/launchers/generate.py; LEVERS=$LITREVIEW_ROOT/scribe/levers/writing_levers.json
THETA=${THETA:?THETA artifact dir required}; TAG=${TAG:?TAG required}
MODE=bundle_entry; LEVEL=${LEVEL:?LEVEL required (a fresh level)}
[[ "$LEVEL" =~ ^[0-9]+$ ]] && [ "$LEVEL" -ge 1 ] || { echo "FATAL: LEVEL=$LEVEL must be a positive integer"; exit 2; }
SYSTEMS=${SYSTEMS:-SCRIBE}; SEEDS=${SEEDS:-0}; WORKERS=${WORKERS:-8}; NTASKS=${NTASKS:-50}
PORT=${PORT:-$(( 8400 + ${SLURM_JOB_ID:-0} % 500 ))}; MAXLEN=65536
[[ "$SEEDS" =~ ^[012](,[012])*$ ]] && [ "$(tr ',' '\n' <<< "$SEEDS" | sort | uniq -d)" = "" ] || { echo "FATAL: SEEDS=$SEEDS must be distinct seeds from 0, 1, 2"; exit 2; }
[[ "$SYSTEMS" =~ ^[A-Za-z0-9_]+$ ]] || { echo "FATAL: SYSTEMS=$SYSTEMS must name exactly one system (SCRIBE)"; exit 2; }
[[ "$TAG" =~ ^[A-Za-z0-9_]+$ ]] || { echo "FATAL: TAG must be [A-Za-z0-9_]+"; exit 2; }
ARM=fixed_input_${TAG}
[ -e "$RUNS/level$LEVEL" ] && { echo "FATAL: level$LEVEL already exists ($(ls $RUNS/level$LEVEL | head -3 | tr '\n' ' ')); a fixed-input arm needs a fresh level"; exit 2; }
IDX_HIT=$(grep -lE "\"level\": $LEVEL[,}]" $RUNS/_index/*.jsonl 2>/dev/null | head -1)
[ -z "$IDX_HIT" ] || { echo "FATAL: a campaign index records runs at level $LEVEL ($IDX_HIT); pick a fresh level"; exit 2; }
RUN=$SCRIBE_WORK_ROOT/runs/eval_${ARM}_${SLURM_JOB_ID:-local$$}
export PYTHONNOUSERSITE=1 HF_HOME=$SCRIBE_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$LITREVIEW_ROOT/third_party/kvskill
export PATH=$(dirname $SERVE_PY):$PATH
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 FLASHINFER_WORKSPACE_BASE=$SCRIBE_WORK_ROOT/jit_cache
if [ -n "${SCRIBE_TOOLCHAIN:-}" ]; then source "$SCRIBE_TOOLCHAIN" || { echo "FATAL: toolchain ($SCRIBE_TOOLCHAIN) failed -- exit 99, before any GPU work or server start"; exit 99; }; fi
[ -n "${CUDA_HOME:-}" ] || { echo "FATAL: CUDA_HOME is not set (set it or SCRIBE_TOOLCHAIN)"; exit 99; }
mkdir -p $RUN $SCRIBE_WORK_ROOT/logs || exit 2
md5sum $RUNNERS/*.py $RUNNERS/egress_guard.sh $SERVING/*.py $DRIVER $LITREVIEW_ROOT/scribe/levers/writing_levers.py $LEVERS \
  $LITREVIEW_ROOT/biolitbench/schemas/*.json $LITREVIEW_ROOT/biolitbench/tasks/build_canonical_bundles.py \
  $LITREVIEW_ROOT/third_party/kvskill/kvskill/*.py > $RUN/code.md5 || { echo "FATAL: cannot fingerprint the code"; exit 2; }
echo "### code recorded at start: $(wc -l < $RUN/code.md5) files ($RUN/code.md5); levers $LEVERS md5 $(md5sum < $LEVERS | cut -c1-32)"
echo "### start=$(date) node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-} theta=$THETA mode=$MODE level=$LEVEL tag=$TAG arm=$ARM seeds=$SEEDS ntasks=$NTASKS"
echo "### launcher: $0 md5 $(md5sum "$0" 2>/dev/null | awk '{print $1}')"
PIDS=(); cleanup(){ for p in "${PIDS[@]:-}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; }; trap cleanup EXIT
wait_http(){ for i in $(seq 1 "$3"); do curl -sf -o /dev/null "$1" 2>/dev/null && { echo "  $2 UP (~$((i*5))s)"; return 0; }; sleep 5; done; echo "FATAL: $2 never came up"; return 1; }

cd $RUN
$RUN_PY $LITREVIEW_ROOT/biolitbench/tasks/build_canonical_bundles.py --check 2>&1 | tail -3
[ "${PIPESTATUS[0]}" = 0 ] || { echo "FATAL: canonical bundles missing or drifted (biolitbench/tasks/build_canonical_bundles.py --check)"; exit 90; }

THETA_PLANNING=${THETA_PLANNING:-$THETA0}
GPU_UTIL_REG=${GPU_UTIL_REG:-0.72}
[ "$(readlink -f $THETA)" = "$(readlink -f $THETA0)" ] || { echo "FATAL: fixed input: THETA must be the initial carrier SCRIBE_THETA0"; exit 2; }
for th in "$THETA0" "$THETA_PLANNING"; do [ -f "$th/meta.json" ] || { echo "FATAL: fixed input: carrier $th has no meta.json"; exit 2; }; done
[ -f "$MODEL/config.json" ] || { echo "FATAL: fixed input: SCRIBE_MODEL_DIR=$MODEL has no config.json"; exit 2; }
SH=$RUN/srv; mkdir -p $SH
$RUN_PY - "$RUN/config.json" "$LEVEL" "$ARM" "$SEEDS" "$MAXLEN" "$THETA0" "$MODEL" "$THETA_PLANNING" "$GPU_UTIL_REG" <<'PYEOF' || { echo "FATAL: fixed input: config"; exit 2; }
import json, os, sys
out, level, arm, seeds, maxlen, th0, m0, thp, util = sys.argv[1:10]
rp = os.path.realpath
th = {"synthesis": rp(th0), "planning": rp(thp), "writing": rp(th0)}
md = {"synthesis": rp(m0), "planning": rp(m0), "writing": rp(m0)}
servers = []
for s in ("synthesis", "planning", "writing"):
    for sv in servers:
        if sv["theta"] == th[s] and sv["model"] == md[s]:
            sv["stages"].append(s); break
    else:
        ident = th[s] == rp(th0)
        servers.append({"theta": th[s], "stages": [s], "model": md[s], "expect_identity": ident,
                        "gpu_mem_util": 0.90 if ident else float(util)})
cfg = {"schema": "fixed_input_config/1", "level": int(level), "arm": arm, "seeds": [int(x) for x in seeds.split(",")], "gen_maxlen": int(maxlen),
       "carriers": th, "models": md, "servers": servers}
json.dump(cfg, open(out, "w"), indent=1)
print(f"### fixed-input config: {len(servers)} server(s): {[(s['stages'], os.path.basename(os.path.dirname(s['theta'])) + '/' + os.path.basename(s['theta']), s['gpu_mem_util']) for s in servers]}")
PYEOF
NSRV=$($RUN_PY -c "import json; print(len(json.load(open('$RUN/config.json'))['servers']))")
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
[ "${#GPUS[@]}" -ge "$NSRV" ] || { echo "FATAL: fixed input: $NSRV servers need $NSRV GPUs, job has ${#GPUS[@]} (${CUDA_VISIBLE_DEVICES:-}); submit with --gres=gpu:$NSRV"; exit 2; }
export PYTHONPATH=$LITREVIEW_ROOT/third_party/kvskill:$SERVING
cfgv(){ $RUN_PY -c "import json,sys; c=json.load(open('$RUN/config.json')); print(json.dumps(eval(sys.argv[1], {'c': c})))" "$1"; }
PORT0=$(( 8400 + (${SLURM_JOB_ID:-0} % 166) * 3 ))
for k in $(seq 0 $((NSRV-1))); do
  TH=$($RUN_PY -c "import json; print(json.load(open('$RUN/config.json'))['servers'][$k]['theta'])")
  $LEARN_PY -m kvskill.export --artifact_path $TH --out $SH/payload_$k.safetensors 2>&1 | tail -2
  [ -s $SH/payload_$k.safetensors ] || { echo "FATAL: carrier export failed for $TH"; exit 91; }
  $RUN_PY -c "import json; m=json.load(open('$TH/meta.json')); print(f\"{m['cn_ds']},{m['cn_dm']},{m['cn_r']},{m['cn_nlayers']}\")" > $SH/shape_$k.txt || exit 91
  $LEARN_PY -c "
from safetensors import safe_open; f=safe_open('$SH/payload_$k.safetensors','pt'); import torch; print(float(f.get_tensor('cn_v').float().norm()))" 2>&1 | tail -1 > $SH/vn_$k.txt
  echo "### server $k: theta=$TH shape=$(cat $SH/shape_$k.txt) ||V||=$(cat $SH/vn_$k.txt) payload sha256 $(sha256sum < $SH/payload_$k.safetensors | cut -c1-16)"
done
TASKS=$($RUN_PY -c "import json; print(','.join(sorted(json.load(open('$SCRIBE_EVAL_TASKS'))['tasks'])[:$NTASKS]))")
TASKS_FILE=${TASKS_FILE:-}; [ -z "$TASKS_FILE" ] || [ -f "$TASKS_FILE" ] || { echo "FATAL: TASKS_FILE=$TASKS_FILE is not a file"; exit 2; }
export RUN SH NSRV PORT0 MAXLEN MODE LEVEL SEEDS WORKERS SYSTEMS ARM TASKS TASKS_FILE RUNS SERVE_PY LEARN_PY RUN_PY DRIVER LEVERS SERVING LITREVIEW_ROOT
export GPU_LIST="${CUDA_VISIBLE_DEVICES:-}"
cat > $RUN/inner.sh <<'INNER'
#!/bin/bash
set -uo pipefail
PIDS=(); cleanup(){ for p in "${PIDS[@]:-}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; }; trap cleanup EXIT
wait_http(){ for i in $(seq 1 "$3"); do curl -sf -o /dev/null "$1" 2>/dev/null && { echo "  $2 UP (~$((i*5))s)"; return 0; }; sleep 5; done; echo "FATAL: $2 never came up"; return 1; }
IFS=',' read -ra GPUS <<< "$GPU_LIST"
cfgv(){ $RUN_PY -c "import json,sys; c=json.load(open('$RUN/config.json')); print(json.dumps(eval(sys.argv[1], {'c': c})))" "$1"; }
SPIDS=()
for k in $(seq 0 $((NSRV-1))); do
  port=$((PORT0 + k)); log=$SH/server_$k.log
  MODEL_K=$($RUN_PY -c "import json; print(json.load(open('$RUN/config.json'))['servers'][$k]['model'])")
  UTIL_K=$($RUN_PY -c "import json; print(json.load(open('$RUN/config.json'))['servers'][$k]['gpu_mem_util'])")
  echo "### server $k model $MODEL_K gpu_mem_util $UTIL_K port $port gpu ${GPUS[$k]}"
  CUDA_VISIBLE_DEVICES=${GPUS[$k]} KVSKILL_CN_SHAPE=$(cat $SH/shape_$k.txt) KVSKILL_CN_PATH=$SH/payload_$k.safetensors setsid $SERVE_PY -m vllm.entrypoints.openai.api_server \
    --model $MODEL_K --served-model-name qwen38 --host 127.0.0.1 --port $port --dtype bfloat16 \
    --max-model-len $MAXLEN --gpu-memory-utilization $UTIL_K --max-num-seqs 48 --seed 0 \
    --worker-extension-cls carrier_digest_ext.DigestWorkerExtension > $log 2>&1 &
  SPIDS+=($!); PIDS+=($!)
done
for k in $(seq 0 $((NSRV-1))); do
  port=$((PORT0 + k)); log=$SH/server_$k.log
  wait_http "http://127.0.0.1:$port/v1/models" "vllm $k" 400 || { tail -30 $log; exit 94; }
  sleep 5; kill -0 "${SPIDS[$k]}" 2>/dev/null || { echo "FATAL: fixed input: server $k process is not alive after UP (lost its port?)"; tail -20 $log; exit 94; }
  grep -qE "Errno 98|address already in use" $log && { echo "FATAL: fixed input: server $k could not bind port $port"; exit 94; }
  grep -q "\[cn_vllm\] staged" $log || { echo "FATAL: server $k: carrier NOT staged (patch inactive)"; exit 95; }
  srv=$(grep -oE '(‖V‖|\|\|V\|\|)=[0-9.]+' $log | head -1 | sed 's/.*=//')
  [ -n "$srv" ] || { echo "FATAL: server $k log reports no carrier norm"; exit 95; }
  ident=$(cfgv "c['servers'][$k]['expect_identity']")
  $RUN_PY -c "
import sys
srv, art = float('$srv'), float(open('$SH/vn_$k.txt').read().strip())
if abs(srv - art) > 1e-3: sys.exit(f'FATAL: server $k staged norm {srv} != exported {art}')
if srv == 0.0 and '$ident' != 'true': sys.exit('FATAL: server $k staged the identity carrier but its carrier is not SCRIBE_THETA0')
if srv != 0.0 and '$ident' == 'true': sys.exit(f'FATAL: server $k: the initial carrier SCRIBE_THETA0 expected, staged norm {srv}')
print(f'### carrier verified (server $k): staged norm {srv} == exported {art}')" || exit 95
  $RUN_PY - "$SH/rendezvous_$k.json" "$k" "$port" "$RUN/config.json" "$SH" "$MAXLEN" "$MODE" <<'PYEOF' || exit 95
import hashlib, json, os, sys
out, k, port, cfgp, sh, maxlen, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), *sys.argv[4:8]
s = json.load(open(cfgp))["servers"][k]; m = s["model"]
rdv = {"host": "127.0.0.1", "port": port, "served_model": "qwen38", "model_path": m, "max_model_len": int(maxlen),
       "job_id": os.environ.get("SLURM_JOB_ID"), "url": f"http://127.0.0.1:{port}/v1", "kvskill_theta": s["theta"],
       "kvskill_payload": f"{sh}/payload_{k}.safetensors", "kvskill_shape": open(f"{sh}/shape_{k}.txt").read().strip(),
       "kvskill_v_norm": float(open(f"{sh}/vn_{k}.txt").read().strip()),
       "chat_template_sha256_16": hashlib.sha256(open(f"{m}/chat_template.jinja", "rb").read()).hexdigest()[:16],
       "entry_mode": mode, "stages": s["stages"], "server": k}
json.dump(rdv, open(out, "w"), indent=1)
PYEOF
done
$LEARN_PY $SERVING/route_check.py --config $RUN/config.json --sh $SH || { echo "FATAL: carrier route check refused"; exit 95; }
srv_of(){ $RUN_PY -c "import json; c=json.load(open('$RUN/config.json')); print([k for k, s in enumerate(c['servers']) if '$1' in s['stages']][0])"; }
KS=$(srv_of synthesis); KP=$(srv_of planning); KW=$(srv_of writing)
export SCRIBE_RENDEZVOUS="$(cat $SH/rendezvous_$KS.json)"
unset SCRIBE_RENDEZVOUS_PLANNING SCRIBE_RENDEZVOUS_WRITING
[ "$KP" != "$KS" ] && export SCRIBE_RENDEZVOUS_PLANNING="$(cat $SH/rendezvous_$KP.json)"
[ "$KW" != "$KS" ] && export SCRIBE_RENDEZVOUS_WRITING="$(cat $SH/rendezvous_$KW.json)"
echo "### routing: synthesis -> server $KS, planning -> server $KP, writing -> server $KW"

cd $RUN
md5sum -c --quiet $RUN/code.md5 >/dev/null 2>&1 || { echo "FATAL: the harness code changed since job start:"; md5sum -c $RUN/code.md5 2>&1 | grep -v ': OK$' | head; exit 88; }
SCRIBE_LEVERS=$LEVERS SCRIBE_DRIVER_RECORD=$RUN/driver_main.json $RUN_PY $DRIVER --phase generation --systems $SYSTEMS --mode $MODE \
  --level $LEVEL --seeds $SEEDS $([ -n "${TASKS_FILE:-}" ] && echo "--tasks-file $TASKS_FILE" || echo "--tasks $TASKS") --workers $WORKERS \
  --campaign "$ARM" 2>&1 | tail -70
RC_GEN=${PIPESTATUS[0]}
echo "### runs: $(ls -d $RUNS/level$LEVEL/$SYSTEMS/$MODE/*/seed* 2>/dev/null | wc -l) dirs; index: $RUNS/_index/${ARM}.jsonl"
echo "### delivery: ok=$(grep -c '"delivery_ok": true' $RUNS/_index/${ARM}.jsonl 2>/dev/null) failed=$(grep -c 'harness_failure:bundle_delivery' $RUNS/_index/${ARM}.jsonl 2>/dev/null) retrieval_reachable=$(grep -c 'harness_failure:retrieval_reachable' $RUNS/_index/${ARM}.jsonl 2>/dev/null)"
[ "$RC_GEN" = 0 ] || { echo "FATAL: the generation driver exited rc=$RC_GEN"; exit 92; }
exit 0
INNER
chmod +x $RUN/inner.sh
$RUNNERS/egress_guard.sh "$RUN/egress_proof.json" bash $RUN/inner.sh; RC_INNER=$?
echo "### inner rc=$RC_INNER"
[ "$RC_INNER" = 0 ] || { echo "FATAL: serving / generation inside the egress guard failed (rc $RC_INNER: 4/5/6 egress_guard refused the namespace, 94/95 server / carrier checks, 92 generation)"; exit $RC_INNER; }
echo "### end=$(date) level=$LEVEL arm=$ARM"
exit 0
