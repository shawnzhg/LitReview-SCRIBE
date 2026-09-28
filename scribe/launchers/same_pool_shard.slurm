#!/bin/bash
# Slurm GPU job that runs one generation shard of a same-pool run: serves each stage's carrier with
# vLLM in a loopback-only namespace, checks the served carriers and runs the generation driver.
# Submitted by run_same_pool.sh.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=160G
set -uo pipefail
LITREVIEW_ROOT=${LITREVIEW_ROOT:?LITREVIEW_ROOT required (run_same_pool.sh passes it)}
[ -f "$LITREVIEW_ROOT/scribe/harness/runners/runner.py" ] || { echo "FATAL: LITREVIEW_ROOT=$LITREVIEW_ROOT is not the repo"; exit 2; }
for v in SCRIBE_WORK_ROOT SCRIBE_SERVE_PY SCRIBE_TRAIN_PY SCRIBE_RUN_PY SCRIBE_HF_HOME SCRIBE_MODEL_DIR SCRIBE_BUDGET_CAPS SCRIBE_KCAP; do
  [ -n "${!v:-}" ] || { echo "FATAL: $v is not set (configs/site.env.example)"; exit 2; }
done
SERVE_PY=$SCRIBE_SERVE_PY; LEARN_PY=$SCRIBE_TRAIN_PY; RUN_PY=$SCRIBE_RUN_PY; MODEL=$SCRIBE_MODEL_DIR
L=$LITREVIEW_ROOT/scribe/launchers; SERVING=$LITREVIEW_ROOT/scribe/serving; TOOLS=$LITREVIEW_ROOT/scribe/harness/tools
RUN=${SCRIBE_RUN_DIR:?SCRIBE_RUN_DIR required}; SHARD=${SCRIBE_SHARD:?SCRIBE_SHARD required}
SH=$RUN/shard_$SHARD
echo "### start=$(date) node=$(hostname) job=${SLURM_JOB_ID:-} gpus=${CUDA_VISIBLE_DEVICES:-none} run=$RUN shard=$SHARD"
echo "### launcher: $0 md5 $(md5sum "$0" | awk '{print $1}')"
[ -f "$RUN/config.json" ] || { echo "FATAL: $RUN/config.json missing"; exit 2; }
[ -e "$SH" ] && { echo "FATAL: $SH exists (a shard runs once)"; exit 2; }
mkdir -p $SH
cfg(){ $RUN_PY -c "import json,sys; c=json.load(open('$RUN/config.json')); v=eval(sys.argv[1], {'c': c, 'S': '$SHARD'}); print(v if not isinstance(v, (list, dict)) else json.dumps(v))" "$1"; }
LEVEL=$(cfg "c['level']"); ARM=$(cfg "c['arm']"); SEED=$(cfg "c['seed']"); GEN_MAXLEN=$(cfg "c['gen_maxlen']")
SYSTEM=$(cfg "c.get('system') or 'SCRIBE'"); RUNS=$(cfg "c['runs_root']")
TASKS=$(cfg "','.join(c['shards'][S])"); WORKERS=$(cfg "c['workers'][S]"); NSRV=$(cfg "len(c['servers'])")
LEVERS=$(cfg "c.get('levers') or ''")
[ -n "$TASKS" ] || { echo "FATAL: shard $SHARD has no tasks"; exit 2; }
[ -n "$LEVERS" ] || { echo "FATAL: the config names no lever file"; exit 2; }
echo "### level=$LEVEL arm=$ARM tasks=$TASKS workers=$WORKERS servers=$NSRV levers=$LEVERS"
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
[ "${#GPUS[@]}" -ge "$NSRV" ] || { echo "FATAL: $NSRV servers need $NSRV GPUs, job has ${#GPUS[@]} (${CUDA_VISIBLE_DEVICES:-})"; exit 2; }

if [ -n "${SCRIBE_TOOLCHAIN:-}" ]; then source "$SCRIBE_TOOLCHAIN" || { echo "FATAL: toolchain ($SCRIBE_TOOLCHAIN) failed"; exit 99; }; fi
[ -n "${CUDA_HOME:-}" ] || { echo "FATAL: CUDA_HOME is not set (set it or SCRIBE_TOOLCHAIN)"; exit 99; }

md5sum -c --quiet $RUN/code.md5 >/dev/null 2>&1 || { echo "FATAL: code drift since submit:"; md5sum -c $RUN/code.md5 2>&1 | grep -v ': OK$' | head; exit 88; }
echo "### code: $(wc -l < $RUN/code.md5) files == submit-time md5s"
$RUN_PY - "$RUN/prepare_record.json" "$LEVEL" <<'PYEOF' || { echo "FATAL: the level's prepare record is not ok"; exit 2; }
import json, sys
r = json.load(open(sys.argv[1]))
ok = r.get("ok") is True and int(r.get("level", -1)) == int(sys.argv[2])
print(f"### prepare record: ok={r.get('ok')} level={r.get('level')} audits={r.get('audits_ok')} job={r.get('job')}")
sys.exit(0 if ok else 1)
PYEOF
echo "$TASKS" | tr ',' '\n' > $SH/tasks.txt
$RUN_PY $L/same_pool_prepare.py verify --provenance $RUN/provenance.json --tasks $SH/tasks.txt --phase pre --out $SH/verify_pre.json || { echo "FATAL: verify pre refused"; exit 3; }
POOL_SNAPSHOT_ID=$($RUN_PY -c 'import sys; sys.path.insert(0, sys.argv[1]); import pool_retrieval_tool as P; print(P.POOL_SNAPSHOT_ID)' "$TOOLS") || { echo "FATAL: cannot read the frozen-pool snapshot id"; exit 2; }

export PYTHONNOUSERSITE=1 HF_HOME=$SCRIBE_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$LITREVIEW_ROOT/third_party/kvskill:$SERVING
export PATH=$(dirname $SERVE_PY):$PATH
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 FLASHINFER_WORKSPACE_BASE=$SCRIBE_WORK_ROOT/jit_cache
for k in $(seq 0 $((NSRV-1))); do
  TH=$(cfg "c['servers'][$k]['theta']")
  $LEARN_PY -m kvskill.export --artifact_path $TH --out $SH/payload_$k.safetensors 2>&1 | tail -2
  [ -s $SH/payload_$k.safetensors ] || { echo "FATAL: carrier export failed for $TH"; exit 91; }
  $RUN_PY -c "import json; m=json.load(open('$TH/meta.json')); print(f\"{m['cn_ds']},{m['cn_dm']},{m['cn_r']},{m['cn_nlayers']}\")" > $SH/shape_$k.txt || exit 91
  $LEARN_PY -c "
from safetensors import safe_open; f=safe_open('$SH/payload_$k.safetensors','pt'); import torch; print(float(f.get_tensor('cn_v').float().norm()))" 2>&1 | tail -1 > $SH/vn_$k.txt
  echo "### server $k: theta=$TH shape=$(cat $SH/shape_$k.txt) ||V||=$(cat $SH/vn_$k.txt) stages=$(cfg "c['servers'][$k]['stages']") payload sha256 $(sha256sum < $SH/payload_$k.safetensors | cut -c1-16)"
done
BUDGET_CAPS_SHA256=$(sha256sum < $SCRIBE_BUDGET_CAPS | cut -c1-64); KCAP_SHA256=$(sha256sum < $SCRIBE_KCAP | cut -c1-64)
export RUN SH LEVEL ARM SEED GEN_MAXLEN TASKS WORKERS NSRV LEVERS MODEL SERVE_PY LEARN_PY RUN_PY SHARD SYSTEM RUNS L SERVING POOL_SNAPSHOT_ID LITREVIEW_ROOT
export GPU_LIST="${CUDA_VISIBLE_DEVICES:-}"
export SCRIBE_BUDGET_CAPS_SHA256=$BUDGET_CAPS_SHA256 SCRIBE_KCAP_SHA256=$KCAP_SHA256
unset POOL_URL POOL_CUTOFFS POOL_INDEX POOL_BOUNDARY POOL_LOG POOL_EDGES POOL_ALLOWLIST_DIR SCRIBE_RENDEZVOUS SCRIBE_RENDEZVOUS_WRITING SCRIBE_RENDEZVOUS_PLANNING SCRIBE_LEVERS

cat > $SH/inner.sh <<'INNER'
#!/bin/bash
set -uo pipefail
PIDS=(); cleanup(){ for p in "${PIDS[@]:-}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; }; trap cleanup EXIT
IFS=',' read -ra GPUS <<< "$GPU_LIST"
cfg(){ $RUN_PY -c "import json,sys; c=json.load(open('$RUN/config.json')); print(json.dumps(eval(sys.argv[1], {'c': c})))" "$1"; }
for k in $(seq 0 $((NSRV-1))); do
  port=$((8401 + k)); log=$SH/server_$k.log
  MODEL_K=$($RUN_PY -c "import json; c=json.load(open('$RUN/config.json')); print(c['servers'][$k].get('model') or '$MODEL')")
  UTIL_K=$($RUN_PY -c "import json; c=json.load(open('$RUN/config.json')); print(c['servers'][$k].get('gpu_mem_util') or 0.90)")
  echo "### server $k model $MODEL_K gpu_mem_util $UTIL_K"
  CUDA_VISIBLE_DEVICES=${GPUS[$k]} KVSKILL_CN_SHAPE=$(cat $SH/shape_$k.txt) KVSKILL_CN_PATH=$SH/payload_$k.safetensors setsid $SERVE_PY -m vllm.entrypoints.openai.api_server \
    --model $MODEL_K --served-model-name qwen38 --host 127.0.0.1 --port $port --dtype bfloat16 \
    --max-model-len $GEN_MAXLEN --gpu-memory-utilization $UTIL_K --max-num-seqs 48 --seed 0 \
    --worker-extension-cls carrier_digest_ext.DigestWorkerExtension > $log 2>&1 &
  PIDS+=($!)
done
for k in $(seq 0 $((NSRV-1))); do
  port=$((8401 + k)); log=$SH/server_$k.log; up=0
  for i in $(seq 1 400); do curl -sf -o /dev/null "http://127.0.0.1:$port/v1/models" 2>/dev/null && { up=1; echo "  vllm $k UP (~$((i*5))s)"; break; }; kill -0 "${PIDS[$k]}" 2>/dev/null || { echo "  vllm $k process exited"; break; }; sleep 5; done
  [ $up = 1 ] || { echo "FATAL: vllm $k never came up"; tail -30 $log; exit 94; }
  grep -q "\[cn_vllm\] staged" $log || { echo "FATAL: server $k: carrier NOT staged (patch inactive)"; exit 95; }
  srv=$(grep -oE '(‖V‖|\|\|V\|\|)=[0-9.]+' $log | head -1 | sed 's/.*=//')
  [ -n "$srv" ] || { echo "FATAL: server $k log reports no carrier norm"; exit 95; }
  ident=$(cfg "c['servers'][$k]['expect_identity']")
  $RUN_PY -c "
import sys
srv, art = float('$srv'), float(open('$SH/vn_$k.txt').read().strip())
if abs(srv - art) > 1e-3: sys.exit(f'FATAL: server $k staged norm {srv} != exported {art}')
if srv == 0.0 and '$ident' != 'true': sys.exit('FATAL: server $k staged the identity carrier but its carrier is not SCRIBE_THETA0')
if srv != 0.0 and '$ident' == 'true': sys.exit(f'FATAL: server $k: the initial carrier SCRIBE_THETA0 expected, staged norm {srv}')
print(f'### carrier verified (server $k): staged norm {srv} == exported {art}')" || exit 95
  $RUN_PY - "$SH/rendezvous_$k.json" "$k" "$port" <<'PYEOF' || exit 95
import hashlib, json, os, sys
out, k, port = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = json.load(open(os.environ["RUN"] + "/config.json")); s = c["servers"][k]; sh = os.environ["SH"]
m = s.get("model") or os.environ["MODEL"]
rdv = {"host": "127.0.0.1", "port": port, "served_model": "qwen38", "model_path": m, "max_model_len": int(os.environ["GEN_MAXLEN"]),
       "job_id": os.environ.get("SLURM_JOB_ID"), "url": f"http://127.0.0.1:{port}/v1", "kvskill_theta": s["theta"],
       "kvskill_payload": f"{sh}/payload_{k}.safetensors", "kvskill_shape": open(f"{sh}/shape_{k}.txt").read().strip(),
       "kvskill_v_norm": float(open(f"{sh}/vn_{k}.txt").read().strip()),
       "chat_template_sha256_16": hashlib.sha256(open(f"{m}/chat_template.jinja", "rb").read()).hexdigest()[:16],
       "entry_mode": "native_chain", "retrieval_backend": "pool", "phase": "generation",
       "retrieval_budget": "cap", "budget_caps_sha256": os.environ["SCRIBE_BUDGET_CAPS_SHA256"],
       "corpus_snapshot_id": os.environ["POOL_SNAPSHOT_ID"], "stages": s["stages"], "server": k,
       "gpu": os.environ["GPU_LIST"].split(",")[k]}
json.dump(rdv, open(out, "w"), indent=1)
PYEOF
done
$LEARN_PY $SERVING/route_check.py --config $RUN/config.json --sh $SH || { echo "FATAL: carrier route check refused"; exit 95; }
srv_of(){ $RUN_PY -c "import json; c=json.load(open('$RUN/config.json')); print([k for k, s in enumerate(c['servers']) if '$1' in s['stages']][0])"; }
KS=$(srv_of synthesis); KP=$(srv_of planning); KW=$(srv_of writing)
export SCRIBE_RENDEZVOUS="$(cat $SH/rendezvous_$KS.json)"
[ "$KP" != "$KS" ] && export SCRIBE_RENDEZVOUS_PLANNING="$(cat $SH/rendezvous_$KP.json)"
[ "$KW" != "$KS" ] && export SCRIBE_RENDEZVOUS_WRITING="$(cat $SH/rendezvous_$KW.json)"
echo "### routing: synthesis -> server $KS, planning -> server $KP, writing -> server $KW"
export SCRIBE_LEVERS=$LEVERS
export SCRIBE_DRIVER_RECORD=$SH/driver.json
md5sum -c --quiet $RUN/code.md5 >/dev/null 2>&1 || { echo "FATAL: code drift before generation"; exit 88; }
cd $SH
echo "### generation start $(date)"
$RUN_PY $L/generate.py --phase generation --systems $SYSTEM --mode native_chain --level $LEVEL --seeds $SEED \
  --tasks-file $SH/tasks.txt --workers $WORKERS --campaign "${ARM}_gen_s${SHARD}" 2>&1 | tail -80
RC_GEN=${PIPESTATUS[0]}
echo "### generation end $(date) rc=$RC_GEN"
[ "$RC_GEN" = 86 ] && exit 86
[ "$RC_GEN" = 0 ] || exit 92
exit 0
INNER
chmod +x $SH/inner.sh
$LITREVIEW_ROOT/scribe/harness/runners/egress_guard.sh "$SH/egress_proof.json" bash $SH/inner.sh; RC=$?
echo "### inner rc=$RC"
$RUN_PY $L/same_pool_prepare.py verify --provenance $RUN/provenance.json --tasks $SH/tasks.txt --phase post --out $SH/verify_post.json; RCV=$?
$RUN_PY - "$SH" "$RC" "$RCV" "$LEVEL" "$RUNS" <<'PYEOF'
import glob, hashlib, json, os, sys
sh, rc, rcv, lvl, R = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
def lj(p):
    try: return json.load(open(p))
    except Exception: return None
def sha(p):
    try: return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except Exception: return None
tasks = open(f"{sh}/tasks.txt").read().split()
n_rep = sum(os.path.exists(f"{R}/level{lvl}/{os.environ.get('SYSTEM') or 'SCRIBE'}/native_chain/{t}/seed{os.environ['SEED']}/report_artifact.json") for t in tasks)
k = 0; servers = []
while os.path.exists(f"{sh}/rendezvous_{k}.json") or os.path.exists(f"{sh}/payload_{k}.safetensors"):
    servers.append({"rendezvous": lj(f"{sh}/rendezvous_{k}.json"), "payload_sha256": sha(f"{sh}/payload_{k}.safetensors")}); k += 1
rec = {"schema": "same_pool_shard/1", "job": os.environ.get("SLURM_JOB_ID"), "rc": rc, "verify_post_rc": rcv, "level": int(lvl),
       "tasks": tasks, "n_reports": n_rep, "servers": servers, "driver": lj(f"{sh}/driver.json"),
       "ok": rc == 0 and rcv == 0 and n_rep == len(tasks)}
json.dump(rec, open(f"{sh}/shard_record.json", "w"), indent=1)
print(f"### shard record: ok={rec['ok']} rc={rc} verify_post={rcv} reports {n_rep}/{len(tasks)}")
PYEOF
echo "### end=$(date) rc=$RC"
[ "$RC" = 0 ] && [ "$RCV" != 0 ] && exit 3
exit $RC
