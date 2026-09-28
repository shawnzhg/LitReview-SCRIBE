#!/bin/bash
# Runs inside the network guard: serves DR-Tulu with vLLM, starts the pool service and runs each
# task once through drtulu_task.py behind a recording proxy.
set -uo pipefail
for v in LITREVIEW_ROOT CAMP_ROOT DRTULU_MODEL_DIR DRTULU_AGENT_DIR BASELINE_INPUTS TASKLIST CUTOFFS POOL_INDEX POOL_GOLD_DIR; do [ -n "${!v:-}" ] || { echo "FATAL: $v is not set"; exit 95; }; done
B="$LITREVIEW_ROOT/scribe/harness/baselines"
source "$B/drtulu_env.sh"
DT8B=$DRTULU_MODEL_DIR
CAMP=$CAMP_ROOT
LOGS="$CAMP/_logs"
mkdir -p "$CAMP" "$LOGS"

PIDS=()
cleanup(){ for p in "${PIDS[@]:-}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

wait_http(){ for i in $(seq 1 "$3"); do
    timeout 3 curl -sS -o /dev/null "$1" 2>/dev/null && { echo "  $2 UP (~$((i*5))s)"; return 0; }
    sleep 5; done; echo "  FATAL: $2 never came up"; return 1; }

VLLM_CMD=("$SERVE_PY" -m vllm.entrypoints.openai.api_server --model "$DT8B"
          --served-model-name DR-Tulu-8B --host 127.0.0.1 --port "$VPORT"
          --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.85)
setsid "${VLLM_CMD[@]}" > $LOGS/camp_vllm_shard${SHARD}.log 2>&1 &
PIDS+=($!)
T_V0=$(date +%s.%N); wait_http "http://127.0.0.1:$VPORT/health" vLLM 700 || { tail -15 $LOGS/camp_vllm_shard${SHARD}.log; exit 1; }

$SERVE_PY "$B/serving_proof.py" \
  --out "$CAMP/serving_proof_shard${SHARD}.json" --model-dir "$DT8B" --python "$SERVE_PY" \
  --vllm-log $LOGS/camp_vllm_shard${SHARD}.log --serve-cmd "${VLLM_CMD[*]}" \
  --repo "$LITREVIEW_ROOT" --repo "$DRTULU_AGENT_DIR" || { echo "FATAL: serving proof refused this shard"; exit 3; }
VLLM_STARTUP_S=$($POOL_PY -c "print(f\"{$(date +%s.%N)-$T_V0:.1f}\")" 2>/dev/null)
echo "  serving proof OK (vllm startup ${VLLM_STARTUP_S}s)"

$POOL_PY -c "import sys; sys.path.insert(0,'$POOL_PYTHONPATH'); import Stemmer, bm25s" 2>/dev/null \
  || { echo "FATAL: pool interpreter cannot import Stemmer/bm25s"; exit 4; }
echo "  pool libs OK"

export POOL_LOG="$CAMP/search_calls_shard${SHARD}.jsonl"
setsid env PYTHONPATH="$POOL_PYTHONPATH" PYTHONNOUSERSITE=1 $POOL_PY "$LITREVIEW_ROOT/biolitbench/pool/pool_service.py" --host 127.0.0.1 --port $POOLPORT \
  --index "$POOL_INDEX" --cutoffs "$CUTOFFS" --gold "$POOL_GOLD_DIR" --log "$POOL_LOG" \
  > $LOGS/camp_pool_shard${SHARD}.log 2>&1 &
PIDS+=($!)
T_P0=$(date +%s.%N); wait_http "http://127.0.0.1:$POOLPORT/healthz" pool 200 || { tail -15 $LOGS/camp_pool_shard${SHARD}.log; exit 1; }

POOL_STARTUP_S=$($POOL_PY -c "print(f\"{$(date +%s.%N)-$T_P0:.1f}\")" 2>/dev/null)

LO=${SLICE%-*}; HI=${SLICE#*-}
mapfile -t TASKS < <($POOL_PY -c "
import json
d=json.load(open('$TASKLIST'))
t=sorted(d['tasks'] if isinstance(d,dict) and 'tasks' in d else d)
for x in t[$LO:$HI+1]: print(x)")
[ -n "${LIMIT:-}" ] && TASKS=("${TASKS[@]:0:$LIMIT}")
echo "  driving ${#TASKS[@]} task(s), slice $SLICE"

n_ok=0; n_fail=0
for TID in "${TASKS[@]}"; do
  OUTD="$CAMP/$TID"
  if [ -e "$OUTD" ]; then echo "  skip $TID (already run)"; continue; fi
  echo "  >>> $TID"
  mkdir -p "$OUTD"
  T_TASK_START=$(date +%s.%N)
  timeout 10 curl -sS "http://127.0.0.1:$POOLPORT/healthz" -o "$OUTD/pool_health.json" 2>/dev/null
  $POOL_PY - "$OUTD" "$POOL_INDEX" <<'PROV' 2>/dev/null
import json, hashlib, os, sys
outd, idx = sys.argv[1], sys.argv[2]
def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()[:16]
m = json.load(open(os.path.join(idx, "meta.json")))
json.dump({"schema": "pool_provenance/1.0",
           "index_dir": idx, "bm25_meta_sha256_16": sha16(os.path.join(idx, "meta.json")),
           "stats_fingerprint": m.get("stats_fingerprint"), "n_docs": m.get("n_docs"),
           "n_shards": m.get("n_shards"), "bm25": m.get("bm25")},
          open(os.path.join(outd, "pool_provenance.json"), "w"), indent=2)
PROV
  setsid env PYTHONPATH="$POOL_PYTHONPATH" PYTHONNOUSERSITE=1 $POOL_PY "$LITREVIEW_ROOT/scribe/harness/runners/llm_proxy.py" \
    --listen $PROXYPORT --upstream $VPORT --log "$OUTD/_calls.jsonl" \
    > $LOGS/camp_proxy_${SHARD}_${TID}.log 2>&1 &
  PXP=$!
  if ! wait_http "http://127.0.0.1:$PROXYPORT/v1/models" "recorder[$TID]" 24; then
    kill -- -$PXP 2>/dev/null; n_fail=$((n_fail+1)); continue
  fi
  drtulu_py "$B/drtulu_task.py" \
    --task-id "$TID" \
    --task-input "$BASELINE_INPUTS/$TID/refs.json" \
    --out-dir "$OUTD" \
    --llm-base-url "http://127.0.0.1:$PROXYPORT/v1" \
    --pool-port $POOLPORT \
    --agent-dir "$DRTULU_AGENT_DIR" \
    --tokenizer "$DT8B" \
    --model-name DR-Tulu-8B \
    --pool-log "$POOL_LOG" \
    --python "$DRTULU_PY" \
    > "$OUTD/stdout.log" 2> "$OUTD/stderr.log"
  rc=$?
  cat "$OUTD/stdout.log" "$OUTD/stderr.log" >> "$CAMP/_run_shard${SHARD}.log" 2>/dev/null
  T_TASK_END=$(date +%s.%N)
  kill -- -$PXP 2>/dev/null; wait $PXP 2>/dev/null
  $POOL_PY - "$OUTD" "$T_TASK_START" "$T_TASK_END" "$VLLM_STARTUP_S" "$POOL_STARTUP_S" "$SHARD" <<'TIM' 2>/dev/null
import json, sys
o,a,b,v,p,w = sys.argv[1:7]
json.dump({"schema":"measure_timing/1.0","worker":w,
  "agent_runtime_s": round(float(b)-float(a),1),
  "job_startup_vllm_s": float(v) if v else None,
  "job_startup_pool_s": float(p) if p else None},
  open(o+"/timing.json","w"), indent=2)
TIM
  ok=$($POOL_PY -c "
import json,sys
try:
    d=json.load(open('$OUTD/report.json'))
    sys.exit(0 if str(d.get('answer','')).strip() else 1)
except Exception: sys.exit(1)" 2>/dev/null && echo 1 || echo 0)
  if [ "$ok" = "1" ]; then
    n_ok=$((n_ok+1)); echo "      ok ($(wc -l < "$OUTD/_calls.jsonl" 2>/dev/null || echo 0) llm calls)"
  else
    n_fail=$((n_fail+1)); echo "      FAILED rc=$rc"
    head -3 "$OUTD/stderr.log" 2>/dev/null | sed "s/^/        /"
  fi
done
echo "  driver: $n_ok ok, $n_fail failed"
