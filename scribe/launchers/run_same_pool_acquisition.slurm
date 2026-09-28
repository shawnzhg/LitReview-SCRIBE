#!/bin/bash
# Slurm GPU launcher that runs and audits the same-pool acquisition of SCRIBE inside the egress guard.
# Usage: THETA=<carrier> TAG=<name> LEVEL=<n> [SEEDS=0|1|2] sbatch --export=ALL
# run_same_pool_acquisition.slurm.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=180G --gres=gpu:1
#SBATCH --time=20:00:00
set -uo pipefail
case "${1:-}" in -h|--help) awk 'NR>1 && /^#SBATCH/ {next} NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0;; esac
LITREVIEW_ROOT=${LITREVIEW_ROOT:-$(cd "$(dirname "$0")/../.." 2>/dev/null && pwd)}
[ -f "$LITREVIEW_ROOT/scribe/harness/runners/runner.py" ] || { echo "FATAL: set LITREVIEW_ROOT to the repo (configs/site.env.example)"; exit 80; }
for v in SCRIBE_WORK_ROOT SCRIBE_SERVE_PY SCRIBE_TRAIN_PY SCRIBE_RUN_PY SCRIBE_POOL_PY SCRIBE_HF_HOME SCRIBE_MODEL_DIR SCRIBE_THETA0 SCRIBE_RUNS_ROOT \
         SCRIBE_EVAL_TASKS SCRIBE_POOL_INDEX SCRIBE_POOL_CUTOFFS SCRIBE_POOL_EDGES POOL_GOLD_DIR SCRIBE_BUDGET_CAPS SCRIBE_KCAP; do
  [ -n "${!v:-}" ] || { echo "FATAL: $v is not set (configs/site.env.example)"; exit 80; }
done
SERVE_PY=$SCRIBE_SERVE_PY; LEARN_PY=$SCRIBE_TRAIN_PY; RUN_PY=$SCRIBE_RUN_PY; MODEL=$SCRIBE_MODEL_DIR
POOL_PY=$SCRIBE_POOL_PY; POOL_PYTHONPATH=${SCRIBE_POOL_PYTHONPATH:-}
POOL_INDEX_DIR=$SCRIBE_POOL_INDEX; POOL_CUTOFFS_FILE=$SCRIBE_POOL_CUTOFFS
POOL_SERVICE=$LITREVIEW_ROOT/biolitbench/pool/pool_service.py
RETRIEVAL=$LITREVIEW_ROOT/scribe/retrieval; RUNNERS=$LITREVIEW_ROOT/scribe/harness/runners; TOOLS=$LITREVIEW_ROOT/scribe/harness/tools
export POOL_EDGES=$SCRIBE_POOL_EDGES POOL_GOLD_DIR
unset POOL_ALLOWLIST_DIR
PARITY_CHECK=$RETRIEVAL/pool_parity.py; PARITY_PROBE=$RETRIEVAL/pool_parity_probe.sh
read -r PIN_META_SHA16 PIN_STATS_FP PIN_N_DOCS POOL_SNAPSHOT_ID < <($RUN_PY -c 'import sys; sys.path.insert(0, sys.argv[1]); import pool_retrieval_tool as P; print(P.PIN_META_SHA16, P.PIN_STATS_FINGERPRINT, P.PIN_N_DOCS, P.POOL_SNAPSHOT_ID)' "$TOOLS")
[ -n "${POOL_SNAPSHOT_ID:-}" ] || { echo "FATAL: cannot read the frozen-pool constants from $TOOLS/pool_retrieval_tool.py"; exit 81; }
THETA=${THETA:?THETA artifact dir required}; TAG=${TAG:?TAG required}
MODE=native_chain; LEVEL=${LEVEL:?LEVEL required (a fresh level)}
BUDGET_CAPS=$SCRIBE_BUDGET_CAPS; KCAP=$SCRIBE_KCAP
[ -f "$BUDGET_CAPS" ] && [ -f "$KCAP" ] || { echo "FATAL: SCRIBE_BUDGET_CAPS / SCRIBE_KCAP are not files"; exit 82; }
BUDGET_CAPS_SHA256=$(sha256sum < "$BUDGET_CAPS" | cut -c1-64); KCAP_SHA256=$(sha256sum < "$KCAP" | cut -c1-64)
SYSTEMS=${SYSTEMS:-SCRIBE}; SEEDS=${SEEDS:-0}; WORKERS=${WORKERS:-8}; NTASKS=${NTASKS:-50}
[[ "$SYSTEMS" =~ ^[A-Za-z0-9_]+$ ]] || { echo "FATAL: SYSTEMS=$SYSTEMS must name exactly one system (SCRIBE)"; exit 80; }
case "$SEEDS" in 0|1|2) ;; *) echo "FATAL: SEEDS=$SEEDS: one seed per same-pool level, 0, 1 or 2"; exit 80;; esac
EXPECT_IDENTITY=0; [ "$(readlink -f "$THETA")" = "$(readlink -f "$SCRIBE_THETA0")" ] && EXPECT_IDENTITY=1
PORT=${PORT:-$(( 8400 + ${SLURM_JOB_ID:-0} % 500 ))}; MAXLEN=65536
$RUN_PY -c 'import sys; sys.path[:0] = sys.argv[1:3]; import query_agent as Q, selection_rule as E; Q.config(); E.config()' "$RETRIEVAL" "$RUNNERS" > /dev/null \
  || { echo "FATAL: the query agent skills / selection rule do not load (checked before the level is claimed)"; exit 80; }
ACQ_DRIVER=$RETRIEVAL/run_acquisition.py
POOL_PORT=8931
[[ "$TAG" =~ ^[A-Za-z0-9_]+$ ]] || { echo "FATAL: TAG must be [A-Za-z0-9_]+"; exit 80; }
ARM=same_pool_${TAG}_cap
RUN=$SCRIBE_WORK_ROOT/runs/eval_${ARM}_${SLURM_JOB_ID:-preflight$$}
export PYTHONNOUSERSITE=1 HF_HOME=$SCRIBE_HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$LITREVIEW_ROOT/third_party/kvskill
export PATH=$(dirname $SERVE_PY):$PATH
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 FLASHINFER_WORKSPACE_BASE=$SCRIBE_WORK_ROOT/jit_cache
if [ -n "${SCRIBE_TOOLCHAIN:-}" ]; then source "$SCRIBE_TOOLCHAIN" || { echo "FATAL: toolchain ($SCRIBE_TOOLCHAIN) failed -- exit 99, before any GPU work, server start or level write"; exit 99; }; fi
[ -n "${CUDA_HOME:-}" ] || { echo "FATAL: CUDA_HOME is not set (set it or SCRIBE_TOOLCHAIN)"; exit 99; }
mkdir -p $RUN $SCRIBE_WORK_ROOT/logs
echo "### start=$(date) node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-} theta=$THETA mode=$MODE level=$LEVEL tag=$TAG arm=$ARM seeds=$SEEDS ntasks=$NTASKS"
echo "### cap budget: K_cap file $KCAP (sha256 ${KCAP_SHA256:0:16}); caps file $BUDGET_CAPS (sha256 ${BUDGET_CAPS_SHA256:0:16})"
echo "### acquisition driver $ACQ_DRIVER: query agent, ranker, selection rule; tokenizer $MODEL; max_model_len $MAXLEN; pool $POOL_SERVICE edges $POOL_EDGES"
echo "### launcher: $0 md5 $(md5sum "$0" 2>/dev/null | awk '{print $1}')"
cp -p "$0" "$RUN/launcher.frozen.slurm" 2>/dev/null
RUNS=$SCRIBE_RUNS_ROOT; LVL_DIR=$RUNS/level$LEVEL

[[ $LEVEL =~ ^[0-9]+$ ]] && [ "$LEVEL" -ge 1 ] || { echo "FATAL: LEVEL=$LEVEL must be a positive integer"; exit 80; }
if compgen -G "$LVL_DIR/_level_infra_failure_*.json" >/dev/null || compgen -G "$LVL_DIR/_level_*failed*.json" >/dev/null; then
  echo "FATAL: level$LEVEL carries a failure marker ($(ls $LVL_DIR/_level_infra_failure_*.json $LVL_DIR/_level_*failed*.json 2>/dev/null | head -1)); re-run at a fresh level"; exit 80
fi
if [ -e "$LVL_DIR" ]; then
  echo "FATAL: level$LEVEL already exists ($(ls $LVL_DIR | head -3 | tr '\n' ' ')); a same-pool arm needs a fresh level"; exit 80
fi
IDX_HIT=$(grep -lE "\"level\": $LEVEL[,}]" $RUNS/_index/*.jsonl 2>/dev/null | grep -v "/${ARM}_" | head -1)
if [ -n "$IDX_HIT" ]; then
  echo "FATAL: another campaign index records runs at level $LEVEL ($IDX_HIT); pick a fresh level"; exit 80
fi

TASKS=$($RUN_PY -c "import json; print(','.join(sorted(json.load(open('$SCRIBE_EVAL_TASKS'))['tasks'])[:$NTASKS]))")
$RUN_PY - "$POOL_INDEX_DIR" "$POOL_CUTOFFS_FILE" "$TASKS" "$PIN_META_SHA16" "$PIN_STATS_FP" "$PIN_N_DOCS" "$RUN/pool_inputs.json" "$RUNS" <<'PYEOF' || exit 81
import hashlib, json, sys
idx, cf, tasks, pin_sha, pin_fp, pin_n, out, runs = sys.argv[1:9]
raw = open(f"{idx}/meta.json", "rb").read(); m = json.loads(raw)
sha16 = hashlib.sha256(raw).hexdigest()[:16]
cuts = json.load(open(cf)); craw = open(cf, "rb").read()
bad = []
if sha16 != pin_sha: bad.append(f"index meta sha16 {sha16} != the frozen pool's {pin_sha}")
if m.get("stats_fingerprint") != pin_fp: bad.append(f"stats_fingerprint {m.get('stats_fingerprint')} != the frozen pool's {pin_fp}")
if int(m.get("n_docs", -1)) != int(pin_n): bad.append(f"n_docs {m.get('n_docs')} != {pin_n}")
L = f"{runs}/taskspecs/dev"
rows = {}
for t in tasks.split(","):
    if t not in cuts: bad.append(f"{t} not in cutoffs.json (the service would answer 400)"); continue
    sc = json.load(open(f"{L}/{t}.json"))["publication_cutoff"]
    rows[t] = cuts[t]
    if int(sc) != int(cuts[t]): bad.append(f"{t}: TaskSpec cutoff {sc} != cutoffs.json {cuts[t]}")
rec = {"index": idx, "bm25_meta_sha256_16": sha16, "stats_fingerprint": m.get("stats_fingerprint"), "n_docs": m.get("n_docs"),
       "cutoffs": cf, "cutoffs_sha16": hashlib.sha256(craw).hexdigest()[:16], "task_cutoffs": rows, "problems": bad}
json.dump(rec, open(out, "w"), indent=1)
print(f"### pool inputs: meta sha16 {sha16} fp {m.get('stats_fingerprint')} n_docs {m.get('n_docs')}; {len(rows)} tasks, cutoffs {sorted(set(rows.values()))}; cutoffs.json sha16 {rec['cutoffs_sha16']}")
if bad:
    print("FATAL: " + "; ".join(bad[:8])); sys.exit(1)
PYEOF
$RUN_PY $PARITY_CHECK static --service "$POOL_SERVICE" --out $RUN/pool_parity.json || { echo "FATAL: the pool config is not the baselines' pool-service configuration (see $RUN/pool_parity.json)"; exit 81; }
$RUN_PY -c "import json,sys; p=json.load(open(sys.argv[1])); p['pool_parity']=json.load(open(sys.argv[2])); json.dump(p, open(sys.argv[1], 'w'), indent=1)" $RUN/pool_inputs.json $RUN/pool_parity.json || exit 81

$RUN_PY - "$BUDGET_CAPS" "$BUDGET_CAPS_SHA256" "$KCAP" "$KCAP_SHA256" "$TASKS" "$RUN" <<'PYEOF' || exit 82
import hashlib, json, statistics as st, sys
cpath, cwant, kpath, kwant, tasks, run = sys.argv[1:7]
craw = open(cpath, "rb").read(); csha = hashlib.sha256(craw).hexdigest(); caps = json.loads(craw)
kraw = open(kpath, "rb").read(); ksha = hashlib.sha256(kraw).hexdigest(); d = json.loads(kraw)
bad = [] if csha == cwant else [f"{cpath} sha256 {csha} != {cwant} recorded at job start"]
if ksha != kwant: bad.append(f"{kpath} sha256 {ksha} != {kwant} recorded at job start")
if d.get("schema") != "kcap/1.0": bad.append(f"{kpath}: schema {d.get('schema')} != kcap/1.0")
if (d.get("inputs") or {}).get("caps_sha256") != csha: bad.append(f"K_cap file built from caps {(d.get('inputs') or {}).get('caps_sha256')} != the caps file of this run {csha}")
T = [t for t in tasks.split(",") if t]; E = d.get("tasks") or {}; C = (caps.get("effective") or {}).get("cap") or {}
miss = [t for t in T if not E.get(t)]
if miss: bad.append(f"{len(miss)} tasks have no K_cap: {miss[:3]}")
miss = [t for t in T if not C.get(t)]
if miss: bad.append(f"{len(miss)} tasks have no measured rows-per-call cap: {miss[:3]}")
bad += [f"{t}: K_cap {E[t]['K_cap']} outside 1..1000" for t in T if E.get(t) and not 1 <= int(E[t]["K_cap"]) <= 1000]
eff = {t: dict((E.get(t) or {}).get("effective") or {}, max_results_per_call=(C.get(t) or {}).get("max_results_per_call")) for t in T}
json.dump({"path": kpath, "sha256": ksha, "rule": d.get("rule"), "K_cap": {t: (E.get(t) or {}).get("K_cap") for t in T},
           "problems": bad}, open(f"{run}/kcap_inputs.json", "w"), indent=1)
json.dump({"path": cpath, "sha256": csha, "raw_sources_md5": caps.get("raw_sources_md5"), "kcap": kpath, "kcap_sha256": ksha,
           "effective": eff, "problems": bad}, open(f"{run}/retrieval_budget_inputs.json", "w"), indent=1)
v = [e for t, e in eff.items() if e.get("max_search_calls") is not None]
if v:
    print(f"### K_cap: file sha256 {ksha[:16]}; caps sha256 {csha[:16]}; {len(v)} tasks; median effective calls "
          f"{st.median(x['max_search_calls'] for x in v)}, K {st.median(x['max_ranked_output_K'] for x in v)}, "
          f"opens {st.median(x['max_document_opens'] for x in v)}, docs {st.median(x['max_docs_read'] for x in v)}")
if bad: print("FATAL: " + "; ".join(bad)); sys.exit(1)
PYEOF

CODE_FILES="$(ls $RUNNERS/*.py $RUNNERS/egress_guard.sh $TOOLS/*.py $RETRIEVAL/*.py $RETRIEVAL/*.sh $RETRIEVAL/skills/*.txt $LITREVIEW_ROOT/biolitbench/schemas/*.json $LITREVIEW_ROOT/biolitbench/pool/pool_service.py $LITREVIEW_ROOT/biolitbench/pool/pool_common.py $LITREVIEW_ROOT/third_party/kvskill/kvskill/*.py) $BUDGET_CAPS $KCAP $POOL_INDEX_DIR/meta.json $POOL_EDGES/meta.json $MODEL/tokenizer.json $MODEL/chat_template.jinja"
md5sum $CODE_FILES > $RUN/code.md5 || { echo "FATAL: cannot fingerprint the code and inputs"; exit 88; }
echo "### code and inputs recorded at start: $(wc -l < $RUN/code.md5) files ($RUN/code.md5)"
$LITREVIEW_ROOT/scribe/harness/runners/egress_guard.sh "$RUN/parity_probe_egress_proof.json" bash $PARITY_PROBE $RUN/parity_probe $POOL_PORT "$POOL_SERVICE" "$POOL_INDEX_DIR" "$POOL_CUTOFFS_FILE"; RC_PROBE=$?
[ "$RC_PROBE" = 0 ] || { echo "FATAL: pool parity probe refused (rc $RC_PROBE; see $RUN/parity_probe): level $LEVEL NOT claimed, nothing served"; exit 83; }

if [ ! -e "$LVL_DIR" ]; then
  mkdir -p "$RUNS" && mkdir "$LVL_DIR" 2>/dev/null || { echo "FATAL: level$LEVEL was created by someone else between the checks and the claim"; exit 80; }
fi
echo "{\"job\": \"${SLURM_JOB_ID:-}\", \"arm\": \"$ARM\", \"tag\": \"$TAG\", \"retrieval_budget\": \"cap\", \"seed\": $SEEDS, \"run\": \"$RUN\", \"utc\": \"$(date -u +%FT%TZ)\"}" > "$LVL_DIR/_level_claim_${SLURM_JOB_ID:-nojob}.json"

$LEARN_PY -m kvskill.export --artifact_path $THETA --out $RUN/payload.safetensors 2>&1 || exit 91
SHAPE=$($RUN_PY -c "import json; m=json.load(open('$THETA/meta.json')); print(f\"{m['cn_ds']},{m['cn_dm']},{m['cn_r']},{m['cn_nlayers']}\")")
VN=$($LEARN_PY -c "
from safetensors import safe_open; f=safe_open('$RUN/payload.safetensors','pt'); import torch; print(float(f.get_tensor('cn_v').float().norm()))" 2>&1 | tail -1)
echo "### carrier shape=$SHAPE ||V||=$VN  (0 => identity carrier)"

export LITREVIEW_ROOT SERVE_PY LEARN_PY RUN_PY MODEL THETA TAG MODE LEVEL SYSTEMS SEEDS WORKERS EXPECT_IDENTITY
export BUDGET_CAPS BUDGET_CAPS_SHA256 KCAP KCAP_SHA256
export SCRIBE_BUDGET_CAPS=$BUDGET_CAPS SCRIBE_BUDGET_CAPS_SHA256=$BUDGET_CAPS_SHA256
export SCRIBE_KCAP=$KCAP SCRIBE_KCAP_SHA256=$KCAP_SHA256
export SCRIBE_RANKING_MAX_MODEL_LEN=$MAXLEN SCRIBE_RANKING_TOKENIZER=$MODEL
export ACQ_DRIVER PARITY_CHECK POOL_EDGES POOL_PY POOL_PYTHONPATH RETRIEVAL
export PORT MAXLEN POOL_PORT ARM RUN TASKS SHAPE VN POOL_INDEX_DIR POOL_CUTOFFS_FILE POOL_SERVICE PIN_STATS_FP PIN_N_DOCS POOL_SNAPSHOT_ID RUNS
cat > $RUN/inner.sh <<'INNER'
#!/bin/bash
set -uo pipefail
PIDS=(); cleanup(){ for p in "${PIDS[@]:-}"; do kill -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done; }; trap cleanup EXIT
wait_http(){ for i in $(seq 1 "$3"); do curl -sf -o /dev/null "$1" 2>/dev/null && { echo "  $2 UP (~$((i*5))s)"; return 0; }; sleep 5; done; echo "FATAL: $2 never came up"; return 1; }
drift(){ md5sum -c --quiet $RUN/code.md5 >/dev/null 2>&1 || { echo "FATAL: code or inputs changed since job start before phase $1:"; md5sum -c $RUN/code.md5 2>&1 | grep -v ': OK$'; exit 88; }; }

VPID=""
start_vllm(){
  local payload=$1 vn=$2 theta=$3 ph=$4 log=$RUN/server_$4.log
  KVSKILL_CN_SHAPE=$SHAPE KVSKILL_CN_PATH=$payload setsid $SERVE_PY -m vllm.entrypoints.openai.api_server \
    --model $MODEL --served-model-name qwen38 --host 127.0.0.1 --port $PORT --dtype bfloat16 \
    --max-model-len $MAXLEN --gpu-memory-utilization 0.90 --max-num-seqs 48 --seed 0 \
    --worker-extension-cls kvskill.vllm_patch.SkillWorkerExtension > $log 2>&1 &
  VPID=$!; PIDS+=($VPID)
  wait_http "http://127.0.0.1:$PORT/v1/models" "vllm ($ph)" 400 || { tail -40 $log; exit 94; }
  grep -E "\[cn_vllm\] (armed|staged)" $log | head -3
  grep -q "\[cn_vllm\] staged" $log || { echo "FATAL: carrier was NOT staged into the OpenAI server (patch inactive)"; exit 95; }
  local srv_vn; srv_vn=$(grep -oE '(‖V‖|\|\|V\|\|)=[0-9.]+' $log | head -1 | sed 's/.*=//')
  [ -n "$srv_vn" ] || { echo "FATAL: server log reports no carrier norm; cannot verify which carrier is loaded"; exit 95; }
  $LEARN_PY -c "
import sys
srv, art = float('$srv_vn'), float('$vn')
if abs(srv - art) > 1e-3:
    sys.exit(f'FATAL: server staged norm {srv} != exported artifact norm {art}')
if srv == 0.0 and '$EXPECT_IDENTITY' != '1':
    sys.exit('FATAL: staged norm is 0.0 (the identity carrier) but THETA is not SCRIBE_THETA0')
if srv != 0.0 and '$EXPECT_IDENTITY' == '1':
    sys.exit(f'FATAL: THETA is SCRIBE_THETA0 but the staged norm is {srv}')
print(f'### carrier verified ($ph): server norm {srv} == artifact norm {art}' + ('  (identity carrier, THETA = SCRIBE_THETA0)' if srv == 0.0 else ''))
" || exit 95
  grep -E "sampler=|attention=|GDN|gdn" $log | head -5
  cat > $RUN/rendezvous_$ph.json <<JSON
{"host": "127.0.0.1", "port": $PORT, "served_model": "qwen38", "model_path": "$MODEL", "max_model_len": $MAXLEN,
 "job_id": "$SLURM_JOB_ID", "url": "http://127.0.0.1:$PORT/v1", "kvskill_theta": "$theta", "kvskill_payload": "$payload",
 "kvskill_shape": "$SHAPE", "kvskill_v_norm": $vn, "chat_template_sha256_16": "$(sha256sum $MODEL/chat_template.jinja | cut -c1-16)",
 "entry_mode": "$MODE", "retrieval_backend": "pool", "phase": "$ph", "pool_port": $POOL_PORT,
 "retrieval_budget": "cap", "budget_caps_sha256": "$BUDGET_CAPS_SHA256",
 "corpus_snapshot_id": "$POOL_SNAPSHOT_ID"}
JSON
  export SCRIBE_RENDEZVOUS="$(cat $RUN/rendezvous_$ph.json)"
}
start_vllm "$RUN/payload.safetensors" "$VN" "$THETA" acquisition

$POOL_PY -c "import sys; sys.path.insert(0,'$POOL_PYTHONPATH'); import Stemmer, bm25s" 2>/dev/null \
  || { echo "FATAL: the pool interpreter cannot import Stemmer/bm25s (SCRIBE_POOL_PY, SCRIBE_POOL_PYTHONPATH)"; exit 83; }
export POOL_LOG="$RUN/search_calls.jsonl"
T_P0=$(date +%s)
setsid env PYTHONPATH="$POOL_PYTHONPATH" PYTHONNOUSERSITE=1 $POOL_PY $POOL_SERVICE --host 127.0.0.1 --port $POOL_PORT \
  --index "$POOL_INDEX_DIR" --cutoffs "$POOL_CUTOFFS_FILE" > $RUN/pool_service.log 2>&1 &
POOL_PID=$!; PIDS+=($POOL_PID)
wait_http "http://127.0.0.1:$POOL_PORT/healthz" pool 60 || { tail -20 $RUN/pool_service.log; exit 83; }
curl -sf "http://127.0.0.1:$POOL_PORT/healthz" > $RUN/pool_healthz.json
$RUN_PY -c "
import json,sys; h=json.load(open('$RUN/pool_healthz.json'))
ok = h.get('stats_fingerprint')=='$PIN_STATS_FP' and int(h.get('n_docs',-1))==$PIN_N_DOCS
print('### pool service: startup $(( $(date +%s) - T_P0 ))s', {k: h.get(k) for k in ('n_docs','stats_fingerprint','index_dir','n_tasks','edges_present')})
sys.exit(0 if ok else 1)" || { echo "FATAL: pool /healthz does not match the frozen pool"; exit 83; }
$RUN_PY $PARITY_CHECK health $RUN/pool_healthz.json --label acquisition-pool || { echo "FATAL: the acquisition pool is not the baselines' pool-service configuration"; exit 83; }
env PYTHONPATH="$POOL_PYTHONPATH:$LITREVIEW_ROOT/biolitbench/pool" PYTHONNOUSERSITE=1 $POOL_PY $RETRIEVAL/pool_boundary.py \
  --index "$POOL_INDEX_DIR" --years 2010-2027 --out $RUN/pool_boundary.json || { echo "FATAL: boundary pmids"; exit 83; }
export POOL_URL=http://127.0.0.1:$POOL_PORT POOL_CUTOFFS=$POOL_CUTOFFS_FILE POOL_INDEX=$POOL_INDEX_DIR POOL_BOUNDARY=$RUN/pool_boundary.json
drift acquisition
cd $RUN
echo "### acquisition start $(date)"
$RUN_PY $ACQ_DRIVER --phase acquisition --systems $SYSTEMS --mode $MODE \
  --level $LEVEL --seeds $SEEDS --tasks "$TASKS" --workers $WORKERS \
  --campaign "${ARM}_acq" 2>&1 | tail -70
RC_ACQ=${PIPESTATUS[0]}
echo "### acquisition end $(date) rc=$RC_ACQ: bundles=$(ls $RUNS/level$LEVEL/$SYSTEMS/$MODE/*/seed*/evidence_bundle.json 2>/dev/null | wc -l) ok=$(grep -c '"status": "ok"' $RUNS/_index/${ARM}_acq.jsonl 2>/dev/null) service_calls=$(wc -l < $RUN/search_calls.jsonl 2>/dev/null)"
ps -o pid=,rss=,cputime=,etime= -p $POOL_PID 2>/dev/null | sed 's/^/### pool service pid rss(KB) cputime etime: /'
kill -- -$POOL_PID 2>/dev/null || kill $POOL_PID 2>/dev/null
for i in 1 2 3 4 5 6; do curl -sf -o /dev/null "http://127.0.0.1:$POOL_PORT/healthz" 2>/dev/null || break; sleep 2; done
if curl -sf -o /dev/null "http://127.0.0.1:$POOL_PORT/healthz" 2>/dev/null; then
  kill -KILL -- -$POOL_PID 2>/dev/null || kill -KILL $POOL_PID 2>/dev/null; sleep 3
  curl -sf -o /dev/null "http://127.0.0.1:$POOL_PORT/healthz" 2>/dev/null && { echo "FATAL: the pool service is still answering after acquisition"; exit 92; }
fi
echo "### pool service stopped (verified: /healthz no longer answers; the late windows have no route to any corpus)"
mark(){ echo "{\"schema\": \"level_marker/1.0\", \"kind\": \"$1\", \"job\": \"${SLURM_JOB_ID:-}\", \"arm\": \"$ARM\", \"why\": \"$2\", \"run\": \"$RUN\", \"utc\": \"$(date -u +%FT%TZ)\", \"rule\": \"this level must not be resumed, generated, exposed or certified; re-run at a fresh level\"}" > "$RUNS/level$LEVEL/_level_$1_${SLURM_JOB_ID:-nojob}.json"; }
[ "$RC_ACQ" = 86 ] && { echo "### INFRA GUARD fired in acquisition (see level$LEVEL/_level_infra_failure_*); level must not be resumed"; exit 86; }
[ "$RC_ACQ" = 0 ] || { mark acq_failed "acquisition driver rc=$RC_ACQ"; echo "FATAL: acquisition driver rc=$RC_ACQ"; exit 92; }
drift audits
$RUN_PY $PARITY_CHECK routes $RUN/search_calls.jsonl --min-rows 1 --out $RUN/pool_routes.json --label acquisition-pool \
  || { mark audit_failed "route guard: a pool route / field other than /search + /paper(title,abstract,year), see $RUN/pool_routes.json"; echo "FATAL: the acquisition used a pool route on which the review under evaluation is not excluded (see $RUN/pool_routes.json); generation NOT run; level marked dead"; exit 87; }
$RUN_PY $RETRIEVAL/acquisition_audit.py --runs $RUNS --level $LEVEL --tasks "$TASKS" --cutoffs $POOL_CUTOFFS_FILE \
  --service-log $RUN/search_calls.jsonl --seeds "$SEEDS" --system $SYSTEMS --out $RUN/acquisition_audit.json \
  || { mark audit_failed "acquisition audit findings, see $RUN/acquisition_audit.json"; echo "FATAL: acquisition audit findings (see $RUN/acquisition_audit.json); generation NOT run; level marked dead"; exit 87; }
$RUN_PY $RETRIEVAL/ranking_budget_audit.py --runs $RUNS --level $LEVEL --tasks "$TASKS" --kcap $KCAP --kcap-sha256 $KCAP_SHA256 \
  --seeds "$SEEDS" --system $SYSTEMS --out $RUN/ranking_budget_audit.json \
  || { mark audit_failed "cap budget / ranking record / query agent record audit findings, see $RUN/ranking_budget_audit.json"; echo "FATAL: cap budget / ranking / query agent audit findings (see $RUN/ranking_budget_audit.json); generation NOT run; level marked dead"; exit 87; }
$RUN_PY $RETRIEVAL/selection_audit.py --glob "$RUNS/level$LEVEL/$SYSTEMS/$MODE/*/seed*" --ranking-audit --require --min-units 1 \
  --module-md5 $(md5sum < $RETRIEVAL/selection_rule.py | cut -c1-32) --gold-root $RUNS/gold/dev --out $RUN/selection_audit.json \
  || { mark audit_failed "selection rule audit findings, see $RUN/selection_audit.json"; echo "FATAL: selection rule audit findings (see $RUN/selection_audit.json); generation NOT run; level marked dead"; exit 87; }
$RUN_PY - "$RUN/acquisition_audit.json" "$RUNS/level$LEVEL/_level_acq_audit_ok.json" "$LEVEL" "$ARM" "$TASKS" "$SEEDS" <<'PYEOF' || { echo "FATAL: cannot write the audit record"; exit 92; }
import hashlib, json, os, sys, time
src, dst, lvl, arm, tasks, seeds = sys.argv[1:7]
a = json.load(open(src)); raw = open(src, "rb").read()
rec = {"schema": "acquisition_audit_ok/1.0", "ok": a.get("n_findings") == 0, "level": int(lvl), "retrieval_budget": "cap",
       "arm": arm, "job": os.environ.get("SLURM_JOB_ID"), "audit": src, "audit_md5": hashlib.md5(raw).hexdigest(),
       "n_units_audited": a.get("n_units_audited"), "no_bundle": a.get("no_bundle"),
       "n_tasks": len([t for t in tasks.split(",") if t]), "seeds": seeds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(rec, open(dst, "w"), indent=1); print(f"### acquisition audit OK record -> {dst}")
sys.exit(0 if rec["ok"] else 1)
PYEOF

exit 0
INNER
chmod +x $RUN/inner.sh
$LITREVIEW_ROOT/scribe/harness/runners/egress_guard.sh "$RUN/egress_proof.json" bash $RUN/inner.sh; RC=$?

$RUN_PY - "$RUN" "$LEVEL" "$TAG" "$THETA" "$ARM" "$RC" "$SEEDS" "$RUNS" "$BUDGET_CAPS" "$BUDGET_CAPS_SHA256" "$TASKS" "$KCAP" "$KCAP_SHA256" "$POOL_SNAPSHOT_ID" <<'PYEOF'
import glob, json, os, sys
run, level, tag, theta, arm, rc, seeds, R, rcaps, rsha, tasks, kcap, kcap_sha, snap = sys.argv[1:15]
def lj(p):
    try: return json.load(open(p))
    except Exception: return None
ka = lj(f"{run}/ranking_budget_audit.json")
sa = lj(f"{run}/selection_audit.json")
units = sorted(glob.glob(f"{R}/level{level}/{os.environ.get('SYSTEMS') or 'SCRIBE'}/native_chain/*/seed*"))
def cnt(n): return sum(os.path.exists(f"{u}/{n}") for u in units)
prov_ok = 0
for u in units:
    try: prov_ok += bool(json.load(open(f"{u}/pool_provenance.json")).get("ok"))
    except Exception: pass
rec = {"schema": "level_record/1.0", "state": "final", "rc": int(rc), "level": int(level), "tag": tag, "arm": arm,
       "theta": theta, "seeds": seeds, "job": os.environ.get("SLURM_JOB_ID"), "mode": "native_chain",
       "retrieval_backend": "pool (frozen static pool, pool_service.py, per-task cutoffs)",
       "corpus_snapshot_id": snap, "board": "native_chain (separate from the fixed-input board)",
       "retrieval_budget": "cap", "budget_caps": rcaps, "budget_caps_sha256": rsha,
       "n_tasks_expected": len([t for t in tasks.split(",") if t]) * len([s for s in seeds.split(",") if s]),
       "acq_audit_ok": (json.load(open(f"{R}/level{level}/_level_acq_audit_ok.json")).get("ok") is True)
                       if os.path.exists(f"{R}/level{level}/_level_acq_audit_ok.json") else False,
       "failure_markers": sorted(glob.glob(f"{R}/level{level}/_level_*failed*.json")),
       "n_units": len(units), "n_bundles": cnt("evidence_bundle.json"), "n_reports": cnt("report_artifact.json"),
       "n_pool_provenance_ok": prov_ok, "infra_markers": sorted(glob.glob(f"{R}/level{level}/_level_infra_failure_*.json")),
       "indexes": {p: os.path.exists(f"{R}/_index/{arm}_{p}.jsonl") for p in ("acq", "gen")},
       "pool_inputs": lj(f"{run}/pool_inputs.json"),
       "egress_proof": f"{run}/egress_proof.json" if os.path.exists(f"{run}/egress_proof.json") else None,
       "service_log": f"{run}/search_calls.jsonl", "code_md5": open(f"{run}/code.md5").read().splitlines(),
       "cap_rule": "K_cap", "kcap": kcap, "kcap_sha256": kcap_sha,
       "kcap_audit": {"n_findings": ka.get("n_findings"), "fallback": ka.get("fallback")} if ka else None,
       "ranking": ({k: (ka.get("ranking") or {}).get(k) for k in ("n_units", "n_no_bundle", "n_task_fallback", "fallback_rate", "threshold",
                    "n_context_overflow_attempts", "n_rank_overflow_attempts", "n_query_step_overflow_attempts", "n_disagree",
                    "n_tournament", "n_single", "n_no_final", "fill_median", "fill_min", "n_fill_below_0_9")}
                   if ka else None),
       "pool_parity": {"acq_driver": os.environ.get("ACQ_DRIVER"), "static": lj(f"{run}/pool_parity.json"),
                       "parity_probe_healthz": lj(f"{run}/parity_probe/probe_healthz.json"), "acq_pool_healthz": lj(f"{run}/pool_healthz.json"),
                       "pool_routes": lj(f"{run}/pool_routes.json")},
       "selection_rule": {"acq_driver": os.environ.get("ACQ_DRIVER"),
                          "audit": ({k: sa.get(k) for k in ("n_units", "n_findings", "findings")} if sa else None)}}
json.dump(rec, open(f"{run}/level_record.json", "w"), indent=1)
print("### record:", {k: rec[k] for k in ("rc", "retrieval_budget", "n_units", "n_bundles", "n_reports", "n_pool_provenance_ok", "acq_audit_ok", "infra_markers", "failure_markers", "kcap_audit", "ranking")})
PYEOF
echo "### end=$(date) rc=$RC"
exit $RC
