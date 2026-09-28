#!/bin/bash
# Runs one published pipeline over the evaluation tasks, one task per network-guarded namespace,
# with its model calls through the recording proxy and its retrieval through the pool service
# (EVIDENCE=pool for same pool, EVIDENCE=ref for fixed input). Usage: sbatch [--gres=gpu:1]
# --export=ALL,BL=<arm>,EVIDENCE=pool|ref campaign.slurm (the GPU is for autosurvey and surveyforge).
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
set -uo pipefail

need() { [ -n "${!1:-}" ] || { echo "[camp] FATAL(95): $1 is not set" >&2; exit 95; }; }
for v in LITREVIEW_ROOT RUNS_ROOT BASELINE_INPUTS TASKLIST CUTOFFS POOL_INDEX POOL_GOLD_DIR POOL_MANIFEST HOSTPY OPENAI_KEY_FILE SCRIBE_RUN_PY; do need "$v"; done
H="$LITREVIEW_ROOT/scribe/harness"
GUARD="$H/runners/egress_guard.sh"
PROXY_PY="$H/runners/llm_proxy.py"
BRIDGE="$H/runners/sock_bridge.py"
TAP="$LITREVIEW_ROOT/biolitbench/pool/embed/tap"
SERVING_PROOF="$H/baselines/serving_proof.py"
POOL_SERVICE="$LITREVIEW_ROOT/biolitbench/pool/pool_service.py"

BL="${BL:?BL not set (autosurvey|surveyforge|sgi|lira|llmxmr|surveyg)}"
SLICE="${SLICE:-}"
LIMIT="${LIMIT:-100000}"
JOBID="${SLURM_JOB_ID:-local$$}"

EVIDENCE="${EVIDENCE:-pool}"
case "$EVIDENCE" in pool|ref) : ;; *) echo "[camp] EVIDENCE must be pool|ref" >&2; exit 2 ;; esac
if [ "$EVIDENCE" = ref ]; then
  [ "$BL" = surveyforge ] && { echo "[camp] surveyforge runs under same pool only (EVIDENCE=pool)" >&2; exit 2; }
  need ALLOW_DIR
  CAMPAIGN="${CAMPAIGN:-campaign50_ref}"
else
  CAMPAIGN="${CAMPAIGN:-campaign50}"
fi
OUTROOT="$RUNS_ROOT/$CAMPAIGN/$BL"
ARMDIR="$OUTROOT/_arm"
IN="$BASELINE_INPUTS"
case "$BL" in
  autosurvey|surveyforge|sgi|lira|llmxmr|surveyg) : ;;
  *) echo "[camp] unknown BL=$BL (nothing created)" >&2; exit 2 ;;
esac
mkdir -p "$ARMDIR" "$RUNS_ROOT/$CAMPAIGN/_logs"

API_BASE="${API_BASE:-https://api.openai.com/v1}"
API_MODEL="gpt-5.6-luna"
API_TIER="${API_TIER:-flex}"
API_EFFORT="none"
PRICE_IN="${PRICE_IN:-0.10}"
PRICE_OUT="${PRICE_OUT:-0.60}"
COST_FILE="${COST_FILE:-$RUNS_ROOT/$CAMPAIGN/openai_costs_$CAMPAIGN.jsonl}"
BUDGET_USD="${BUDGET_USD:-50}"
WARN_USD="${WARN_USD:-5}"
KEY_FILE="$OPENAI_KEY_FILE"
LANE_DIR="${LANE_DIR:-/tmp/bl_lanes/$JOBID}"

APP_PORT="${APP_PORT:-18080}"
case "$BL" in
  autosurvey)  PROXY_BASE=31100; POOL_PORT=31900 ;;
  surveyforge) PROXY_BASE="${SF_PROXY_BASE:-32100}"; POOL_PORT="${SF_POOL_PORT:-32900}" ;;
  sgi)         PROXY_BASE=33100; POOL_PORT=33900 ;;
  lira)        PROXY_BASE=35100; POOL_PORT=35900 ;;
  llmxmr)      PROXY_BASE=36100; POOL_PORT=36900 ;;
  surveyg)     PROXY_BASE=37100; POOL_PORT=37900 ;;
esac
case "$BL" in
  autosurvey)
    ARM_PY="${PY_AUTOSURVEY:?PY_AUTOSURVEY not set}"; ARM_PYVER=3.12
    ARM_APP="${APP_AUTOSURVEY:?APP_AUTOSURVEY not set}"
    ARM_CLONE="$ARM_APP"; ARM_DEVPATCH="${PATCH_AUTOSURVEY:?PATCH_AUTOSURVEY not set}" ;;
  surveyforge)
    ARM_PY="${PY_SURVEYFORGE:?PY_SURVEYFORGE not set}"; ARM_PYVER=3.11
    ARM_APP="${APP_SURVEYFORGE:?APP_SURVEYFORGE not set}"
    ARM_CLONE="$ARM_APP"; ARM_DEVPATCH="" ;;
  sgi)
    ARM_PY="${PY_SGI:?PY_SGI not set}"; ARM_PYVER=3.10
    ARM_APP="${APP_SGI:?APP_SGI not set}"
    ARM_CLONE="$ARM_APP"; ARM_DEVPATCH="${PATCH_SGI:?PATCH_SGI not set}" ;;
  lira)
    ARM_PY="${PY_LIRA:?PY_LIRA not set}"; ARM_PYVER=3.10
    ARM_APP="${APP_LIRA:?APP_LIRA not set}"
    ARM_CLONE="$ARM_APP"; ARM_DEVPATCH="${PATCH_LIRA:?PATCH_LIRA not set}" ;;
  llmxmr)
    ARM_PY="${PY_LLMXMR:?PY_LLMXMR not set}"; ARM_PYVER=3.10
    ARM_CLONE="${APP_LLMXMR:?APP_LLMXMR not set}"
    ARM_APP="$ARM_CLONE/LLMxMapReduce_V2"
    ARM_DEVPATCH="${PATCH_LLMXMR:?PATCH_LLMXMR not set}" ;;
  surveyg)
    ARM_PY="${PY_SURVEYG:?PY_SURVEYG not set}"; ARM_PYVER=3.11
    ARM_APP="${APP_SURVEYG:?APP_SURVEYG not set}"
    ARM_CLONE="$ARM_APP"; ARM_DEVPATCH="${PATCH_SURVEYG:?PATCH_SURVEYG not set}" ;;
esac

case "$BL" in
  autosurvey)  PIN_COMMIT=5e8f389f3d51b29bad16dc6ae75db3e8a45a3b65; PIN_DIFF=d9d6637005bb76a0800e6fe56062aa6999ffb77e39eeb7812b8a3cc7e37ad850 ;;
  surveyforge) PIN_COMMIT=9114a0b7895a0f7eb614938d9bc0c956cf25245b; PIN_DIFF=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 ;;
  sgi)         PIN_COMMIT=92f84ba3bbfb6ea3437d23fa773d90e6c96e1997
               if [ "$EVIDENCE" = ref ]; then PIN_DIFF=73c3d44038ca19355a70580154c8035d1d2d2f900cdf628c1ee51d1ca497fee2
               else PIN_DIFF=791e8821798d3922d98de72e142e23eb143edce7df13a59d3a79e498c22d2d05; fi ;;
  surveyg)     PIN_COMMIT=089a05ba0fd870cc4c2285e762e1db3bd489884c; PIN_DIFF=6ccbe8c1f36d84ee59742168ab7060dd94f68d77fd9dd4a8c3ed2050ef19a431 ;;
  llmxmr)      PIN_COMMIT=0e93cc9ef8b89d6b0c75032b24d219dc15117908; PIN_DIFF=e987aaa333b82462c80f196dde5668a284543e5060de02e3bdd95ec6eb64272f ;;
  lira)        PIN_COMMIT=2bc77a6e7b30586343119bd104cc07ebaec380a1; PIN_DIFF=c6bcd821306533130206e6427683b452a4684c81872698a53459271a5d936a7b ;;
esac

case "$BL" in
  autosurvey|surveyforge) ARM_GPUS=1 ;;
  *)                      ARM_GPUS=0 ;;
esac

PAR="${PAR:-16}"
[ "$BL" = "llmxmr" ] && PAR="${PAR_LLMXMR:-1}"

APP_MODEL_NAME="$API_MODEL"

echo "[camp] ================================================================"
echo "[camp] arm=$BL evidence=$EVIDENCE host=$(hostname) job=$JOBID"
echo "[camp] slice=${SLICE:-all} limit=$LIMIT par=$PAR gpus_needed=$ARM_GPUS"
echo "[camp] app=$ARM_APP"
echo "[camp] py =$ARM_PY"
echo "[camp] out=$OUTROOT"
echo "[camp] serving=$API_MODEL tier=$API_TIER effort=$API_EFFORT budget=\$$BUDGET_USD cost=$COST_FILE"
echo "[camp] ================================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "[camp] (no GPU visible)"
[ -x "$ARM_PY" ] || { echo "[camp] FATAL(90): interpreter not executable: $ARM_PY" >&2; exit 90; }
GOTVER="$("$ARM_PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
[ "$GOTVER" = "$ARM_PYVER" ] || {
  echo "[camp] FATAL(90): $ARM_PY is python $GOTVER, expected $ARM_PYVER" >&2; exit 90; }
echo "[camp] interpreter OK: $ARM_PY (python $GOTVER)"
[ -d "$ARM_APP" ] || { echo "[camp] FATAL(90): app tree missing: $ARM_APP" >&2; exit 90; }

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:?HF_HOME not set (the offline model cache)}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export SENTENCE_TRANSFORMERS_HOME="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
[ -n "${TIKTOKEN_CACHE_DIR:-}" ] && export TIKTOKEN_CACHE_DIR
[ -n "${PLAYWRIGHT_BROWSERS_PATH:-}" ] && export PLAYWRIGHT_BROWSERS_PATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export OPENAI_API_KEY=dummy
unset SERP_API_KEY SERPER_API_KEY JINA_API_KEY 2>/dev/null || true


POOL_BASE="http://127.0.0.1:$POOL_PORT"

REPO_ARGS=(--repo "$LITREVIEW_ROOT")
[ -n "$ARM_CLONE" ] && REPO_ARGS+=(--repo "$ARM_CLONE")

now_s() { date +%s; }

json_field() {
  "$ARM_PY" - "$1" "$2" <<'DONE' 2>/dev/null
import json, sys
print(json.load(open(sys.argv[1]))[sys.argv[2]])
DONE
}

json_ok() {
  "$ARM_PY" - "$1" "$2" <<'DONE' >/dev/null 2>&1
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if str(d.get(sys.argv[2], "")).strip() else 1)
DONE
}

record_status() {
  "$HOSTPY" - "$ARMDIR/task_status.jsonl" "$1" "$2" "$3" "${4:-}" <<'DONE'
import json, sys, time
p, task, status, secs, note = sys.argv[1:6]
with open(p, "a") as f:
    f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                        "task": task, "status": status,
                        "wall_s": float(secs), "note": note}) + "\n")
DONE
}

tap_engaged() {
  local e="$1/stderr.log"; shift
  [ -s "$e" ] || return 1
  grep -qE '^\[[a-z_]+\] (disabled|FATAL):' "$e" && return 1
  local p; for p in "$@"; do grep -qE -- "$p" "$e" || return 1; done
}

bl_engaged() { return 0; }
bl_done() { return 1; }
bl_artifact() { bl_done "$1"; }
bl_run()  { return 1; }

case "$BL" in

  autosurvey)
    AS_DB="${AUTOSURVEY_DB:?AUTOSURVEY_DB not set}"
    AS_EMB="${AUTOSURVEY_EMBED_MODEL:?AUTOSURVEY_EMBED_MODEL not set (nomic-embed-text-v1 dir)}"
    export PYTHONPATH="$TAP:$ARM_APP"
    bl_engaged() { local t; t="$(basename "$1")"
      local view="[0-9]{4}\\.$t\\.npy"; [ "$EVIDENCE" = ref ] && view="$t\\.autosurvey\\.idx\\.npy"
      tap_engaged "$1" '^\[pool_selector\] cutoff selector applied to src\.database\.database$' \
        "^\\[pool_selector\\] cutoff selector: [0-9]+ ids from $view\$" '^\[pool_docstore\] patched [0-9]+ name'; }
    bl_artifact() { local o="$1"; local f
      for f in "$o"/*.json; do [ -e "$f" ] || continue; json_ok "$f" survey && return 0; done; return 1; }
    bl_done() { bl_artifact "$1" && bl_engaged "$1"; }
    bl_run() {
      local t="$1" o="$2" p="$3" c="$4"
      local refs="$IN/$t/refs.json"
      local topic; topic="$(json_field "$refs" topic)"
      if [ "$EVIDENCE" = ref ]; then
        IDXNPY="$ALLOW_DIR/$t.$BL.idx.npy"
        [ -f "$IDXNPY" ] || { echo "[camp] FATAL: index-space allowlist missing: $IDXNPY" >&2; return 1; }
        export POOL_SELECTOR_CUTOFF_NPY="$IDXNPY"
      else export POOL_SELECTOR_CUTOFF_NPY="${CUTOFF_DIR:?CUTOFF_DIR not set}/$c.$t.npy"; fi
      export RETRIEVAL_TAP_LOG="$o/retrieval_tap.jsonl"
      cd "$ARM_APP" || return 1
      CUDA_VISIBLE_DEVICES="${EMBED_GPU:-0}" \
      timeout -s KILL "${TASK_TIMEOUT:-14400}" "$ARM_PY" main.py \
        --topic "$topic" --saving_path "$o" --model "$APP_MODEL_NAME" \
        --api_url "http://127.0.0.1:$p/v1/chat/completions" --api_key dummy \
        --db_path "$AS_DB" --embedding_model "$AS_EMB" \
        --section_num 7 --subsection_len 700 --rag_num 60 --outline_reference_num 1500 \
        > "$o/stdout.log" 2> "$o/stderr.log"
    } ;;

  surveyforge)
    SF_DB="${SURVEYFORGE_DB:?SURVEYFORGE_DB not set}"
    SF_EMB="${SURVEYFORGE_EMBED_MODEL:?SURVEYFORGE_EMBED_MODEL not set (gte-large-en-v1.5 dir)}"
    SF_OUTLINES="${SURVEYFORGE_OUTLINE_PATH:-$ARM_APP}"
    export PYTHONPATH="$TAP:$ARM_APP"
    bl_engaged() { local t; t="$(basename "$1")"
      tap_engaged "$1" '^\[pool_selector\] cutoff selector applied to src\.database\.database$' \
      "^\\[pool_selector\\] cutoff selector: [0-9]+ ids from [0-9]{4}\\.$t\\.npy\$" \
      '^\[pool_selector\] cutoff selector applied to src\.database\.database_survey$' \
      '^\[pool_selector\] cutoff post-filter applied to ' '^\[pool_selector\] survey-db selector: ' \
      '^\[pool_docstore\] patched [0-9]+ name'; }
    bl_artifact() { local o="$1"; local f
      for f in "$o"/*.json; do [ -e "$f" ] || continue; json_ok "$f" survey && return 0; done; return 1; }
    bl_done() { bl_artifact "$1" && bl_engaged "$1"; }
    bl_run() {
      local t="$1" o="$2" p="$3" c="$4"
      local refs="$IN/$t/refs.json"
      local topic; topic="$(json_field "$refs" topic)"
      export POOL_SELECTOR_CUTOFF_NPY="${CUTOFF_DIR:?CUTOFF_DIR not set}/$c.$t.npy"
      export RETRIEVAL_TAP_LOG="$o/retrieval_tap.jsonl"
      cd "$ARM_APP" || return 1
      CUDA_VISIBLE_DEVICES="${EMBED_GPU:-0}" \
      timeout -s KILL "${TASK_TIMEOUT:-14400}" "$ARM_PY" main.py \
        --topic "$topic" --saving_path "$o" --model "$APP_MODEL_NAME" \
        --api_url "http://127.0.0.1:$p/v1/chat/completions" --api_key dummy \
        --db_path "$SF_DB" --embedding_model "$SF_EMB" \
        --section_num 7 --subsection_len 500 --rag_num 100 --rag_max_out 60 \
        --outline_reference_num 1500 --survey_outline_path "$SF_OUTLINES" \
        > "$o/stdout.log" 2> "$o/stderr.log"
    } ;;

  sgi)
    export CUDA_VISIBLE_DEVICES=""
    bl_done() { local o="$1"
      [ -s "$o/acm/final_survey_refined.md" ] && return 0
      [ -s "$o/acm/final_survey_original.md" ] && return 0
      return 1; }
    bl_run() {
      local t="$1" o="$2" p="$3" c="$4"
      local refs="$IN/$t/refs.json"
      local topic; topic="$(json_field "$refs" topic)"
      mkdir -p "$o/_harness"
      "$ARM_PY" - "$ARM_APP/config.json" "$o/_harness/config.json" "${SGI_MODELS_DIR:?SGI_MODELS_DIR not set}" <<'PY'
import json, sys
src, dst, models = sys.argv[1:4]
cfg = json.load(open(src))
def fix(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and v.startswith("MODELS_DIR/"):
                node[k] = v.replace("MODELS_DIR", models, 1)
            else:
                fix(v)
    elif isinstance(node, list):
        for v in node:
            fix(v)
fix(cfg)
json.dump(cfg, open(dst, "w"), indent=2)
PY
      export S2_BASE_URL="$POOL_BASE/t/$t/graph/v1"
      export POOL_S2_PDF=local
      export LOCAL_LLM_URL="http://127.0.0.1:$p/v1" LOCAL_LLM_MODEL="$APP_MODEL_NAME" LOCAL_LLM_KEY=dummy
      cd "$ARM_APP" || return 1
      timeout -s KILL "${TASK_TIMEOUT:-14400}" "$ARM_PY" run_pipeline.py \
        --config "$o/_harness/config.json" --topic "$topic" --output "$o" \
        > "$o/stdout.log" 2> "$o/stderr.log"
    } ;;

  lira)
    LIRA_SRC="${LIRA_INPUT:-}"
    export CUDA_VISIBLE_DEVICES=""
    bl_done() { local o="$1"
      [ -s "$o/articles/lira_abs_same_noresearch/papers_srg_subset_125_42.csv" ]; }
    bl_run() {
      local t="$1" o="$2" p="$3" c="$4"
      [ -f "$LIRA_SRC" ] || { echo "[camp] LIRA_INPUT not set or missing: $LIRA_SRC" >&2; return 1; }
      mkdir -p "$o/data/scireviewgen"
      "$ARM_PY" - "$LIRA_SRC" "$t" "$o/data/scireviewgen/full_data_abs.json" <<'PY' || return 1
import json, sys
src, task, dst = sys.argv[1:4]
recs = [r for r in json.load(open(src)) if r.get("task_id") == task]
if len(recs) != 1:
    sys.exit("expected exactly 1 record for %s, got %d" % (task, len(recs)))
json.dump(recs, open(dst, "w"))
PY
      export LOCAL_LLM_URL="http://127.0.0.1:$p/v1" LOCAL_LLM_MODEL="$APP_MODEL_NAME"
      export OPENAI_BASE_URL="http://127.0.0.1:$p/v1"
      cd "$ARM_APP" || return 1
      timeout -s KILL "${TASK_TIMEOUT:-14400}" "$ARM_PY" main.py \
        base.data_dir="$o/data" base.final_papers_dir="$o/articles" base.temp_dir="$o/temp" \
        base.writer_model="$APP_MODEL_NAME" base.editor_model="$APP_MODEL_NAME" \
        base.reviewer_model="$APP_MODEL_NAME" base.researcher_model=none \
        base.do_rouge=False base.do_recall=False base.do_citation=False base.do_prometheus=False \
        hydra.run.dir="$o/hydra" \
        > "$o/stdout.log" 2> "$o/stderr.log"
    } ;;

  llmxmr)
    export PYTHONPATH="$ARM_APP"
    export PROMPT_LANGUAGE=en
    export CUDA_VISIBLE_DEVICES=""
    bl_done() { local o="$1"
      local f="$o/survey.jsonl"; [ -s "$f" ] || return 1
      "$ARM_PY" - "$f" <<'PY' >/dev/null 2>&1
import json, sys
line = [l for l in open(sys.argv[1]) if l.strip()][-1]
sys.exit(0 if json.loads(line).get("content", "").strip() else 1)
PY
    }
    bl_run() {
      local t="$1" o="$2" p="$3" c="$4"
      local refs="$IN/$t/refs.json"
      local topic; topic="$(json_field "$refs" topic)"
      local outf="$o/survey.jsonl"
      export OPENAI_API_BASE="http://127.0.0.1:$p/v1"
      export BING_SEARCH_V7_ENDPOINT="$POOL_BASE/t/$t/v7.0/search"
      export BING_SEARCH_V7_SUBSCRIPTION_KEY=dummy
      unset SERP_API_KEY 2>/dev/null || true
      cd "$ARM_APP" || return 1
      setsid timeout -s KILL "${LLMXMR_TIMEOUT:-14400}" "$ARM_PY" ./src/start_pipeline.py \
        --topic "$topic" --output_file "$outf" \
        --config_file ./config/model_config_openai.json --block_count 1 \
        > "$o/stdout.log" 2> "$o/stderr.log" &
      local child=$!
      ( while kill -0 "$child" 2>/dev/null; do
          if bl_done "$o"; then
            sleep 20
            local pgid; pgid=$(ps -o pgid= -p "$child" 2>/dev/null | tr -d ' ')
            [ -n "$pgid" ] && { echo "watchdog: output complete, reaping pgid $pgid" >> "$o/stdout.log"; kill -KILL -"$pgid" 2>/dev/null; }
            break
          fi
          sleep 30
        done ) &
      wait "$child" 2>/dev/null
      if grep -q "Error in parsing skeleton" "$o/stderr.log" "$o/stdout.log" 2>/dev/null; then
        echo '{"skeleton_parse_failed": true}' > "$o/_harness_flags.json"
        echo "[camp] NOTE $t: skeleton parse failed -- survey built from a synthesised one-node outline"
      else
        echo '{"skeleton_parse_failed": false}' > "$o/_harness_flags.json"
      fi
      bl_done "$o"
    } ;;

  surveyg)
    export PYTHONPATH="$ARM_APP"
    export API_KEY=dummy EMBED_TOKEN=dummy SEMANTIC_SCHOLAR_API_KEY=dummy LOCAL_LLM_KEY=dummy
    export CUDA_VISIBLE_DEVICES=""
    SG_NUM_PAPERS="${SURVEYG_NUM_PAPERS:-100}"
    sg_dirkey() {
      local s="${1// /_}"; printf '%s' "${s//:/}"; }
    bl_done() { local o="$1"; [ -s "$o/literature_review.md" ]; }
    bl_run() {
      local t="$1" o="$2" p="$3" c="$4"
      local gm="$IN/$t/refs.json"
      local topic; topic="$(json_field "$gm" topic)"
      local dq; dq="$(sg_dirkey "$topic")"
      local sd="$ARM_APP/paper_data/$dq"
      local out="$sd/literature_review_output_gpt"
      export LOCAL_LLM_URL="http://127.0.0.1:$p/v1" LOCAL_LLM_MODEL="$APP_MODEL_NAME"
      export S2_SEARCH_URL="$POOL_BASE/t/$t/graph/v1/paper/search"
      export S2_BATCH_URL="$POOL_BASE/t/$t/graph/v1/paper/batch"
      rm -rf "$sd"
      cd "$ARM_APP" || return 1
      local lg="$o/stdout.log"; : > "$lg"
      timeout -s KILL "${TASK_TIMEOUT:-14400}" bash -c '
        set -e
        PY="$1"; T="$2"; KW="$3"; N="$4"; SD="$5"
        $PY scripts/survey_crawler.py        "$T" "$KW" "$N"
        [ -s "$SD/info/crawl_papers.json" ] || { echo "FAIL crawl-empty (exit 0 but no crawl_papers.json)"; exit 11; }
        $PY scripts/create_survey_graph.py   "$T"
        [ -s "$SD/info/paper_citation_graph.json" ] || { echo "FAIL graph-empty (exit 0 but no graph written)"; exit 12; }
        $PY scripts/fetch_cited_by_batch.py  "$T"
        $PY scripts/pdf_downloader.py        "$T" || echo "WARN pdf_downloader rc=$?"
        $PY writing/summarize.py             "$T"
        $PY scripts/traversal.py             "$T" "_gpt"
        $PY writing/writing_survey_gpt.py    "$T" "_gpt"
      ' _ "$ARM_PY" "$topic" "$topic" "$SG_NUM_PAPERS" "$sd" >> "$lg" 2> "$o/stderr.log"
      local rc=$?
      if [ -s "$out/literature_review.md" ]; then
        cp -r "$out"/. "$o/" || { echo "FAIL copy $t"; return 1; }
      fi
      cp -r "$sd/info" "$o/_app_info" 2>/dev/null || true
      mkdir -p "$o/_app_logs"
      cp "$ARM_APP"/outputs/logs/*.log "$o/_app_logs"/ 2>/dev/null || true
      git -C "$ARM_APP" checkout -q -- outputs/logs/ 2>/dev/null || true
      bl_done "$o"
    } ;;
esac

if [ -n "${CAMPAIGN_TASK_BODY:-}" ]; then
  T="$CAMPAIGN_TASK_BODY"; OUTD="$CAMPAIGN_TASK_OUT"; CUTOFF="$CAMPAIGN_TASK_CUTOFF"
  APORT="$CAMPAIGN_TASK_APPPORT"

  case "$BL" in sgi) export POOL_S2_PDF=local ;; esac
  POOL_ALLOWLIST_DIR="$( [ "$EVIDENCE" = ref ] && printf %s "$ALLOW_DIR" )" \
  POOL_INDEX="$POOL_INDEX" POOL_CUTOFFS="$CUTOFFS" POOL_GOLD_DIR="$POOL_GOLD_DIR" \
  POOL_EDGES="${POOL_EDGES:?POOL_EDGES not set}" \
  POOL_LOG="$OUTD/search_calls.jsonl" PYTHONPATH="${POOL_PYTHONPATH:-}" \
    "$HOSTPY" "$POOL_SERVICE" --host 127.0.0.1 --port "$POOL_PORT" \
    > "$OUTD/pool_service.log" 2>&1 &
  TPOOL=$!
  trap 'kill $TPOOL 2>/dev/null' EXIT
  PH=0
  for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$POOL_PORT/healthz" > "$OUTD/pool_health.json" 2>/dev/null && { PH=1; break; }
    kill -0 "$TPOOL" 2>/dev/null || { echo "[task] pool service died at startup"; tail -20 "$OUTD/pool_service.log"; exit 93; }
    sleep 2
  done
  [ "$PH" = 1 ] || { echo "[task] pool service never became healthy"; tail -20 "$OUTD/pool_service.log"; exit 93; }

  DECL="$APORT,$POOL_PORT"
  "$HOSTPY" "$BRIDGE" --assert-listeners "$DECL" \
      --census-out "$OUTD/listener_census.json" > /dev/null || {
    echo "[task] REFUSING: in-namespace listener set != declared ($DECL); see $OUTD/listener_census.json"
    exit 8; }

  bl_run "$T" "$OUTD" "$APORT" "$CUTOFF"
  rc=$?
  kill $TPOOL 2>/dev/null
  exit $rc
fi
if [ "$ARM_GPUS" = 1 ]; then
  { nvidia-smi -L >/dev/null 2>&1 && [ "${CUDA_VISIBLE_DEVICES-all}" != "" ]; } || {
    echo "[camp] FATAL(90): arm $BL needs one GPU for its embedder; submit with --gres=gpu:1" >&2; exit 90; }
fi

INTEG="$ARMDIR/integrity.json"
HEADSHA="$(git -C "$ARM_CLONE" rev-parse HEAD 2>/dev/null || echo NONE)"
DIFFSHA="$(git -C "$ARM_CLONE" diff 2>/dev/null | sha256sum | cut -d' ' -f1)"
if [ -n "$ARM_DEVPATCH" ]; then
  ARTSHA="$( [ -f "$ARM_DEVPATCH" ] && sha256sum "$ARM_DEVPATCH" | cut -d' ' -f1 || echo MISSING)"
else
  ARTSHA="$(printf '' | sha256sum | cut -d' ' -f1)"
fi
MATCH=0
[ "$HEADSHA" = "$PIN_COMMIT" ] && [ "$DIFFSHA" = "$PIN_DIFF" ] && [ "$ARTSHA" = "$PIN_DIFF" ] && MATCH=1
"$ARM_PY" - "$INTEG" "$BL" "$ARM_APP" "$ARM_CLONE" "$HEADSHA" "$PIN_COMMIT" "$DIFFSHA" "$ARTSHA" "$PIN_DIFF" "${ARM_DEVPATCH:-}" "$MATCH" <<'PY'
import json, sys
out, bl, app, clone, head, pin_head, diff, art, pin_diff, patch, match = sys.argv[1:12]
json.dump({"schema": "campaign_integrity/1.1", "arm": bl, "app": app, "clone": clone,
           "head": head, "pinned_commit": pin_head,
           "worktree_diff_sha256": diff, "patch": patch or None, "patch_sha256": art,
           "pinned_patch_sha256": pin_diff, "match": match == "1"},
          open(out, "w"), indent=1)
PY
if [ "$MATCH" != 1 ]; then
  echo "[camp] FATAL(91): $ARM_CLONE does not match its pinned code." >&2
  echo "[camp]   HEAD            = $HEADSHA (pinned $PIN_COMMIT)" >&2
  echo "[camp]   git diff sha256 = $DIFFSHA (pinned $PIN_DIFF)" >&2
  echo "[camp]   patch sha256    = $ARTSHA (${ARM_DEVPATCH:-no patch})" >&2
  echo "[camp]   apply exactly the deviation patch to a clean clone at the pinned commit" >&2
  exit 91
fi
echo "[camp] integrity OK: $(basename "$ARM_CLONE") @ ${HEADSHA:0:12}, diff sha256 ${DIFFSHA:0:12}"

[ -f "$TASKLIST" ] || { echo "[camp] FATAL(95): $TASKLIST missing" >&2; exit 95; }
[ -f "$CUTOFFS" ]  || { echo "[camp] FATAL(95): $CUTOFFS missing" >&2; exit 95; }
mapfile -t ALLTASKS < <("$ARM_PY" - "$TASKLIST" <<'PY'
import json, sys
print("\n".join(json.load(open(sys.argv[1]))["tasks"]))
PY
)
[ "${#ALLTASKS[@]}" -gt 0 ] || { echo "[camp] FATAL(95): empty task list" >&2; exit 95; }
LO=0; HI=$(( ${#ALLTASKS[@]} - 1 ))
if [ -n "$SLICE" ]; then LO="${SLICE%%-*}"; HI="${SLICE##*-}"; fi
TASKS=()
for i in $(seq "$LO" "$HI"); do
  [ "$i" -lt "${#ALLTASKS[@]}" ] || break
  [ "${#TASKS[@]}" -lt "$LIMIT" ] || break
  TASKS+=("${ALLTASKS[$i]}")
done
echo "[camp] tasks: ${#TASKS[@]} of ${#ALLTASKS[@]} (indices $LO..$HI)"

declare -A CUT
while IFS=$'\t' read -r k v; do CUT["$k"]="$v"; done < <("$ARM_PY" - "$CUTOFFS" <<'PY'
import json, sys
for k, v in json.load(open(sys.argv[1])).items():
    print("%s\t%s" % (k, v))
PY
)
for t in "${TASKS[@]}"; do
  [ -n "${CUT[$t]:-}" ] || { echo "[camp] FATAL(95): no cutoff for $t in $CUTOFFS" >&2; exit 95; }
done
echo "[camp] cutoffs resolved for all ${#TASKS[@]} tasks"


PXPIDFILE="$ARMDIR/_proxy_pids.${SLURM_JOB_ID:-$$}"; : > "$PXPIDFILE"
BRPIDFILE="$ARMDIR/_bridge_pids.${SLURM_JOB_ID:-$$}"; : > "$BRPIDFILE"
cleanup() {
  for f in "$PXPIDFILE" "$BRPIDFILE"; do
    [ -f "$f" ] && while read -r p; do [ -n "$p" ] && kill "$p" 2>/dev/null; done < "$f"
  done
  rm -rf "$LANE_DIR" 2>/dev/null
}
trap cleanup EXIT INT TERM

proxy_args() {
  printf '%s\n' --listen "$1" --log "$2" \
    --upstream-url "$API_BASE" --api-key-file "$KEY_FILE" \
    --force-model "$API_MODEL" --service-tier "$API_TIER" --reasoning-effort "$API_EFFORT" \
    --price-in "$PRICE_IN" --price-out "$PRICE_OUT" --cost-file "$COST_FILE" \
    --budget-usd "$BUDGET_USD" --warn-usd "$WARN_USD"
}

start_host_proxy() {
  local p="$1" lg="$2"
  mkdir -p "$(dirname "$lg")"
  mapfile -t _pa < <(proxy_args "$p" "$lg")
  "$HOSTPY" "$PROXY_PY" "${_pa[@]}" >> "$ARMDIR/proxy.log" 2>&1 &
  local pid=$! j
  for j in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$p/__proxy_health" >/dev/null 2>&1 && { echo "$pid"; return 0; }
    curl -sf "http://127.0.0.1:$p/v1/models"      >/dev/null 2>&1 && { echo "$pid"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo ""; return 1; }
    sleep 1
  done
  echo ""; return 1
}

[ -f "$KEY_FILE" ] || { echo "[camp] FATAL(96): API key file not found: $KEY_FILE" >&2; exit 96; }
KPERM=$(stat -c '%a' "$KEY_FILE")
case "$KPERM" in
  600|400) : ;;
  *) echo "[camp] FATAL(96): $KEY_FILE is mode $KPERM; chmod 600 it. The proxy refuses a" >&2
     echo "[camp]   group/other-readable key file, and so does this runner." >&2; exit 96 ;;
esac
[ -x "$HOSTPY" ] || { echo "[camp] FATAL(96): $HOSTPY not executable" >&2; exit 96; }
[ -f "$BRIDGE" ] || { echo "[camp] FATAL(96): $BRIDGE missing" >&2; exit 96; }
mkdir -p "$(dirname "$COST_FILE")" "$LANE_DIR" || { echo "[camp] FATAL(96): cannot create $LANE_DIR" >&2; exit 96; }
touch "$COST_FILE" 2>/dev/null || { echo "[camp] FATAL(96): cost file not writable: $COST_FILE" >&2; exit 96; }
_probe_sock="$LANE_DIR/99.sock"
[ "${#_probe_sock}" -le 107 ] || { echo "[camp] FATAL(96): lane socket path ${#_probe_sock} > 107 bytes: $_probe_sock" >&2; exit 96; }
case "$BUDGET_USD" in ""|0|0.0) echo "[camp] FATAL(96): BUDGET_USD must be > 0" >&2; exit 96 ;; esac
SPENT=$("$HOSTPY" -c "
import json,sys
t=0.0
try:
  for l in open('$COST_FILE'):
      try: t+=json.loads(l).get('cost_usd',0.0)
      except Exception: pass
except FileNotFoundError: pass
print('%.4f'%t)")
echo "[camp] budget: \$$SPENT spent of \$$BUDGET_USD (warn at \$$WARN_USD), shared file $COST_FILE"

PROBE_PORT=$((PROXY_BASE + 900))
PROBEPX="$(start_host_proxy "$PROBE_PORT" "$ARMDIR/_probe_calls.jsonl")"
[ -n "$PROBEPX" ] || { echo "[camp] FATAL(92): probe proxy did not start; see $ARMDIR/proxy.log" >&2; exit 92; }
echo "$PROBEPX" >> "$PXPIDFILE"
"$HOSTPY" "$SERVING_PROOF" --api-mode \
  --out "$ARMDIR/serving_proof.json" \
  --api-base "$API_BASE" --probe-url "http://127.0.0.1:$PROBE_PORT/v1" \
  --api-key-file "$KEY_FILE" --force-model "$API_MODEL" \
  --service-tier "$API_TIER" --reasoning-effort "$API_EFFORT" \
  --price-in "$PRICE_IN" --price-out "$PRICE_OUT" \
  --budget-usd "$BUDGET_USD" --cost-file "$COST_FILE" \
  --python "$HOSTPY" \
  --pool-manifest "$POOL_MANIFEST" \
  "${REPO_ARGS[@]}"
rc=$?
kill "$PROBEPX" 2>/dev/null
[ "$rc" -eq 0 ] || { echo "[camp] FATAL(92): serving_proof --api-mode exited $rc -- the arm is ABORTED." >&2; exit 92; }
echo "[camp] serving_proof OK -> $ARMDIR/serving_proof.json"


run_task() {
  local t="$1" idx="$2"
  local o="$OUTROOT/$t"
  local c="${CUT[$t]}"
  local pport=$((PROXY_BASE + idx))

  if [ -e "$o" ]; then echo "[camp] skip $t (already run)"; return 0; fi
  mkdir -p "$o"

  local px bp sock t0 t1 rc
  px="$(start_host_proxy "$pport" "$o/_calls.jsonl")"
  if [ -z "$px" ]; then
    echo "[camp] FAIL $t: proxy did not come up on $pport (see $ARMDIR/proxy.log)"
    record_status "$t" proxy_fail 0 "port $pport"; return 1
  fi
  echo "$px" >> "$PXPIDFILE"

  t0="$(now_s)"
  sock="$LANE_DIR/$idx.sock"
  "$HOSTPY" "$BRIDGE" --unix-listen "$sock" --tcp-connect "127.0.0.1:$pport" \
    >> "$o/lane.log" 2>&1 &
  bp=$!
  echo "$bp" >> "$BRPIDFILE"
  local j ok=0
  for j in $(seq 1 60); do
    [ -S "$sock" ] && { ok=1; break; }
    kill -0 "$bp" 2>/dev/null || break
    sleep 0.5
  done
  if [ "$ok" != 1 ]; then
    echo "[camp] FAIL $t: lane socket never appeared at $sock"
    kill "$px" "$bp" 2>/dev/null; record_status "$t" lane_fail 0 "$sock"; return 1
  fi
  echo "$sock" > "$o/lane.path"

  CAMPAIGN_TASK_BODY="$t" CAMPAIGN_TASK_OUT="$o" CAMPAIGN_TASK_CUTOFF="$c" \
  CAMPAIGN_TASK_APPPORT="$APP_PORT" \
  EGRESS_LANE_SOCK="$sock" EGRESS_LANE_PORT="$APP_PORT" EGRESS_LANE_BRIDGE="$BRIDGE" \
  EGRESS_LANE_EXPECT_MODEL="$API_MODEL" EGRESS_LANE_EXPECT_TIER="$API_TIER" \
  EGRESS_DECLARED_PORTS="$APP_PORT" \
    "$GUARD" "$o/egress_proof.json" bash "$0"
  rc=$?
  kill "$bp" 2>/dev/null
  rm -f "$sock" 2>/dev/null
  t1="$(now_s)"
  kill "$px" 2>/dev/null

  if bl_done "$o"; then
    echo "[camp] OK   $t  rc=$rc  $((t1-t0))s  cutoff=$c  calls=$(wc -l < "$o/_calls.jsonl" 2>/dev/null || echo 0)"
    record_status "$t" ok "$((t1-t0))" "rc=$rc cutoff=$c"
    return 0
  fi
  case "$rc" in
    4)  echo "[camp] FAIL $t: GUARD REFUSED -- an external host was reachable inside the namespace" ;;
    7)  echo "[camp] FAIL $t: GUARD REFUSED -- the sanctioned lane did not verify" ;;
    8)  echo "[camp] FAIL $t: GUARD/CENSUS REFUSED -- unaccounted in-namespace listener" ;;
    86) echo "[camp] FAIL $t: the retrieval tap stopped the app (FATAL in $o/stderr.log)"
        record_status "$t" tap_fatal "$((t1-t0))" "rc=$rc cutoff=$c"; return 1 ;;
    93) echo "[camp] FAIL $t: pool service unhealthy" ;;
    *)  bl_artifact "$o" && ! bl_engaged "$o" && { echo "[camp] FAIL $t: the cutoff selector or docstore did not engage (see $o/stderr.log)"
          record_status "$t" not_engaged "$((t1-t0))" "rc=$rc cutoff=$c"; return 1; }
        echo "[camp] FAIL $t  rc=$rc  $((t1-t0))s  cutoff=$c  (final artifact absent or unparseable)" ;;
  esac
  record_status "$t" fail "$((t1-t0))" "rc=$rc cutoff=$c"
  return 1
}

budget_exhausted() {
  "$HOSTPY" -c "
import json,sys
t=0.0
try:
    for l in open('$COST_FILE'):
        try: t+=json.loads(l).get('cost_usd',0.0)
        except Exception: pass
except FileNotFoundError: pass
sys.exit(0 if t >= float('$BUDGET_USD') else 1)"
}

idx=0
for t in "${TASKS[@]}"; do
  if budget_exhausted; then
    echo "[camp] STOPPING: the shared budget cap \$$BUDGET_USD is reached ($COST_FILE)."
    echo "[camp] Remaining tasks are not attempted."
    break
  fi
  [ -d "$IN/$t" ] || { echo "[camp] no input dir for $t, skipping"; record_status "$t" no_input 0 ""; idx=$((idx+1)); continue; }
  run_task "$t" "$idx" &
  idx=$((idx+1))
  while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do sleep 5; done
done
wait

echo "[camp] ================================================================"
"$HOSTPY" - "$ARMDIR/task_status.jsonl" "$COST_FILE" api <<'DONE'
import collections, json, os, sys
p, cost, mode = sys.argv[1:4]
if os.path.exists(p):
    last = {}
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        last[r["task"]] = r
    c = collections.Counter(r["status"] for r in last.values())
    sec = [r["wall_s"] for r in last.values() if r["status"] == "ok"]
    print("[camp] tasks=%d  %s" % (len(last), dict(c)))
    if sec:
        sec.sort()
        print("[camp] ok wall_s: median=%.0f p90=%.0f max=%.0f"
              % (sec[len(sec) // 2], sec[int(len(sec) * 0.9)], sec[-1]))
else:
    print("[camp] no task_status.jsonl")
if mode == "api":
    t = 0.0
    n = 0
    try:
        for l in open(cost):
            try:
                t += json.loads(l).get("cost_usd", 0.0); n += 1
            except Exception:
                pass
    except FileNotFoundError:
        pass
    print("[camp] spend in %s: $%.4f over %d calls" % (cost, t, n))
DONE
echo "[camp] arm=$BL evidence=$EVIDENCE done. proofs+logs in $ARMDIR and each task dir"
