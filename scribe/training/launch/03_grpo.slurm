#!/bin/bash
# Step 3 of the planning-training chain (Slurm GPU): trains the seeded planning carrier with GRPO for one
# epoch over the 192 training tasks and exports it. Usage: OUT_ROOT=<dir> BASE_CACHE=<dir>
# TASKS_FILE=<file> READOUT_TARGETS=<json> BANDS=<json> sbatch --export=ALL 03_grpo.slurm.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=400G --gres=gpu:1
#SBATCH --time=24:00:00
set -uo pipefail
source "${LITREVIEW_ROOT:?set LITREVIEW_ROOT (sbatch runs a spooled copy)}/scribe/training/launch/chain_env.sh"
OUT_ROOT=${OUT_ROOT:?OUT_ROOT=<run root of this chain>}
BASE_CACHE=${BASE_CACHE:?BASE_CACHE=<base runs of the training tasks>}
TASKS_FILE=${TASKS_FILE:?TASKS_FILE=<finalize_train_list.py output>}
READOUT_TARGETS=${READOUT_TARGETS:?READOUT_TARGETS=<reward_weights.py output>}
BANDS=${BANDS:?BANDS=<precompute_train_bands.py output for TASKS_FILE>}
INIT_DIR=$OUT_ROOT/initial_carrier
SEEDED_DIR=$OUT_ROOT/seed_planning/seeded
JOBN=${SLURM_JOB_ID:-0}
RUN=$OUT_ROOT/grpo_$JOBN
STEPS=48; PPS=4; GS=8; LR=0.0001; SEED=17; MICRO=1; GMU=0.85; MAX_NEW=8000; MAX_PROMPT=20000
N_TRAIN_ITEMS=192
PORT=$(( 9300 + JOBN % 100 )); EPORT=$(( 9400 + JOBN % 100 ))
fatal(){ echo "### FATAL: $1"; exit "${2:-89}"; }
echo "### launcher: $(readlink -f "$0") md5=$(md5sum < "$0" | cut -c1-32) job=$JOBN"

[ $(( STEPS * PPS )) -eq "$N_TRAIN_ITEMS" ] || fatal "STEPS x PPS = $((STEPS*PPS)) != $N_TRAIN_ITEMS: one epoch uses every training task once" 80
[ "$(grep -c . "$TASKS_FILE")" -eq "$N_TRAIN_ITEMS" ] || fatal "$TASKS_FILE does not list $N_TRAIN_ITEMS tasks" 80
$LEARN_PY "$LITREVIEW_ROOT/scribe/training/backbone.py" check "$MODEL" --pin "$BACKBONE_PIN" || fatal "not the pinned backbone" 3
[ -s "$SEEDED_DIR/../seed_summary.json" ] || fatal "no $SEEDED_DIR/../seed_summary.json (run 02_seed_init.slurm)" 84
mkdir -p "$RUN"
$LEARN_PY - "$INIT_DIR" "$SEEDED_DIR" "$EXPECT_PARAMS" "$D_S" "$R" "$READOUT_TARGETS" "$BANDS" "$TASKS_FILE" <<'PY' \
  || fatal "pre-flight failed (carrier / targets / bands / tasks)" 81
import sys, torch
from safetensors.torch import load_file
from kvskill.artifact import KVSkillArtifact
init, seeded, expect, d_s, r, targets, bands, tasks = sys.argv[1:9]
expect, d_s, r = int(expect), int(d_s), int(r)
a0, a1 = KVSkillArtifact.load(init), KVSkillArtifact.load(seeded)
for a in (a0, a1):
    assert (int(a.meta["cn_ds"]), int(a.meta["cn_r"])) == (d_s, r), (a.meta["cn_ds"], a.meta["cn_r"])
    assert a.num_params() == expect, a.num_params()
assert float(a0.cn_v.abs().max()) == 0.0, "the initial carrier must have V = 0 (it sizes the vLLM buffers)"
t0, t1 = load_file(init + "/tensors.safetensors"), load_file(seeded + "/tensors.safetensors")
diff = [k for k in t0 if k not in ("cn_v", "cn_k") and not torch.equal(t0[k], t1[k])]
assert not diff, f"seeded differs from the initial carrier beyond cn_v/cn_k: {diff}"
import readout_reward as RR, planning_exit_readouts as PE, planning_task as LP
T = RR.load_targets(targets)
assert "planning_exit" in T["window_sets"], "targets lack the planning_exit window set"
B = PE.load_bands(bands)
ids = [l.strip() for l in open(tasks) if l.strip()]
LP.refuse_eval(ids, f"03_grpo {tasks}")
bad = [t for t in ids if not (B["tasks"].get(t) or {}).get("complete")]
assert not bad, f"{len(bad)} tasks lack a complete train band: {bad[:5]}"
print(f"### pre-flight OK: carrier d_s={d_s} r={r} params={expect:,}; targets {T['_sha256_16']} "
      f"weights {({k: v['w'] for k, v in T['window_readouts']['planning_exit'].items()})}; "
      f"bands {B['_sha256_16']}; {len(ids)} tasks")
PY

if [ -n "${SCRIBE_TOOLCHAIN:-}" ]; then source "$SCRIBE_TOOLCHAIN" || fatal "toolchain" 99; fi
[ -n "${CUDA_HOME:-}" ] || fatal "CUDA_HOME is not set" 99
export PATH=$(dirname "$SERVE_PY"):$PATH VLLM_ALLOW_INSECURE_SERIALIZATION=1
export CUDA_MPS_PIPE_DIRECTORY=${TMPDIR:-/tmp}/mps-pipe-$JOBN CUDA_MPS_LOG_DIRECTORY=${TMPDIR:-/tmp}/mps-log-$JOBN
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS=0; nvidia-cuda-mps-control -d 2>/dev/null && MPS=1
PIDS=(); cleanup(){ for p in "${PIDS[@]:-}"; do kill -TERM -- -"$p" 2>/dev/null || kill "$p" 2>/dev/null; done
  [ $MPS = 1 ] && { echo quit | nvidia-cuda-mps-control 2>/dev/null || true; }; }; trap cleanup EXIT
wait_http(){ for i in $(seq 1 "$3"); do curl -sf -o /dev/null "$1" 2>/dev/null && return 0; sleep 5; done; echo "FATAL: $2 never came up"; return 1; }
INIT_PAYLOAD=$RUN/initial-payload.safetensors
$LEARN_PY -m kvskill.export --artifact_path "$INIT_DIR" --out "$INIT_PAYLOAD" || fatal "export of $INIT_DIR failed" 91
EDEV=cuda; [ $MPS = 1 ] || EDEV=cpu
setsid $EMB_PY "$LITREVIEW_ROOT/scribe/training/embed_service.py" --model "$NOMIC_MODEL" --batch 32 \
  --port "$EPORT" --device "$EDEV" > "$RUN/embed.log" 2>&1 & PIDS+=($!)
wait_http "http://127.0.0.1:$EPORT/health" nomic-embed 90 || { tail -20 "$RUN/embed.log"; exit 93; }
( unset PYTORCH_CUDA_ALLOC_CONF
  setsid $SERVE_PY -m kvskill.vllm_server --model "$MODEL" --theta_payload "$INIT_PAYLOAD" \
    --port "$PORT" --max_model_len 65536 --gpu_memory_utilization "$GMU" \
    --max_num_seqs 48 --seed 0 > "$RUN/server.log" 2>&1 & echo $! > "$RUN/server.pid" )
PIDS+=($(cat "$RUN/server.pid")); wait_http "http://127.0.0.1:$PORT/health" vllm 420 || { tail -40 "$RUN/server.log"; exit 94; }
grep -q "d_s=$D_S d_m=" "$RUN/server.log" || fatal "the rollout server did not arm the carrier branch at d_s=$D_S" 94
curl -s -X POST -H 'Content-Type: application/json' -d '{"level":1}' "http://127.0.0.1:$PORT/sleep"; echo

export SCRIBE_PLAN_TASKS_FILE=$TASKS_FILE SCRIBE_BASE_CACHE=$BASE_CACHE
export SCRIBE_PLAN_BANDS=$BANDS SCRIBE_READOUT_TARGETS=$READOUT_TARGETS
export SCRIBE_EMBED_URL=http://127.0.0.1:$EPORT
GRPO_ARGV=(
  "$LITREVIEW_ROOT/scribe/training/kv_entry.py" grpo
  --task planning --model_name "$MODEL" --artifact_path "$SEEDED_DIR" --out_dir "$RUN/grpo"
  --vllm_url "http://127.0.0.1:$PORT" --base_cache "$BASE_CACHE"
  --steps "$STEPS" --prompts_per_step "$PPS" --group_size "$GS" --lr "$LR" --seed "$SEED"
  --beta_kl 0.01 --beta_ent 0.05 --tis_cap 2.0 --learner_micro_seqs "$MICRO"
  --max_new_tokens "$MAX_NEW" --max_prompt_tokens "$MAX_PROMPT"
)
printf '%s\n' "${GRPO_ARGV[@]}" > "$RUN/grpo_argv.txt"
env | grep -E "^(SCRIBE_|PYTHONPATH)" | sort > "$RUN/launch_env.txt"
$LEARN_PY -u "${GRPO_ARGV[@]}" 2>&1 | tee "$RUN/learner.log"
RC=${PIPESTATUS[0]}; [ "$RC" = 0 ] || fatal "learner exited $RC" "$RC"

grep -q "planning-exit reward armed" "$RUN/learner.log" || fatal "the planning-exit reward never armed" 70
grep -q "items: train $N_TRAIN_ITEMS " "$RUN/learner.log" || fatal "planning did not build $N_TRAIN_ITEMS training items" 71
grep -q "\[backbone\] frozen" "$RUN/learner.log" || fatal "frozen-backbone guard line missing" 72
$LEARN_PY - "$RUN" "$STEPS" "$EXPECT_PARAMS" <<'PY' || fatal "post-run check failed" 73
import json, shutil, sys
from kvskill.artifact import KVSkillArtifact
run, steps, expect = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rows = [json.loads(l) for l in open(f"{run}/grpo/metrics.jsonl") if l.strip()]
aud = [r["base_params_with_grad"] for r in rows if "base_params_with_grad" in r]
assert aud and all(int(x) == 0 for x in aud), f"base_params_with_grad {aud}"
last = f"{run}/grpo/artifact-last"
m = json.load(open(f"{last}/meta.json"))
assert int(m.get("rl_step", -1)) == steps, f"artifact-last rl_step {m.get('rl_step')} != {steps}"
a = KVSkillArtifact.load(last)
assert a.num_params() == expect, a.num_params()
dst = f"{run}/final_planning_carrier"
shutil.copytree(last, dst)
print(f"### final carrier: rl_step={steps} params={a.num_params():,} -> {dst}")
PY
$LEARN_PY -m kvskill.export --artifact_path "$RUN/final_planning_carrier" --out "$RUN/final_planning_carrier-payload.safetensors"
echo "### SCRIBE_PLANNING_CARRIER=$RUN/final_planning_carrier"
