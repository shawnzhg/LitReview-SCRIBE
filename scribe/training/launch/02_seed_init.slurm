#!/bin/bash
# Step 2 of the planning-training chain (Slurm GPU): compiles the planning skill text into the
# initial carrier with kvskill.seed_init and checks that only the slot keys and values changed.
# Usage: OUT_ROOT=<dir> BASE_CACHE=<dir> TASKS_FILE=<file> sbatch --export=ALL 02_seed_init.slurm.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=200G --gres=gpu:1
#SBATCH --time=01:30:00
set -uo pipefail
source "${LITREVIEW_ROOT:?set LITREVIEW_ROOT (sbatch runs a spooled copy)}/scribe/training/launch/chain_env.sh"
OUT_ROOT=${OUT_ROOT:?OUT_ROOT=<run root of this chain>}
BASE_CACHE=${BASE_CACHE:?BASE_CACHE=<base runs of the training tasks>}
TASKS_FILE=${TASKS_FILE:?TASKS_FILE=<finalize_train_list.py output>}
INIT_DIR=$OUT_ROOT/initial_carrier
OUT=$OUT_ROOT/seed_planning
fatal() { echo "### FATAL: $1"; exit "${2:-1}"; }
echo "### launcher: $(readlink -f "$0") md5=$(md5sum < "$0" | cut -c1-32) job=${SLURM_JOB_ID:-manual}"
[ -f "$INIT_DIR/meta.json" ] || fatal "$INIT_DIR has no meta.json (run 01_init_theta.sh)" 84
[ -s "$SKILL" ] || fatal "SKILL $SKILL missing/empty" 82
[ -s "$TASKS_FILE" ] || fatal "TASKS_FILE $TASKS_FILE missing (run finalize_train_list.py)" 83
[ -e "$OUT/seeded/meta.json" ] && fatal "$OUT/seeded already exists" 85
$LEARN_PY "$LITREVIEW_ROOT/scribe/training/backbone.py" check "$MODEL" --pin "$BACKBONE_PIN" || fatal "not the pinned backbone" 3
mkdir -p "$OUT"; cp "$SKILL" "$OUT/skill.txt"
echo "### skill sha256-16 $(sha256sum "$SKILL" | cut -c1-16); tasks $TASKS_FILE sha256-16 $(sha256sum "$TASKS_FILE" | cut -c1-16) ($(wc -l < "$TASKS_FILE") ids)"
if [ -n "${SCRIBE_TOOLCHAIN:-}" ]; then source "$SCRIBE_TOOLCHAIN" || fatal "toolchain" 99; fi
export SCRIBE_BASE_CACHE=$BASE_CACHE SCRIBE_PLAN_TASKS_FILE=$TASKS_FILE
$LEARN_PY -u "$LITREVIEW_ROOT/scribe/training/kv_entry.py" seed_init \
  --base_init "$INIT_DIR" --skill "$OUT/skill.txt" --out "$OUT/seeded" --model "$MODEL" --fit_task planning \
  2>&1 | tee "$OUT/seed.log"
RC=${PIPESTATUS[0]}; [ "$RC" = 0 ] || exit "$RC"
$LEARN_PY - "$OUT/seeded" "$INIT_DIR" "$EXPECT_PARAMS" "$D_S" "$R" <<'PYR' || fatal "seeded carrier check failed" 88
import json, sys, torch
from safetensors.torch import load_file
from kvskill.artifact import KVSkillArtifact
seeded, init, expect, d_s, r = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
m = json.load(open(seeded + "/meta.json")); si = m.get("seed_init", {})
t = load_file(seeded + "/tensors.safetensors"); t0 = load_file(init + "/tensors.safetensors")
diff = [k for k in t0 if k not in ("cn_v", "cn_k") and k in t and not torch.equal(t[k], t0[k])]
print(f"### seeded ||cn_v||={float(t['cn_v'].float().norm()):.6f} rel_residual={si.get('fit', {}).get('rel_residual')} "
      f"inj_final={si.get('inj_final')}")
assert not diff, f"seed_init changed tensors other than cn_v/cn_k: {diff}"
assert len(t) == len(t0), "tensor count differs from the initial carrier"
assert (int(m["cn_ds"]), int(m["cn_r"])) == (d_s, r), (m["cn_ds"], m["cn_r"])
n = KVSkillArtifact.load(seeded).num_params()
assert n == expect, f"seeded carrier has {n:,} parameters, expected {expect:,}"
json.dump({"rel_residual": si.get("fit", {}).get("rel_residual"), "inj_final": si.get("inj_final"),
           "seed_tokens": si.get("seed_tokens"), "params": n, "d_s": d_s, "r": r},
          open(seeded + "/../seed_summary.json", "w"), indent=1)
PYR
echo "### SEEDED_DIR=$OUT/seeded"
