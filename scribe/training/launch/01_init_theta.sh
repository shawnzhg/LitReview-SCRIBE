#!/bin/bash
# Step 1 of the planning-training chain: creates the initial carrier (zero slot values, so the
# backbone is unchanged) through kvskill.init_theta and checks its shape. Usage: OUT_ROOT=<dir> bash
# 01_init_theta.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/chain_env.sh"
OUT_ROOT=${OUT_ROOT:?OUT_ROOT=<run root of this chain>}
INIT_DIR=$OUT_ROOT/initial_carrier
[ -e "$INIT_DIR/meta.json" ] && chain_fatal "$INIT_DIR exists" 85
mkdir -p "$OUT_ROOT"
$LEARN_PY "$LITREVIEW_ROOT/scribe/training/backbone.py" check "$MODEL" --pin "$BACKBONE_PIN" || chain_fatal "$MODEL is not the pinned backbone" 3
$LEARN_PY -m kvskill.init_theta --out "$INIT_DIR" --model "$MODEL" --d_s "$D_S" --r "$R" --seed 0
$LEARN_PY - "$INIT_DIR" "$EXPECT_PARAMS" "$D_S" "$R" "$MODEL" <<'PY' || chain_fatal "init_theta output check failed" 91
import sys
from kvskill.artifact import KVSkillArtifact
path, expect, d_s, r, model = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
a = KVSkillArtifact.load(path)
m = a.meta
assert (int(m["cn_ds"]), int(m["cn_r"])) == (d_s, r), (m["cn_ds"], m["cn_r"])
assert m["base_model"] == model, m["base_model"]
assert float(a.cn_v.abs().max()) == 0.0, "the initial carrier must have V = 0"
assert a.num_params() == expect, f"carrier has {a.num_params():,} parameters, expected {expect:,}"
print(f"### initial carrier d_s={d_s} r={r} depths={m['cn_depths']} params={a.num_params():,} V=0")
PY
echo "### INIT_DIR=$INIT_DIR"
