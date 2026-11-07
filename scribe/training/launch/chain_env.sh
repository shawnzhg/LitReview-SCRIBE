# Shared settings sourced by the planning-training chain scripts: repo paths, interpreters, backbone
# and its pin, embedding model, evaluation exclusions, skill file and carrier size.
LITREVIEW_ROOT=${LITREVIEW_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
[ -f "$LITREVIEW_ROOT/scribe/training/planning_task.py" ] || { echo "### FATAL: LITREVIEW_ROOT=$LITREVIEW_ROOT is not the repo" >&2; exit 2; }
LEARN_PY=${SCRIBE_TRAIN_PY:?set SCRIBE_TRAIN_PY (configs/site.env.example)}
SERVE_PY=${SCRIBE_SERVE_PY:?set SCRIBE_SERVE_PY (configs/site.env.example)}
EMB_PY=${SCRIBE_EMBED_PY:?set SCRIBE_EMBED_PY (configs/site.env.example)}
MODEL=${SCRIBE_MODEL_DIR:?set SCRIBE_MODEL_DIR (configs/site.env.example)}
BACKBONE_PIN=$LITREVIEW_ROOT/configs/models/Qwen3.8-27B-nothink.json
NOMIC_MODEL=${SCRIBE_EMBED_MODEL_DIR:?set SCRIBE_EMBED_MODEL_DIR (configs/site.env.example)}
EVAL_TASKS=${SCRIBE_EVAL_TASKS:?set SCRIBE_EVAL_TASKS (configs/site.env.example)}
EVAL_EXCLUDED=$LITREVIEW_ROOT/configs/train_tasks/eval_excluded.txt
SKILL=$LITREVIEW_ROOT/scribe/training/skills/planning_skill.txt
D_S=1408; R=32; EXPECT_PARAMS=14513156
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=${SCRIBE_HF_HOME:?set SCRIBE_HF_HOME (configs/site.env.example)} HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0
export PYTHONPATH=$LITREVIEW_ROOT/third_party/kvskill:$LITREVIEW_ROOT/scribe/training
export SCRIBE_EVAL_TASKS=$EVAL_TASKS SCRIBE_EVAL_EXCLUDED=$EVAL_EXCLUDED
export SCRIBE_BACKBONE_PIN=$BACKBONE_PIN
chain_fatal(){ echo "### FATAL: $1" >&2; exit "${2:-1}"; }
