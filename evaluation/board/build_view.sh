#!/bin/bash
# Runs view_builder.py as a Slurm job, or in the current shell with LOCAL=1, to build the merged
# view of one external agent or one of our same-pool rows from its isolated scoring directory.
# Usage: build_view.sh <agent> <iso dir> | build_view.sh native <key> <iso dir>.
set -uo pipefail
LITREVIEW_ROOT=${LITREVIEW_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
[ -f "$LITREVIEW_ROOT/evaluation/board/view_builder.py" ] || { echo "FATAL: set LITREVIEW_ROOT to the repository root"; exit 2; }
export LITREVIEW_ROOT
AGENTS="openai_tool_loop claude_code claude_science gemini_deep_research elicit"
AGENT=${1:?usage: build_view.sh <agent> <iso dir> | build_view.sh native <key> <iso dir>}
if [ "$AGENT" = native ]; then
  KEY=${2:?native needs <key> <iso dir>}; ISO=${3:?native needs <key> <iso dir>}
else
  ISO=${2:?usage: build_view.sh <agent> <iso dir>}
fi
case "$AGENT" in
  native|openai_tool_loop|claude_code|claude_science|gemini_deep_research|elicit) ;;
  *) echo "FATAL: unknown agent $AGENT ($AGENTS native)"; exit 2;;
esac
[ -f "$ISO/out/E13/window_scores.parquet" ] || { echo "FATAL: $ISO has no out/E13/window_scores.parquet"; exit 2; }
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${LOCAL:-0}" != 1 ]; then
  LOGS=${SCRIBE_WORK_ROOT:?set SCRIBE_WORK_ROOT (configs/site.env.example)}/logs
  mkdir -p "$LOGS" || exit 2
  exec sbatch --parsable --job-name=board_view --gres=gpu:1 --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=32G --time=01:00:00 --export=ALL,LITREVIEW_ROOT=$LITREVIEW_ROOT \
    --output=$LOGS/board_view_%j.log "$0" "$@"
fi
source "${CCBENCH_ENV:?set CCBENCH_ENV to the site environment script}"
export PYTHONPATH=$LITREVIEW_ROOT/evaluation:$PYTHONPATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
unset WINDOWBENCH_OUTLINE_AXIS_ROWS WINDOWBENCH_ROSTER_INJECT CCBENCH_DEVICE
VB=$LITREVIEW_ROOT/evaluation/board/view_builder.py
echo "### build_view $* start=$(date) host=$(hostname) job=${SLURM_JOB_ID:-local}"
( cd "$LITREVIEW_ROOT/evaluation/board" && md5sum view_builder.py view_agents.py run_view.py ) | sed 's/^/### md5 /'
cd "$LITREVIEW_ROOT/evaluation" || exit 2
if [ "$AGENT" = native ]; then
  $PY -u "$VB" native --key "$KEY" --iso "$ISO"
else
  KEYS=$($PY -c "import sys; sys.path.insert(0, 'board'); from view_agents import AGENTS; print(' '.join(AGENTS['$AGENT']['keys']))") || exit 2
  ISOARGS=(); for k in $KEYS; do ISOARGS+=(--iso "$k=$ISO"); done
  $PY -u "$VB" view --agent "$AGENT" "${ISOARGS[@]}"
fi
rc=$?
echo "### end=$(date) rc=$rc"
exit $rc
