#!/bin/bash
# Slurm job that scores our fixed-input runs (<TAG>.bundle_entry[.seed<n>] keys) into the shared tables:
# rollouts, unit scores and window scores with matched length. Usage: SYSTEMS=<keys> sbatch
# score_new_arms.slurm.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --gres=gpu:1
#SBATCH --time=08:00:00
set -uo pipefail
LITREVIEW_ROOT=${LITREVIEW_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
export LITREVIEW_ROOT

SYSTEMS=${SYSTEMS:?SYSTEMS required (comma list of <TAG>.bundle_entry or <TAG>.bundle_entry.seed<n> keys)}
for k in ${SYSTEMS//,/ }; do
  [[ "$k" =~ ^[A-Za-z0-9_]+\.bundle_entry(\.seed[0-9]+)?$ ]] || { echo "FATAL: '$k' is not a <TAG>.bundle_entry[.seed<n>] key"; exit 2; }
done

WANT_OUT=${CCBENCH_OUT:-}
source "${CCBENCH_ENV:?set CCBENCH_ENV to the site environment script}"
[ -n "$WANT_OUT" ] && export CCBENCH_OUT="$WANT_OUT"
[ -d "$LITREVIEW_ROOT/evaluation/ccbench" ] || { echo "FATAL: LITREVIEW_ROOT=$LITREVIEW_ROOT has no evaluation/ccbench; export LITREVIEW_ROOT"; exit 2; }
export PYTHONPATH=$LITREVIEW_ROOT/evaluation:$PYTHONPATH
DEVICE=${CCBENCH_DEVICE:-cuda}
export CCBENCH_PANEL_B_EXTRA="$SYSTEMS" CCBENCH_DEVICE=$DEVICE
NCPU=${SLURM_CPUS_PER_TASK:-4}
export OMP_NUM_THREADS=$NCPU MKL_NUM_THREADS=$NCPU PYTHONWARNINGS=ignore
NC=""; [ "${NOCACHE:-1}" = 1 ] && NC="--no-cache"
T0=$(date +%s)
echo "### start=$(date) node=$(hostname) device=$DEVICE gpu=${CUDA_VISIBLE_DEVICES:-none} systems=$SYSTEMS out=$CCBENCH_OUT"
if [ "$DEVICE" = cuda ]; then
  { [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && nvidia-smi -L >/dev/null 2>&1; } \
    || { echo "FATAL: CCBENCH_DEVICE=cuda but no GPU is visible (submit with --gres, or use CCBENCH_DEVICE=cpu)"; exit 2; }
fi
cd "$LITREVIEW_ROOT/evaluation"
exec 9>"$CCBENCH_OUT/E13/.certify.lock" || { echo "FATAL: cannot open the E13 lock"; exit 2; }
flock -w "${LOCK_WAIT:-86400}" 9 || { echo "FATAL: timed out waiting for the E13 lock"; exit 2; }
step() {
  local label=$1; shift
  echo "[$(date -u +%H:%M:%S)] === $label: $*"
  "$@" 2>&1 | tail -${TAIL:-25}
  local rc=${PIPESTATUS[0]}
  echo "[$(date -u +%H:%M:%S)] === $label rc=$rc ($(( $(date +%s) - T0 )) s since start)"
  return $rc
}
TAIL=12 step "build" $PY -u -m ccbench.build --systems "$SYSTEMS" --redo || exit 3
TAIL=15 step "score_units" $PY -u -m ccbench.score_units --systems "$SYSTEMS" $NC || exit 3
TAIL=30 step "score_windows" $PY -u -m ccbench.score_windows --systems "$SYSTEMS" $NC --truncate || exit 3
echo "### end=$(date) rc=0 ($(( $(date +%s) - T0 )) s)"
exit 0
