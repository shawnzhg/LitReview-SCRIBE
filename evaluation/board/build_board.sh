#!/bin/bash
# Runs build_board.py build as a CPU Slurm job, or in the current shell with LOCAL=1, and records
# the md5 of the board code. Usage: build_board.sh [--ref S,...] [--self S,...] [--extra
# NAME=VIEW,...] [--out <dir>].
set -uo pipefail
LITREVIEW_ROOT=${LITREVIEW_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
[ -f "$LITREVIEW_ROOT/evaluation/board/build_board.py" ] || { echo "FATAL: set LITREVIEW_ROOT to the repository root"; exit 2; }
export LITREVIEW_ROOT
WORK=${SCRIBE_WORK_ROOT:?set SCRIBE_WORK_ROOT (configs/site.env.example)}
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${LOCAL:-0}" != 1 ]; then
  mkdir -p "$WORK/logs" || exit 2
  exec sbatch --parsable --job-name=build_board --nodes=1 --ntasks=1 \
    --cpus-per-task=4 --mem=48G --time=02:00:00 --export=ALL,LITREVIEW_ROOT=$LITREVIEW_ROOT \
    --output=$WORK/logs/build_board_%j.log "$0" "$@"
fi
source "${CCBENCH_ENV:?set CCBENCH_ENV to the site script that defines PY and the CCBENCH_* roots}"
export PYTHONPATH=$LITREVIEW_ROOT/evaluation:$PYTHONPATH
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0 PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
unset WINDOWBENCH_OUTLINE_AXIS_ROWS WINDOWBENCH_ROSTER_INJECT CCBENCH_DEVICE
REC=$WORK/board/code/$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID:-local$$}
mkdir -p "$REC" || exit 2
( cd "$LITREVIEW_ROOT/evaluation" && md5sum board/*.py board/*.sh windowbench/*.py ) > "$REC/code.md5" || exit 2
echo "### build_board $* start=$(date) host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "### code md5 record $REC/code.md5"; sed 's/^/### md5 /' "$REC/code.md5" | grep -E 'board/build_board|board/view_builder|windowbench/merged_board'
echo "### PY=$PY OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS PYTHONHASHSEED=$PYTHONHASHSEED"
cd "$LITREVIEW_ROOT/evaluation" || exit 2
$PY -u board/build_board.py build "$@"
rc=$?
( cd "$LITREVIEW_ROOT/evaluation" && md5sum -c --quiet "$REC/code.md5" ) || { echo "FATAL: the board code changed during the build"; rc=88; }
echo "### end=$(date) rc=$rc"
exit $rc
