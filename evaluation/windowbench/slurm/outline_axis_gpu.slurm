#!/bin/bash
# Slurm GPU job that builds the planning axis with the embedding title match for every roster system
# and its peer band into $WINDOWBENCH_OUT/outline_axis. Usage: sbatch outline_axis_gpu.slurm.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1
#SBATCH --time=02:00:00
LITREVIEW_ROOT=${LITREVIEW_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
[ -d "$LITREVIEW_ROOT/evaluation/windowbench" ] || { echo "FATAL: set LITREVIEW_ROOT to the repository root"; exit 2; }
source "${CCBENCH_ENV:?set CCBENCH_ENV to the site environment script}"
OUT=${WINDOWBENCH_OUT:?set WINDOWBENCH_OUT (configs/site.env.example)}
export PYTHONPATH=$LITREVIEW_ROOT/evaluation:${PYTHONPATH:-} CCBENCH_DEVICE=cuda PYTHONWARNINGS=ignore PYTHONDONTWRITEBYTECODE=1
cd $LITREVIEW_ROOT/evaluation
echo "### start=$(date) node=$(hostname)"
$PY -m windowbench.outline_axis --out $OUT/outline_axis 2>&1 | grep -vE "Warning|warn" | tail -60
echo "### end=$(date)"
