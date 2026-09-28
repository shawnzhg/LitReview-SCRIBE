#!/bin/bash
# Runs DR-Tulu over a slice of the evaluation tasks under same pool inside the network guard. Usage:
# SHARD=<n> SLICE=<lo-hi> sbatch --export=ALL drtulu_pool.slurm.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=96G --gres=gpu:1
#SBATCH --time=48:00:00
set -uo pipefail
for v in LITREVIEW_ROOT RUNS_ROOT DRTULU_MODEL_DIR DRTULU_AGENT_DIR PATCH_DRTULU HF_HOME SCRIBE_RUN_PY; do [ -n "${!v:-}" ] || { echo "FATAL: $v is not set"; exit 95; }; done
B="$LITREVIEW_ROOT/scribe/harness/baselines"
SHARD="${SHARD:?SHARD not set (the shard number)}"
SLICE="${SLICE:?SLICE not set, e.g. 0-33}"
LIMIT="${LIMIT:-}"
CAMP_ROOT=${CAMP_ROOT:-$RUNS_ROOT/campaign50_native/drtulu}
CAMP=$CAMP_ROOT
mkdir -p "$CAMP" "$CAMP/_logs"

PIN_COMMIT=9d7b0371c085e9311ddec483ed39768c0bd9fe99
PIN_DIFF=2b8ab863c2db50f60bfc0bf56668afb97170c494417452286f7bbb7f90fad50f
HEADSHA="$(git -C "$DRTULU_AGENT_DIR" rev-parse HEAD 2>/dev/null || echo NONE)"
DIFFSHA="$(git -C "$DRTULU_AGENT_DIR" diff 2>/dev/null | sha256sum | cut -d' ' -f1)"
ARTSHA="$( [ -f "$PATCH_DRTULU" ] && sha256sum "$PATCH_DRTULU" | cut -d' ' -f1 || echo MISSING)"
MATCH=false
[ "$HEADSHA" = "$PIN_COMMIT" ] && [ "$DIFFSHA" = "$PIN_DIFF" ] && [ "$ARTSHA" = "$PIN_DIFF" ] && MATCH=true
printf '{"schema": "campaign_integrity/1.1", "arm": "drtulu", "clone": "%s", "head": "%s", "pinned_commit": "%s", "worktree_diff_sha256": "%s", "patch": "%s", "patch_sha256": "%s", "pinned_patch_sha256": "%s", "match": %s}\n' \
  "$DRTULU_AGENT_DIR" "$HEADSHA" "$PIN_COMMIT" "$DIFFSHA" "$PATCH_DRTULU" "$ARTSHA" "$PIN_DIFF" "$MATCH" > "$CAMP/integrity_shard${SHARD}.json"
if [ "$MATCH" != true ]; then
  echo "FATAL(91): $DRTULU_AGENT_DIR does not match its pinned code."
  echo "  HEAD            = $HEADSHA (pinned $PIN_COMMIT)"
  echo "  git diff sha256 = $DIFFSHA (pinned $PIN_DIFF)"
  echo "  patch sha256    = $ARTSHA ($PATCH_DRTULU)"
  exit 91
fi
echo "### integrity OK: dr-tulu @ ${HEADSHA:0:12}, diff sha256 ${DIFFSHA:0:12}"

[ -n "${SITE_ENV:-}" ] && { source "$SITE_ENV" || exit $?; }
source "$B/drtulu_env.sh" || exit $?
drtulu_assert             || exit $?

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

VPORT=18200
PROXYPORT=30001
POOLPORT=8931

echo "### shard=$SHARD slice=$SLICE node=$(hostname) start=$(date)"

export VPORT PROXYPORT POOLPORT SHARD SLICE LIMIT CAMP_ROOT
"$LITREVIEW_ROOT/scribe/harness/runners/egress_guard.sh" "$CAMP/egress_proof_shard${SHARD}.json" \
  bash "$B/drtulu_pool_inner.sh"
GRC=$?
echo "### guard rc=$GRC"
echo "### reports: $(ls "$CAMP"/*/report.json 2>/dev/null | wc -l)"
echo "### end=$(date)"
