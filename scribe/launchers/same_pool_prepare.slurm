#!/bin/bash
# Slurm CPU job that copies the pinned acquisition units into a new same-pool generation run, audits
# the copies and writes the prepare records. Submitted by run_same_pool.sh.
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G
#SBATCH --time=00:45:00
set -uo pipefail
LITREVIEW_ROOT=${LITREVIEW_ROOT:?LITREVIEW_ROOT required (run_same_pool.sh passes it)}
[ -f "$LITREVIEW_ROOT/scribe/harness/runners/runner.py" ] || { echo "FATAL: LITREVIEW_ROOT=$LITREVIEW_ROOT is not the repo"; exit 2; }
for v in SCRIBE_RUN_PY SCRIBE_POOL_CUTOFFS SCRIBE_POOL_EDGES SCRIBE_KCAP; do
  [ -n "${!v:-}" ] || { echo "FATAL: $v is not set (configs/site.env.example)"; exit 2; }
done
RUN_PY=$SCRIBE_RUN_PY; L=$LITREVIEW_ROOT/scribe/launchers; RETRIEVAL=$LITREVIEW_ROOT/scribe/retrieval
RUN=${SCRIBE_RUN_DIR:?SCRIBE_RUN_DIR required}
echo "### start=$(date) node=$(hostname) job=${SLURM_JOB_ID:-} run=$RUN"
echo "### launcher: $0 md5 $(md5sum "$0" | awk '{print $1}')"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
cfg(){ $RUN_PY -c "import json,sys; c=json.load(open('$RUN/config.json')); v=eval(sys.argv[1], {'c': c}); print(v if not isinstance(v, (list, dict)) else json.dumps(v))" "$1"; }
LEVEL=$(cfg "c['level']"); ARM=$(cfg "c['arm']"); RUNS=$(cfg "c['runs_root']")
SYSTEM=$(cfg "c.get('system') or 'SCRIBE'"); SEED=$(cfg "c['seed']")
SRC_AUDIT_OK=$(cfg "c.get('src_audit_ok') or ''")
LVL_DIR=$RUNS/level$LEVEL
fail(){ echo "{\"schema\": \"level_marker/1.0\", \"kind\": \"acq_failed\", \"job\": \"${SLURM_JOB_ID:-}\", \"arm\": \"$ARM\", \"why\": \"same-pool prepare: $1\", \"run\": \"$RUN\", \"utc\": \"$(date -u +%FT%TZ)\"}" > $LVL_DIR/_level_acq_failed_${SLURM_JOB_ID:-nojob}.json
        echo "{\"ok\": false, \"level\": $LEVEL, \"why\": \"$1\", \"job\": \"${SLURM_JOB_ID:-}\"}" > $RUN/prepare_record.json
        echo "FATAL: $1 (level marked dead)"; exit 3; }
[ -d "$LVL_DIR" ] || { echo "FATAL: $LVL_DIR not claimed"; exit 2; }
md5sum -c --quiet $RUN/code.md5 >/dev/null 2>&1 || fail "code drift since submit"
cfg "c['tasks']" | $RUN_PY -c "import json,sys; print('\n'.join(json.load(sys.stdin)))" > $RUN/tasks_all.txt
$RUN_PY $L/same_pool_prepare.py materialize --pins "$(cfg "c['pins']")" --pins-sha256 "$(cfg "c['pins_sha256']")" \
  --tasks $RUN/tasks_all.txt --level $LEVEL --runs-root $RUNS --system $SYSTEM --out $RUN/provenance.json || fail "materialize refused"
$RUN_PY $L/same_pool_prepare.py verify --provenance $RUN/provenance.json --phase pre --out $RUN/verify_prepare.json || fail "verify after materialize refused"
TASKS=$(paste -sd, $RUN/tasks_all.txt)
SLOG_SRC=$(cfg "c['service_log']"); SLOG_SHA=$(cfg "c['service_log_sha256']")
cp -p "$SLOG_SRC" $RUN/search_calls.jsonl || fail "cannot copy the source service log"
[ "$(sha256sum < $RUN/search_calls.jsonl | cut -c1-64)" = "$SLOG_SHA" ] || fail "service log sha256 != the one recorded at submit"
KCAP_SHA256=$(sha256sum < $SCRIBE_KCAP | cut -c1-64)
cd $RUN
export POOL_EDGES=$SCRIBE_POOL_EDGES
$RUN_PY $RETRIEVAL/acquisition_audit.py --runs $RUNS --level $LEVEL --tasks "$TASKS" --cutoffs $SCRIBE_POOL_CUTOFFS \
  --service-log $RUN/search_calls.jsonl --seeds $SEED --system $SYSTEM --out $RUN/acquisition_audit.json | tail -2
[ "${PIPESTATUS[0]}" = 0 ] || fail "acquisition audit findings"
$RUN_PY $RETRIEVAL/ranking_budget_audit.py --runs $RUNS --level $LEVEL --tasks "$TASKS" --kcap $SCRIBE_KCAP \
  --kcap-sha256 $KCAP_SHA256 --seeds $SEED --system $SYSTEM --out $RUN/ranking_budget_audit.json | tail -2
[ "${PIPESTATUS[0]}" = 0 ] || fail "cap budget / ranking audit findings"
$RUN_PY $RETRIEVAL/selection_audit.py --glob "$RUNS/level$LEVEL/$SYSTEM/native_chain/*/seed*" --ranking-audit --require --min-units 1 \
  --module-md5 $(md5sum < $RETRIEVAL/selection_rule.py | cut -c1-32) --gold-root $RUNS/gold/dev --out $RUN/selection_audit.json | tail -2
[ "${PIPESTATUS[0]}" = 0 ] || fail "selection rule audit findings"
$RUN_PY - "$RUN" "$LVL_DIR/_level_acq_audit_ok.json" "$LEVEL" "$ARM" "$TASKS" "$SRC_AUDIT_OK" "$SEED" <<'PYEOF' || fail "cannot write the audit record"
import glob, hashlib, json, os, sys, time
run, dst, lvl, arm, tasks, src, seed = sys.argv[1:8]
aud = {}
for f in sorted(glob.glob(f"{run}/*audit*.json")):
    a = json.load(open(f)); aud[os.path.basename(f)] = {"n_findings": a.get("n_findings"), "findings": len(a.get("findings") or []),
                                                         "md5": hashlib.md5(open(f, "rb").read()).hexdigest()}
ok = len(aud) == 3 and all(v["n_findings"] in (0, None) and v["findings"] == 0 for v in aud.values())
a = json.load(open(f"{run}/acquisition_audit.json")); raw = open(f"{run}/acquisition_audit.json", "rb").read()
rec = {"schema": "acquisition_audit_ok/1.0", "ok": ok and a.get("n_findings") == 0, "level": int(lvl), "retrieval_budget": "cap",
       "arm": arm, "job": os.environ.get("SLURM_JOB_ID"), "audit": f"{run}/acquisition_audit.json", "audit_md5": hashlib.md5(raw).hexdigest(),
       "n_units_audited": a.get("n_units_audited"), "no_bundle": a.get("no_bundle"),
       "n_tasks": len([t for t in tasks.split(",") if t]), "seeds": seed, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "same_pool_generation": {"what": "acquisition not re-run: the units are byte-identical copies of the source level's (sha256-pinned); "
                                        "the three acquisition audits were re-run unchanged on these copies", "audits": aud,
                                "source_audit_ok_record": src, "source_audit_ok_md5": hashlib.md5(open(src, "rb").read()).hexdigest()}}
json.dump(rec, open(dst, "w"), indent=1); print(f"### audit record ok={rec['ok']} -> {dst}: {aud}")
sys.exit(0 if rec["ok"] else 1)
PYEOF
echo "{\"ok\": true, \"level\": $LEVEL, \"src\": \"level\", \"audits_ok\": \"ok\", \"job\": \"${SLURM_JOB_ID:-}\", \"utc\": \"$(date -u +%FT%TZ)\"}" > $RUN/prepare_record.json
echo "### prepare ok: $(wc -l < $RUN/tasks_all.txt) units; audits ok; end=$(date)"
