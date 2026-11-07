#!/bin/bash
# Runs SCRIBE-Luna, the SCRIBE harness on gpt-5.6-luna behind the recording proxy and egress guard.
# Usage: RUN_DIR=<dir> LEVEL=<n> TASKS_FILE=<file> LUNA_API_KEY_FILE=<file> LUNA_BUDGET_USD=<usd>
# [SEEDS=<s>] bash eval_luna.sh fixed|pool.
set -uo pipefail
COND=${1:-}
case "$COND" in fixed|pool) ;; *) echo "usage: eval_luna.sh fixed|pool" >&2; exit 2;; esac
LITREVIEW_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export LITREVIEW_ROOT
REPO=$LITREVIEW_ROOT
CONFIG=${LUNA_CONFIG:-$REPO/scribe/launchers/configs/scribe_luna.env}
set -a; source "$CONFIG"; set +a
die(){ echo "eval_luna: $1" >&2; exit "${2:-2}"; }
PY=${SCRIBE_RUN_PY:-}
for v in SCRIBE_RUN_PY RUN_DIR LEVEL TASKS_FILE LUNA_API_KEY_FILE LUNA_BUDGET_USD SCRIBE_LEVERS LUNA_MODEL LUNA_SERVICE_TIER LUNA_REASONING_EFFORT; do
  [ -n "${!v:-}" ] || die "$v is not set"
done
[ -f "$TASKS_FILE" ] || die "TASKS_FILE=$TASKS_FILE is not a file"
[ -f "$SCRIBE_LEVERS" ] || die "SCRIBE_LEVERS=$SCRIBE_LEVERS is not a file"
[ -f "$LUNA_API_KEY_FILE" ] || die "LUNA_API_KEY_FILE=$LUNA_API_KEY_FILE is not a file"
SYSTEM=${SYSTEM:-SCRIBE}; SEEDS=${SEEDS:-0}; TAG=${TAG:-scribe_luna}; WORKERS=${WORKERS:-8}
if [ "$COND" = pool ]; then
  case "$SEEDS" in 0|1|2) ;; *) die "SEEDS=$SEEDS: one seed per same-pool level, 0, 1 or 2";; esac
  for v in SCRIBE_POOL_PY SCRIBE_POOL_INDEX SCRIBE_POOL_EDGES SCRIBE_POOL_CUTOFFS POOL_GOLD_DIR SCRIBE_BUDGET_CAPS SCRIBE_KCAP SCRIBE_MODEL_DIR; do
    [ -n "${!v:-}" ] || die "$v is not set (same pool)"
  done
  [ -f "$SCRIBE_BUDGET_CAPS" ] && [ -f "$SCRIBE_KCAP" ] || die "SCRIBE_BUDGET_CAPS / SCRIBE_KCAP are not files"
  SCRIBE_BUDGET_CAPS_SHA256=$(sha256sum < "$SCRIBE_BUDGET_CAPS" | cut -c1-64); SCRIBE_KCAP_SHA256=$(sha256sum < "$SCRIBE_KCAP" | cut -c1-64)
  POOL_INDEX=$SCRIBE_POOL_INDEX POOL_EDGES=$SCRIBE_POOL_EDGES POOL_CUTOFFS=$SCRIBE_POOL_CUTOFFS
  POOL_PY=$SCRIBE_POOL_PY POOL_PYTHONPATH=${SCRIBE_POOL_PYTHONPATH:-}
  SCRIBE_RANKING_TOKENIZER=$SCRIBE_MODEL_DIR SCRIBE_RANKING_MAX_MODEL_LEN=65536
  export POOL_INDEX POOL_EDGES POOL_CUTOFFS POOL_GOLD_DIR POOL_PY POOL_PYTHONPATH SCRIBE_BUDGET_CAPS SCRIBE_BUDGET_CAPS_SHA256 SCRIBE_KCAP \
    SCRIBE_KCAP_SHA256 SCRIBE_RANKING_TOKENIZER SCRIBE_RANKING_MAX_MODEL_LEN
  unset POOL_ALLOWLIST_DIR
else
  [[ "$SEEDS" =~ ^[012](,[012])*$ ]] && [ "$(tr ',' '\n' <<< "$SEEDS" | sort | uniq -d)" = "" ] || die "SEEDS=$SEEDS must be distinct seeds from 0, 1, 2"
fi
BRIDGE=${EGRESS_LANE_BRIDGE:-$REPO/scribe/harness/runners/sock_bridge.py}
[ -f "$BRIDGE" ] || die "EGRESS_LANE_BRIDGE=$BRIDGE is not a file"
LUNA_API_BASE=${LUNA_API_BASE:-https://api.openai.com/v1}
LUNA_PRICE_IN=${LUNA_PRICE_IN:-0.10}; LUNA_PRICE_OUT=${LUNA_PRICE_OUT:-0.60}
PROXY_TIMEOUT=${PROXY_TIMEOUT:-1800}; CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-2400}
PROXY_PORT=${PROXY_PORT:-$(( 18000 + RANDOM % 1000 ))}; APP_PORT=${APP_PORT:-$(( 19000 + RANDOM % 1000 ))}
POOL_PORT=${POOL_PORT:-8931}
RUNNERS=$REPO/scribe/harness/runners
export PATH="$(dirname "$PY"):$PATH" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
RUNS=$("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import common; print(common.RUNS)' "$RUNNERS") || die "cannot resolve the runs root"
if [ "$COND" = pool ]; then
  read -r PIN_STATS_FP PIN_N_DOCS POOL_SNAPSHOT_ID < <("$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import pool_retrieval_tool as P; print(P.PIN_STATS_FINGERPRINT, P.PIN_N_DOCS, P.POOL_SNAPSHOT_ID)' "$REPO/scribe/harness/tools")
  [ -n "${POOL_SNAPSHOT_ID:-}" ] || die "cannot read the frozen pool constants from scribe/harness/tools/pool_retrieval_tool.py"
  export PIN_STATS_FP PIN_N_DOCS
fi
TASKS=$(grep -v '^[[:space:]]*#' "$TASKS_FILE" | tr -s '[:space:]' ',' | sed 's/^,//; s/,$//')
mkdir -p "$RUN_DIR"
if [ "$COND" = pool ]; then
  SVC=$REPO/biolitbench/pool/pool_service.py
  "$PY" "$REPO/scribe/retrieval/pool_parity.py" static --service "$SVC" --out "$RUN_DIR/pool_parity.json" \
    || die "the pool config is not the baselines' pool-service configuration (see $RUN_DIR/pool_parity.json)" 81
  bash "$RUNNERS/egress_guard.sh" "$RUN_DIR/parity_probe_egress_proof.json" bash "$REPO/scribe/retrieval/pool_parity_probe.sh" \
    "$RUN_DIR/parity_probe" "$POOL_PORT" "$SVC" "$POOL_INDEX" "$POOL_CUTOFFS" || die "pool parity probe refused (see $RUN_DIR/parity_probe)" 83
fi
PIDS=()
cleanup(){ for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; }
trap cleanup EXIT
wait_http(){ for _i in $(seq 1 "$2"); do curl -sf -o /dev/null "$1" 2>/dev/null && return 0; sleep 2; done; return 1; }

"$PY" -u "$RUNNERS/llm_proxy.py" --listen "$PROXY_PORT" --upstream-url "$LUNA_API_BASE" \
  --api-key-file "$LUNA_API_KEY_FILE" --force-model "$LUNA_MODEL" --service-tier "$LUNA_SERVICE_TIER" \
  --reasoning-effort "$LUNA_REASONING_EFFORT" --price-in "$LUNA_PRICE_IN" --price-out "$LUNA_PRICE_OUT" \
  --cost-file "$RUN_DIR/costs.jsonl" --budget-usd "$LUNA_BUDGET_USD" --arm "$TAG" --timeout "$PROXY_TIMEOUT" \
  --log "$RUN_DIR/_calls.jsonl" > "$RUN_DIR/proxy.log" 2>&1 &
PIDS+=($!)
wait_http "http://127.0.0.1:$PROXY_PORT/__proxy_health" 30 || die "the recording proxy did not start (see $RUN_DIR/proxy.log)" 94

LANE_SOCK=$RUN_DIR/lane.sock; rm -f "$LANE_SOCK"
"$PY" -u "$BRIDGE" --unix-listen "$LANE_SOCK" --tcp-connect "127.0.0.1:$PROXY_PORT" > "$RUN_DIR/lane.log" 2>&1 &
PIDS+=($!)
for _i in $(seq 1 30); do [ -S "$LANE_SOCK" ] && break; sleep 1; done
[ -S "$LANE_SOCK" ] || die "the lane socket $LANE_SOCK did not appear" 94
DECLARED=$APP_PORT; [ "$COND" = pool ] && DECLARED="$APP_PORT,$POOL_PORT"
export EGRESS_LANE_SOCK=$LANE_SOCK EGRESS_LANE_PORT=$APP_PORT EGRESS_LANE_BRIDGE=$BRIDGE \
  EGRESS_LANE_EXPECT_MODEL=$LUNA_MODEL EGRESS_LANE_EXPECT_TIER=$LUNA_SERVICE_TIER EGRESS_DECLARED_PORTS=$DECLARED
LLM_PORT=$APP_PORT
SCRIBE_RENDEZVOUS=$("$PY" -c 'import json, sys
p, m, t, e, c = sys.argv[1:6]
print(json.dumps({"backend": "openai-api", "url": f"http://127.0.0.1:{p}/v1", "host": "127.0.0.1", "port": int(p),
                  "served_model": m, "service_tier": t, "reasoning_effort": e, "model_path": None,
                  "max_model_len": None, "client_timeout_s": float(c)}))' \
  "$LLM_PORT" "$LUNA_MODEL" "$LUNA_SERVICE_TIER" "$LUNA_REASONING_EFFORT" "$CLIENT_TIMEOUT") || die "rendezvous"
printf '%s\n' "$SCRIBE_RENDEZVOUS" > "$RUN_DIR/rendezvous.json"
export PY SCRIBE_RENDEZVOUS REPO RUNS RUN_DIR LEVEL TASKS TASKS_FILE SYSTEM SEEDS TAG WORKERS POOL_PORT SCRIBE_LEVERS COND
unset SCRIBE_RENDEZVOUS_WRITING SCRIBE_RENDEZVOUS_PLANNING

cat > "$RUN_DIR/inner.sh" <<'INNER'
#!/bin/bash
set -uo pipefail
cd "$REPO"
wait_http(){ for _i in $(seq 1 "$2"); do curl -sf -o /dev/null "$1" 2>/dev/null && return 0; sleep 2; done; return 1; }
LUNA="$PY $REPO/scribe/luna/run_luna.py"
if [ "$COND" = fixed ]; then
  SCRIBE_DRIVER_RECORD=$RUN_DIR/driver.json LUNA_MODULES_RECORD=$RUN_DIR/modules.json \
    $LUNA fixed --phase generation --systems "$SYSTEM" --mode bundle_entry --level "$LEVEL" --seeds "$SEEDS" \
    --tasks-file "$TASKS_FILE" --workers "$WORKERS" --campaign "${TAG}_gen"
  exit $?
fi
export POOL_LOG=$RUN_DIR/search_calls.jsonl
env PYTHONPATH="$POOL_PYTHONPATH" PYTHONNOUSERSITE=1 "$POOL_PY" "$REPO/biolitbench/pool/pool_service.py" --host 127.0.0.1 \
  --port "$POOL_PORT" --index "$POOL_INDEX" --cutoffs "$POOL_CUTOFFS" > "$RUN_DIR/pool_service.log" 2>&1 &
POOL_PID=$!
wait_http "http://127.0.0.1:$POOL_PORT/healthz" 300 || { kill $POOL_PID; echo "pool service did not start"; exit 83; }
curl -sf "http://127.0.0.1:$POOL_PORT/healthz" > "$RUN_DIR/pool_healthz.json"
"$PY" -c 'import json, sys; h = json.load(open(sys.argv[1])); sys.exit(0 if h.get("stats_fingerprint") == sys.argv[2] and int(h.get("n_docs", -1)) == int(sys.argv[3]) else 1)' \
  "$RUN_DIR/pool_healthz.json" "$PIN_STATS_FP" "$PIN_N_DOCS" || { kill $POOL_PID; echo "pool /healthz does not match the frozen pool"; exit 83; }
"$PY" "$REPO/scribe/retrieval/pool_parity.py" health "$RUN_DIR/pool_healthz.json" --label acquisition-pool \
  || { kill $POOL_PID; echo "the acquisition pool is not the baselines' pool-service configuration"; exit 83; }
PYTHONPATH="$REPO/biolitbench/pool${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$REPO/scribe/retrieval/pool_boundary.py" \
  --index "$POOL_INDEX" --years 2010-2027 --out "$RUN_DIR/pool_boundary.json" || { kill $POOL_PID; exit 83; }
export POOL_URL=http://127.0.0.1:$POOL_PORT POOL_BOUNDARY=$RUN_DIR/pool_boundary.json
LUNA_MODULES_RECORD=$RUN_DIR/modules_acq.json $LUNA acquisition --phase acquisition --systems "$SYSTEM" \
  --mode native_chain --level "$LEVEL" --seeds "$SEEDS" --tasks-file "$TASKS_FILE" \
  --workers "$WORKERS" --campaign "${TAG}_acq"
RC_ACQ=$?
kill $POOL_PID 2>/dev/null; wait $POOL_PID 2>/dev/null
curl -sf -o /dev/null "http://127.0.0.1:$POOL_PORT/healthz" 2>/dev/null && { echo "the pool service still answers"; exit 92; }
[ "$RC_ACQ" = 0 ] || { echo "acquisition failed (rc $RC_ACQ)"; exit 92; }
"$PY" "$REPO/scribe/retrieval/pool_parity.py" routes "$RUN_DIR/search_calls.jsonl" --min-rows 1 \
  --out "$RUN_DIR/pool_routes.json" --label acquisition-pool || { echo "route guard refused"; exit 87; }
"$PY" "$REPO/scribe/retrieval/acquisition_audit.py" --runs "$RUNS" --level "$LEVEL" --tasks "$TASKS" \
  --cutoffs "$POOL_CUTOFFS" --service-log "$RUN_DIR/search_calls.jsonl" --seeds "$SEEDS" \
  --system "$SYSTEM" --out "$RUN_DIR/acquisition_audit.json" || { echo "acquisition audit findings"; exit 87; }
"$PY" "$REPO/scribe/retrieval/ranking_budget_audit.py" --runs "$RUNS" --level "$LEVEL" --tasks "$TASKS" \
  --kcap "$SCRIBE_KCAP" --kcap-sha256 "$SCRIBE_KCAP_SHA256" --seeds "$SEEDS" --system "$SYSTEM" \
  --out "$RUN_DIR/ranking_budget_audit.json" || { echo "budget and ranking audit findings"; exit 87; }
"$PY" "$REPO/scribe/retrieval/selection_audit.py" --glob "$RUNS/level$LEVEL/$SYSTEM/native_chain/*/seed*" --ranking-audit \
  --require --min-units 1 --module-md5 "$(md5sum < "$REPO/scribe/retrieval/selection_rule.py" | cut -c1-32)" \
  --gold-root "$RUNS/gold/dev" --out "$RUN_DIR/selection_audit.json" || { echo "selection-rule audit findings"; exit 87; }
"$PY" - "$RUN_DIR" "$RUNS/level$LEVEL/_level_acq_audit_ok.json" "$LEVEL" "$TAG" "$TASKS" "$SEEDS" <<'PYEOF' || { echo "acquisition audit record not ok"; exit 87; }
import glob, hashlib, json, os, sys, time
run, dst, lvl, arm, tasks, seeds = sys.argv[1:7]
aud = {}
for f in sorted(glob.glob(f"{run}/*audit*.json")):
    a = json.load(open(f))
    aud[os.path.basename(f)] = {"n_findings": a.get("n_findings"), "findings": len(a.get("findings") or []),
                                "md5": hashlib.md5(open(f, "rb").read()).hexdigest()}
src = f"{run}/acquisition_audit.json"
a = json.load(open(src))
ok = len(aud) == 3 and all(v["n_findings"] in (0, None) and v["findings"] == 0 for v in aud.values()) and a.get("n_findings") == 0
rec = {"schema": "acquisition_audit_ok/1.0", "ok": ok, "level": int(lvl), "retrieval_budget": "cap", "arm": arm,
       "job": os.environ.get("SLURM_JOB_ID"), "audit": src, "audit_md5": hashlib.md5(open(src, "rb").read()).hexdigest(),
       "n_units_audited": a.get("n_units_audited"), "no_bundle": a.get("no_bundle"),
       "n_tasks": len([t for t in tasks.split(",") if t]), "seeds": seeds, "audits": aud,
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(rec, open(dst, "w"), indent=1)
print(f"acquisition audit record ok={ok} -> {dst}")
sys.exit(0 if ok else 1)
PYEOF
SCRIBE_DRIVER_RECORD=$RUN_DIR/driver.json LUNA_MODULES_RECORD=$RUN_DIR/modules_gen.json \
  $LUNA generation --phase generation --systems "$SYSTEM" --mode native_chain --level "$LEVEL" --seeds "$SEEDS" \
  --tasks-file "$TASKS_FILE" --workers "$WORKERS" --campaign "${TAG}_gen"
INNER
chmod +x "$RUN_DIR/inner.sh"
bash "$RUNNERS/egress_guard.sh" "$RUN_DIR/egress_proof.json" bash "$RUN_DIR/inner.sh"; RC=$?
if [ "$COND" = pool ]; then
  "$PY" - "$RUN_DIR" "$RUNS" "$LEVEL" "$TAG" "$SYSTEM" "$SEEDS" "$TASKS" "$RC" "$POOL_SNAPSHOT_ID" \
    "$SCRIBE_BUDGET_CAPS" "$SCRIBE_BUDGET_CAPS_SHA256" "$SCRIBE_KCAP" "$SCRIBE_KCAP_SHA256" <<'PYEOF' || { echo "eval_luna: cannot write the level record" >&2; [ "$RC" = 0 ] && RC=92; }
import glob, json, os, sys, time
run, R, level, arm, system, seeds, tasks, rc, snap, caps, caps_sha, kcap, kcap_sha = sys.argv[1:14]
lvl = f"{R}/level{level}"
units = sorted(glob.glob(f"{lvl}/{system}/native_chain/*/seed*"))
def lj(p):
    try:
        return json.load(open(p))
    except Exception:
        return None
def cnt(n):
    return sum(os.path.exists(f"{u}/{n}") for u in units)
okrec = lj(f"{lvl}/_level_acq_audit_ok.json") or {}
rec = {"schema": "level_record/1.0", "state": "final", "rc": int(rc), "level": int(level), "tag": arm, "arm": arm,
       "seeds": seeds, "job": os.environ.get("SLURM_JOB_ID"), "mode": "native_chain", "corpus_snapshot_id": snap,
       "retrieval_budget": "cap", "budget_caps": caps, "budget_caps_sha256": caps_sha, "cap_rule": "K_cap",
       "kcap": kcap, "kcap_sha256": kcap_sha,
       "n_tasks_expected": len([t for t in tasks.split(",") if t]) * len([s for s in seeds.split(",") if s]),
       "acq_audit_ok": okrec.get("ok") is True and okrec.get("level") == int(level),
       "failure_markers": sorted(glob.glob(f"{lvl}/_level_*failed*.json")),
       "infra_markers": sorted(glob.glob(f"{lvl}/_level_infra_failure_*.json")),
       "n_units": len(units), "n_bundles": cnt("evidence_bundle.json"), "n_reports": cnt("report_artifact.json"),
       "n_pool_provenance_ok": sum(bool((lj(f"{u}/pool_provenance.json") or {}).get("ok")) for u in units),
       "service_log": f"{run}/search_calls.jsonl",
       "egress_proof": f"{run}/egress_proof.json" if os.path.exists(f"{run}/egress_proof.json") else None,
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(rec, open(f"{run}/level_record.json", "w"), indent=1)
print(f"eval_luna: level record rc={rc}: units {rec['n_units']} bundles {rec['n_bundles']} reports {rec['n_reports']} of "
      f"{rec['n_tasks_expected']}; acq_audit_ok={rec['acq_audit_ok']}")
PYEOF
fi
echo "eval_luna: $COND level $LEVEL finished rc=$RC; calls $RUN_DIR/_calls.jsonl, costs $RUN_DIR/costs.jsonl"
exit $RC
