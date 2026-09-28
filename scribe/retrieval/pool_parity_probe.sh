#!/bin/bash
# Starts one pool service, checks its /healthz with pool_parity.py health and stops it again. Usage:
# pool_parity_probe.sh <outdir> <port> <service.py> <index dir> <cutoffs.json> [n_tasks].
set -uo pipefail
[ $# -ge 5 ] || { echo "usage: $0 <outdir> <port> <service.py> <index dir> <cutoffs.json> [n_tasks]"; exit 2; }
OUT=$1; PORT=$2; SVC=$3; IDX=$4; CUT=$5; NT=${6:-}
POOL_PY=${SCRIBE_POOL_PY:?set SCRIBE_POOL_PY (configs/site.env.example)}; POOL_PYTHONPATH=${SCRIBE_POOL_PYTHONPATH:-}
CHECK_PY=${SCRIBE_RUN_PY:?set SCRIBE_RUN_PY (configs/site.env.example)}; : "${POOL_GOLD_DIR:?set POOL_GOLD_DIR (configs/site.env.example)}"; CHECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pool_parity.py"
mkdir -p "$OUT" || exit 83
echo "### parity probe: service $SVC port $PORT env POOL_EDGES=${POOL_EDGES:-<unset>}"
T0=$(date +%s)
POOL_LOG="$OUT/probe_calls.jsonl" setsid env PYTHONPATH="$POOL_PYTHONPATH" PYTHONNOUSERSITE=1 $POOL_PY "$SVC" --host 127.0.0.1 --port $PORT \
  --index "$IDX" --cutoffs "$CUT" > "$OUT/probe_service.log" 2>&1 &
PID=$!
stop(){ kill -- -$PID 2>/dev/null || kill $PID 2>/dev/null
  for i in 1 2 3 4 5 6; do curl -sf -o /dev/null "http://127.0.0.1:$PORT/healthz" 2>/dev/null || break; sleep 2; done
  if curl -sf -o /dev/null "http://127.0.0.1:$PORT/healthz" 2>/dev/null; then
    kill -KILL -- -$PID 2>/dev/null || kill -KILL $PID 2>/dev/null; sleep 3
    curl -sf -o /dev/null "http://127.0.0.1:$PORT/healthz" 2>/dev/null && { echo "FATAL: the probe pool is still answering"; return 1; }
  fi
  wait $PID 2>/dev/null; return 0; }
UP=0
for i in $(seq 1 60); do
  curl -sf -o /dev/null "http://127.0.0.1:$PORT/healthz" 2>/dev/null && { UP=1; break; }
  kill -0 $PID 2>/dev/null || break
  sleep 5
done
if [ "$UP" != 1 ]; then
  echo "FATAL: the parity probe pool never came up ($(( $(date +%s) - T0 ))s); service log tail:"; tail -5 "$OUT/probe_service.log" | cut -c1-240
  stop; exit 83
fi
curl -sf "http://127.0.0.1:$PORT/healthz" > "$OUT/probe_healthz.json"
if [ -n "$NT" ]; then $CHECK_PY $CHECK health "$OUT/probe_healthz.json" --n-tasks "$NT" --label "probe:$PORT"; RC=$?
else $CHECK_PY $CHECK health "$OUT/probe_healthz.json" --label "probe:$PORT"; RC=$?; fi
echo "### parity probe: up after $(( $(date +%s) - T0 ))s; health rc=$RC"
stop || exit 83
echo "### parity probe: service stopped (verified: /healthz no longer answers)"
[ "$RC" = 0 ] || exit 83
exit 0
