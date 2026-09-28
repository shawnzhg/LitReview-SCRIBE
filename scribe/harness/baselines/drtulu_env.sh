# Defines and checks the three interpreters of the DR-Tulu run (vLLM serving, pool service, agent)
# from environment variables.
export SERVE_PY="${SERVE_PY:?SERVE_PY not set (vLLM interpreter)}"
export POOL_PY="${POOL_PY:?POOL_PY not set (pool service interpreter)}"
export POOL_PYTHONPATH="${POOL_PYTHONPATH:-}"
export DRTULU_PY="${DRTULU_PY:?DRTULU_PY not set (DR-Tulu agent interpreter)}"
export DRTULU_LD="${DRTULU_LD:-}"
export VLLM_EXPECT="${VLLM_EXPECT:-0.23.1rc1.dev528+g9036c89ee}"
export PYTHONNOUSERSITE=1

drtulu_py() { env LD_LIBRARY_PATH="$DRTULU_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                  PYTHONNOUSERSITE=1 "$DRTULU_PY" "$@"; }
pool_py()   { env PYTHONPATH="$POOL_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
                  PYTHONNOUSERSITE=1 "$POOL_PY" "$@"; }

drtulu_assert() {
  local rc=0
  drtulu_py -c "
import sys; v=sys.version_info
assert (v.major,v.minor)==(3,10), f'app env wrong: {v.major}.{v.minor}, lockfile needs 3.10'
import diskcache, fastmcp, transformers
print(f'  app  OK  py{v.major}.{v.minor}.{v.micro} diskcache {diskcache.__version__} transformers {transformers.__version__}')
" || rc=90
  pool_py -c "
import sys, bm25s; v=sys.version_info
assert (v.major,v.minor)==(3,12), f'pool env wrong: {v.major}.{v.minor}, it must be python 3.12'
print(f'  pool OK  py{v.major}.{v.minor}.{v.micro} bm25s {bm25s.__version__}')
" || rc=91
  "$SERVE_PY" -c "
import vllm
import os
want=os.environ['VLLM_EXPECT']
assert vllm.__version__==want, f'SERVING ENV MISMATCH: {vllm.__version__} != {want}'
print(f'  serve OK vllm {vllm.__version__}')
" || rc=92
  [ $rc -ne 0 ] && echo "FATAL: env assertion failed (rc=$rc)"
  return $rc
}
