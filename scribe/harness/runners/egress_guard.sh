#!/bin/bash
# Runs a command in a loopback-only network namespace and writes a JSON proof that external routes
# are blocked; an optional declared unix-socket lane reaches the recording proxy. Usage:
# egress_guard.sh <proof.json> <command...>.
set -u
PROOF="${1:?usage: egress_guard.sh <proof.json> <command...>}"; shift
[ "$#" -gt 0 ] || { echo "egress_guard: no command given" >&2; exit 2; }

command -v unshare >/dev/null 2>&1 || { echo "egress_guard: unshare not found" >&2; exit 3; }
export EG_PY="${SCRIBE_RUN_PY:-}"
[ -x "$EG_PY" ] || { echo "egress_guard: SCRIBE_RUN_PY is not an executable interpreter (configs/site.env.example)" >&2; exit 3; }

mkdir -p "$(dirname "$PROOF")"

export EGRESS_PROOF="$PROOF"
export EG_HOST="$(hostname)"
export EG_CMD="$*"
export EG_LANE_SOCK="${EGRESS_LANE_SOCK:-}"
export EG_LANE_PORT="${EGRESS_LANE_PORT:-}"
export EG_LANE_BRIDGE="${EGRESS_LANE_BRIDGE:-}"
export EG_LANE_MODEL="${EGRESS_LANE_EXPECT_MODEL:-}"
export EG_LANE_TIER="${EGRESS_LANE_EXPECT_TIER:-}"
export EG_DECLARED_PORTS="${EGRESS_DECLARED_PORTS:-}"

export EG_RESOLVED="$("$EG_PY" - <<'RESOLVE'
import json, socket
hosts = ["eutils.ncbi.nlm.nih.gov", "doi.org", "api.semanticscholar.org", "api.openai.com"]
out = {}
for h in hosts:
    try:
        out[h] = socket.gethostbyname(h)
    except Exception as e:
        out[h] = None
print(json.dumps(out))
RESOLVE
)"

exec unshare -rn bash -c '
  EG_LO_ERR=""
  if command -v ip >/dev/null 2>&1; then
    EG_LO_ERR="$(ip link set lo up 2>&1)" || EG_LO_ERR="ip link set lo up failed: $EG_LO_ERR"
  elif command -v ifconfig >/dev/null 2>&1; then
    EG_LO_ERR="$(ifconfig lo up 2>&1)" || EG_LO_ERR="ifconfig lo up failed: $EG_LO_ERR"
  else
    EG_LO_ERR="neither ip nor ifconfig available; cannot bring lo up"
  fi
  export EG_LO_ERR

  EG_BRIDGE_PID=""
  if [ -n "$EG_LANE_SOCK" ]; then
    if [ ! -S "$EG_LANE_SOCK" ]; then
      echo "egress_guard: REFUSING -- declared lane socket $EG_LANE_SOCK is not a socket" >&2
      exit 7
    fi
    "$EG_PY" "$EG_LANE_BRIDGE" --tcp-listen "127.0.0.1:$EG_LANE_PORT" \
            --unix-connect "$EG_LANE_SOCK" >&2 &
    EG_BRIDGE_PID=$!
    EG_LANE_UP=0
    for _i in $(seq 1 60); do
      if "$EG_PY" -c "import socket,sys
s=socket.socket(); s.settimeout(1)
sys.exit(0 if s.connect_ex((chr(49)+chr(50)+chr(55)+chr(46)+chr(48)+chr(46)+chr(48)+chr(46)+chr(49), $EG_LANE_PORT))==0 else 1)"; then
        EG_LANE_UP=1; break
      fi
      kill -0 $EG_BRIDGE_PID 2>/dev/null || { echo "egress_guard: lane bridge died" >&2; exit 7; }
      sleep 0.5
    done
    [ "$EG_LANE_UP" = 1 ] || { echo "egress_guard: lane bridge never listened" >&2; exit 7; }
  fi
  export EG_BRIDGE_PID

  "$EG_PY" - <<PROBE
import json, os, socket, sys, time

ENETUNREACH, EHOSTUNREACH = 101, 113

def probe(target, port):
    t0 = time.time()
    try:
        socket.create_connection((target, port), 4).close()
        return {"result": "OPEN", "class": "open", "detail": None,
                "ms": round((time.time() - t0) * 1000)}
    except socket.gaierror as e:
        return {"result": "BLOCKED", "class": "dns_failure",
                "detail": "gaierror: " + str(e)[:100], "ms": round((time.time() - t0) * 1000)}
    except OSError as e:
        cls = "no_route" if getattr(e, "errno", None) in (ENETUNREACH, EHOSTUNREACH) else "other"
        return {"result": "BLOCKED", "class": cls,
                "detail": type(e).__name__ + ": " + str(e)[:100],
                "ms": round((time.time() - t0) * 1000)}
    except Exception as e:
        return {"result": "BLOCKED", "class": "other",
                "detail": type(e).__name__ + ": " + str(e)[:100],
                "ms": round((time.time() - t0) * 1000)}

resolved = json.loads(os.environ.get("EG_RESOLVED") or "{}")

ext = {}
for h in resolved:
    ext["name:" + h] = probe(h, 443)
for h, ip in resolved.items():
    if ip:
        ext["addr:" + h + "(" + ip + ")"] = probe(ip, 443)
ext["addr:1.1.1.1"] = probe("1.1.1.1", 443)

lo = probe("127.0.0.1", 1)

route_proofs = [k for k, v in ext.items() if v["class"] == "no_route"]
any_open = [k for k, v in ext.items() if v["result"] == "OPEN"]

MANDATORY = ["eutils.ncbi.nlm.nih.gov", "doi.org", "api.semanticscholar.org", "api.openai.com"]
mandatory_status = {}
for h in MANDATORY:
    ks = [k for k in ext if k.startswith("addr:" + h + "(")]
    if not ks:
        ks = [k for k in ext if k.startswith("name:" + h)]
    if not ks:
        mandatory_status[h] = "NOT_PROBED"
    else:
        mandatory_status[h] = "BLOCKED" if all(ext[k]["result"] == "BLOCKED" for k in ks) else "OPEN"
mandatory_bad = [h for h, v in mandatory_status.items() if v != "BLOCKED"]

lanes, lane_err = [], None
lane_sock = os.environ.get("EG_LANE_SOCK") or ""
if lane_sock:
    lport = int(os.environ.get("EG_LANE_PORT") or 0)
    want_model = os.environ.get("EG_LANE_MODEL") or ""
    want_tier = os.environ.get("EG_LANE_TIER") or ""
    sig, herr = None, None
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:%d/__proxy_health" % lport, timeout=20) as r:
            sig = json.loads(r.read())
    except Exception as e:
        herr = "%s: %s" % (type(e).__name__, str(e)[:160])
    if sig is None:
        lane_err = "lane health probe failed (%s)" % herr
    elif sig.get("mode") != "api":
        lane_err = "lane answered mode=%r, expected api" % sig.get("mode")
    elif want_model and sig.get("force_model") != want_model:
        lane_err = "lane force_model=%r, declared %r" % (sig.get("force_model"), want_model)
    elif want_tier and sig.get("service_tier") != want_tier:
        lane_err = "lane service_tier=%r, declared %r" % (sig.get("service_tier"), want_tier)
    lanes.append({"kind": "unix-socket", "path": lane_sock, "in_ns_tcp_port": lport,
                  "terminates_at": "llm_proxy api-mode", "verified": lane_err is None,
                  "health_signature": sig, "verification_error": lane_err})

declared = [int(x) for x in (os.environ.get("EG_DECLARED_PORTS") or "").replace(" ", "").split(",") if x]
observed, census_src, census_err = [], None, None
try:
    br = os.environ.get("EG_LANE_BRIDGE") or ""
    if br and os.path.exists(br):
        sys.path.insert(0, os.path.dirname(os.path.abspath(br)))
        import sock_bridge
        _ports, census_src = sock_bridge.listener_ports()
        observed = sorted(_ports)
    else:
        census_err = "sock_bridge.py not found; set EGRESS_LANE_BRIDGE to enable the census"
except Exception as e:
    census_err = "%s: %s" % (type(e).__name__, str(e)[:160])
unaccounted = sorted(set(observed) - set(declared)) if census_src else []
missing_declared = sorted(set(declared) - set(observed)) if census_src else []

out = {
    "schema": "egress_proof/1.3",
    "mandatory_blocked": mandatory_status,
    "sanctioned_lanes": lanes,
    "observed_listeners": {"ports": observed, "declared": sorted(declared),
                           "source": census_src, "unaccounted": unaccounted,
                           "declared_but_absent": missing_declared, "error": census_err},
    "host": os.environ.get("EG_HOST"),
    "command": os.environ.get("EG_CMD"),
    "namespace": "unshare -rn (loopback only)",
    "resolved_outside": resolved,
    "external": ext,
    "loopback_probe": lo,
    "lo_bringup": os.environ.get("EG_LO_ERR") or "ok",
    "route_level_proofs": route_proofs,
    "external_all_blocked": not any_open,
    "route_proven_unreachable": len(route_proofs) > 0,
    "loopback_reachable": lo["detail"] is not None and "ConnectionRefused" in (lo["detail"] or ""),
}
with open(os.environ["EGRESS_PROOF"], "w") as f:
    json.dump(out, f, indent=2)

if mandatory_bad:
    sys.stderr.write("egress_guard: REFUSING -- these hosts MUST be unreachable from inside the "
                     "namespace and are not: "
                     + json.dumps({h: mandatory_status[h] for h in mandatory_bad}) + "\n"
                     "The sanctioned lane is a unix socket and creates no route, so a route to "
                     "any of them means a lane exists that this proof does not describe.\n")
    sys.exit(4)
if lane_sock and lane_err:
    sys.stderr.write("egress_guard: REFUSING -- sanctioned lane declared but NOT verified: "
                     + lane_err + "\n")
    sys.exit(7)
if lane_sock and not census_src:
    sys.stderr.write("egress_guard: REFUSING -- a lane is declared but the listener census did not run: "
                     + str(census_err) + "\n")
    sys.exit(8)
if census_src and (unaccounted or missing_declared):
    sys.stderr.write("egress_guard: REFUSING -- in-namespace TCP listener set does not match the "
                     "declared set.\n  unaccounted (possible undeclared lane): "
                     + json.dumps(unaccounted) + "\n  declared but absent (service not up): "
                     + json.dumps(missing_declared) + "\n")
    sys.exit(8)
if not out["external_all_blocked"]:
    sys.stderr.write("egress_guard: REFUSING -- external host still reachable inside namespace\n")
    sys.stderr.write(json.dumps({k: ext[k] for k in any_open}, indent=2) + "\n")
    sys.exit(4)
if not out["route_proven_unreachable"]:
    sys.stderr.write("egress_guard: REFUSING -- every probe failed at DNS, none proved the route "
                     "is gone. That is compatible with a working route and broken resolver, so "
                     "it is not evidence of isolation.\n")
    sys.stderr.write(json.dumps(ext, indent=2) + "\n")
    sys.exit(6)
if not out["loopback_reachable"]:
    sys.stderr.write("egress_guard: REFUSING -- loopback not usable; vLLM would be unreachable\n")
    sys.stderr.write("lo bring-up: " + (os.environ.get("EG_LO_ERR") or "ok") + "\n")
    sys.stderr.write(json.dumps(lo, indent=2) + "\n")
    sys.exit(5)
sys.stderr.write("egress_guard: route proven unreachable (" + ", ".join(route_proofs[:2]) + "), loopback up -> proof at " + os.environ["EGRESS_PROOF"] + "\n")
PROBE

  rc=$?
  [ $rc -eq 0 ] || { [ -n "$EG_BRIDGE_PID" ] && kill $EG_BRIDGE_PID 2>/dev/null; exit $rc; }
  if [ -z "$EG_BRIDGE_PID" ]; then
    exec "$@"
  fi
  "$@"
  rc=$?
  kill $EG_BRIDGE_PID 2>/dev/null
  exit $rc
' bash "$@"
