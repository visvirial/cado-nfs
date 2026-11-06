#!/bin/bash
# Run sieving clients on Nosana (api.nosana.com, API key from ~/.config/nosana/api_key, or
# NOSANA_KEY_FILE), next to
# the Vast fleet.
#
#   nosana.sh prices                     GPU markets and their price ($/h, the network fee included)
#   nosana.sh gpus                       nodes per GPU model: all, available, working
#   nosana.sh create MODEL [N] [TIMEOUT] N (1) started deployments of one replica each of
#                                        host/container-boot.sh in the market of MODEL (5090, 4090,
#                                        3090, or a market address), strategy SIMPLE-EXTEND: the job
#                                        is extended by TIMEOUT minutes (180) each time, until the
#                                        credits run out; prints the deployment ids. One replica per
#                                        deployment, so that each client has its own id
#   nosana.sh list                       the deployments: id, status, name
#   nosana.sh jobs ID                    the jobs of a deployment
#   nosana.sh stop ID... / archive ID... stop deployments, archive stopped ones
#   nosana.sh balance                    the credits, in dollars (the API schema says cents; they are dollars)
#
# SERVER, CERTSHA1, MAKEFB and MAKEFB_OUT come from $FLEET_DIR/fleet.conf; the container fetches
# commit SRC_COMMIT (default: origin/rsa-270, so it must be pushed) from GitHub, as on
# SaladCloud; its client id is vast-svsalad-nosana-<model>-<n>.
set -eu
HERE=$(dirname "$(readlink -f "$0")"); . "$HERE/env.sh"; set -a; . "$FLEET_DIR/fleet.conf"; set +a
U=https://api.nosana.com/api
K() { cat ${NOSANA_KEY_FILE:-~/.config/nosana/api_key}; }
api() { m=$1; u=$2; shift 2; curl -sS -m 60 -X $m -H "Authorization: Bearer $(K)" -H "Content-Type: application/json" "$U$u" "$@"; }
market() { case $1 in
  5090|4090|3090) curl -sS -m 30 $U/markets/prices | python3 -c '
import json,sys
m=[x for x in json.load(sys.stdin) if x["name"]=="NVIDIA "+sys.argv[1]]
print(m[0]["address"] if m else "")' $1;;
  *) echo $1;; esac; }
case ${1:-} in
prices) curl -sS -m 30 $U/markets/prices | python3 -c '
import json,sys,os
S={k.replace("RTX_",""):float(v) for k,v in (x.split("=") for x in os.environ["SPQ_TABLE"].split())}
for m in sorted(json.load(sys.stdin), key=lambda m:m["usd_reward_per_hour"]):
    k=next((k for k in S if k in m["name"]),None)
    print("%-32s %s $%.3f/h%s"%(m["name"],m["address"],m["usd_reward_per_hour"], "  %.1f $/Msq"%(m["usd_reward_per_hour"]*S[k]/3600*1e6) if k else ""))';;
gpus) curl -sS -m 30 $U/reservations/gpus | python3 -c '
import json,sys
for g in json.load(sys.stdin)["gpus"]: print("%-34s nodes %4d available %4d working %4d"%(g["name"],g["nodes"],g["available"],g["working"]))';;
create) model=$2; count=${3:-1}; timeout=${4:-180}; mk=$(market $model); [ -n "$mk" ] || { echo "no market for $model" >&2; exit 1; }
  c=${SRC_COMMIT:-$(git -C "$HERE" rev-parse origin/rsa-270)}
  for i in $(seq $count); do
  n=$(( $(cat "$FLEET_DIR/nosana.counter" 2>/dev/null || echo 0) + 1 )); echo $n > "$FLEET_DIR/nosana.counter"
  body=$(python3 - "$model" "$mk" "$timeout" "$c" "$n" "$HERE/host/container-boot.sh" <<'PY'
import json,os,sys
model,mk,timeout,c,n,boot=sys.argv[1:]
env={"SRC_COMMIT":c,"SERVER":os.environ["SERVER"],"CERTSHA1":os.environ["CERTSHA1"],
     "MAKEFB":os.environ.get("MAKEFB",""),"MAKEFB_OUT":os.environ.get("MAKEFB_OUT",""),
     "SALAD_MACHINE_ID":"nosana-%s-%s"%(model[:6],n)}
print(json.dumps({"name":"rsa270-sv-%s-%s"%(model[:6],n),"market":mk,"replicas":1,"timeout":int(timeout),
  "strategy":"SIMPLE-EXTEND","autostart":True,
  "job_definition":{"version":"0.1","type":"container","meta":{"trigger":"api"},
    "ops":[{"type":"container/run","id":"sieve","args":{"image":"nvidia/cuda:12.8.1-devel-ubuntu22.04",
      "cmd":["/bin/bash","-c",open(boot).read()],"gpu":True,"env":env}}]}}))
PY
)
  api POST /deployments/create --data "$body" | python3 -c 'import json,sys
d=json.load(sys.stdin)
if not d.get("id"): sys.exit("create failed: %s" % json.dumps(d)[:500])
print(d["id"], d.get("status"))'
  done;;
list) api GET /deployments | python3 -c 'import json,sys
d=json.load(sys.stdin); d=d.get("deployments",d) if isinstance(d,dict) else d
for x in d: print(x.get("id"), x.get("status"), x.get("name"), x.get("market"))';;
jobs) api GET /deployments/$2/jobs | python3 -m json.tool | head -60;;
stop) shift; for i in "$@"; do echo "$i $(api POST /deployments/$i/stop --data {} | head -c 200)"; done;;
archive) shift; for i in "$@"; do echo "$i $(api POST /deployments/$i/archive --data {} | head -c 200)"; done;;
balance) api GET /credits/balance | python3 -c 'import json,sys
b=json.load(sys.stdin)
if "assignedCredits" not in b: sys.exit(json.dumps(b)[:300])
# in dollars, though the API schema says USD cents: 100 dollars paid in read 100 (2026-11-06)
print("available $%.2f (assigned %.2f reserved %.2f settled %.2f)" % (b["assignedCredits"]-b["reservedCredits"]-b["settledCredits"], b["assignedCredits"], b["reservedCredits"], b["settledCredits"]))';;
*) sed -n '2,22p' "$0"; exit 1;;
esac
