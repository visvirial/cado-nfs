#!/bin/bash
# Run sieving clients on SaladCloud, next to the Vast fleet.
#
#   salad.sh classes                         GPU classes and their prices
#   salad.sh create NAME REPLICAS PRIORITY GPU-CLASS-NAME...
#                                            a container group of host/container-boot.sh, started
#   salad.sh status [NAME]                   groups, or the instances of one
#   salad.sh replicas NAME N                 change the number of replicas
#   salad.sh start NAME                      start a stopped group (one made over the replica
#                                            quota stays stopped)
#   salad.sh logs NAME [MINUTES]             the container output of the last MINUTES (30)
#   salad.sh delete NAME                    stop and delete a group
#   salad.sh counts NAME                     the replicas a group asks for, and those it has
#                                            (creating or running)
#   salad.sh fit PRIORITY                    the GPU classes whose price at PRIORITY is within CAP
#                                            dollars per million special-q (SPQ_TABLE)
#   salad.sh quota                           the replica quota of the organization
#   salad.sh manage                          start salad-fleet.sh, which fills the quota with
#                                            batch replicas first and low ones after
#
# The API key is read from ~/.config/salad/api_key; SALAD_ORG and SALAD_PROJECT, SERVER,
# CERTSHA1, MAKEFB and MAKEFB_OUT from $FLEET_DIR/fleet.conf. PRIORITY is high, medium, low or batch; a lower
# priority is cheaper and is interrupted more. SRC_COMMIT (default: origin/rsa-270) is the
# commit the containers fetch from GitHub, so it must be pushed.
set -eu
HERE=$(dirname "$(readlink -f "$0")"); . "$HERE/env.sh"; set -a; . "$FLEET_DIR/fleet.conf"; set +a
K=$(cat ~/.config/salad/api_key)
A=https://api.salad.com/api/public/organizations/${SALAD_ORG:?set SALAD_ORG in fleet.conf}
P=$A/projects/${SALAD_PROJECT:?set SALAD_PROJECT in fleet.conf}/containers
api() { m=$1; u=$2; shift 2; curl -sS -X $m -H "Salad-Api-Key: $K" -H "Content-Type: ${CT:-application/json}" "$u" "$@"; }
case ${1:-} in
classes) api GET $A/gpu-classes | python3 -c '
import json,sys
for c in json.load(sys.stdin)["items"]:
    print(c["name"], " ".join("%s=%s" % (p["priority"],p["price"]) for p in c.get("prices",[])))';;
create)
  name=$2; n=$3; prio=$4; shift 4
  ids=$(api GET $A/gpu-classes | python3 -c '
import json,sys
want=sys.argv[1:]; got={c["name"]:c["id"] for c in json.load(sys.stdin)["items"]}
miss=[w for w in want if w not in got]
if miss: sys.exit("unknown GPU classes: "+", ".join(miss))
print(json.dumps([got[w] for w in want]))' "$@")
  commit=${SRC_COMMIT:-$(git -C "$HERE" rev-parse origin/rsa-270)}
  python3 - "$name" "$n" "$prio" "$ids" "$commit" "$HERE/host/container-boot.sh" > /tmp/salad-create.$$ <<'EOF'
import json,os,sys
name,n,prio,ids,commit,boot=sys.argv[1:]
print(json.dumps({
  "name": name, "replicas": int(n), "autostart_policy": True, "restart_policy": "always",
  "container": {
    "image": "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    "resources": {"cpu": int(os.environ.get("SALAD_CPU", 16)), "memory": int(os.environ.get("SALAD_MEMORY", 24576)), "gpu_classes": json.loads(ids)},
    "priority": prio,
    "command": ["bash", "-c", open(boot).read()],
    "environment_variables": {"SRC_COMMIT": commit, "SERVER": os.environ["SERVER"], "CERTSHA1": os.environ["CERTSHA1"],
                            "MAKEFB": os.environ.get("MAKEFB", ""), "MAKEFB_OUT": os.environ.get("MAKEFB_OUT", "")}}}))
EOF
  api POST $P --data @/tmp/salad-create.$$ | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("name"),d.get("current_state",d))'
  rm -f /tmp/salad-create.$$;;
status)
  if [ -z "${2:-}" ]; then api GET $P | python3 -c '
import json,sys
for g in json.load(sys.stdin)["items"]:
    s=g["current_state"]; print(g["name"],g["replicas"],g.get("priority",g["container"].get("priority")),s["status"],s.get("instance_status_counts"))'
  else api GET $P/$2/instances | python3 -c '
import json,sys
for i in json.load(sys.stdin)["instances"]: print(i.get("machine_id"),i.get("state"),i.get("update_time"),i.get("ready"),i.get("started"))'; fi;;
replicas) CT=application/merge-patch+json api PATCH $P/$2 --data "{\"replicas\": $3}" >/dev/null && echo "$2 replicas $3";;
logs) q="resource.type = \"container\" and resource.labels.project_name = \"$SALAD_PROJECT\" and resource.labels.container_group_name = \"$2\""
  api POST $A/log-entries --data "$(python3 -c 'import json,sys;print(json.dumps({"query":sys.argv[1],"start_time":sys.argv[2],"end_time":sys.argv[3],"page_size":100,"sort_order":"desc"}))' "$q" $(date -u -d "-${3:-30} min" +%FT%TZ) $(date -u +%FT%TZ))" | python3 -c '
import json,sys
d=json.load(sys.stdin)
for e in reversed(d.get("items",[]) if isinstance(d,dict) else d):
    l=e.get("resource",{}).get("labels",{}); print(e.get("time","")[11:19], l.get("instance_id","")[:8], e.get("message",e.get("text_log",e.get("json_log",""))))
if isinstance(d,dict) and "items" not in d: print(d)';;
start) api POST $P/$2/start && echo "$2 started";;
counts) api GET $P/$2 | python3 -c '
import json,sys
g=json.load(sys.stdin)
if "replicas" not in g: sys.exit("no group: %s" % json.dumps(g)[:200])
c=g["current_state"]["instance_status_counts"]
print(g["replicas"], c.get("creating_count",0)+c.get("running_count",0), g["current_state"]["status"])';;
fit) api GET $A/gpu-classes | python3 -c '
import json,os,sys
S={k:float(v) for k,v in (x.split("=") for x in os.environ["SPQ_TABLE"].split())}
prio,cap=sys.argv[1],float(sys.argv[2])
for c in json.load(sys.stdin)["items"]:
    k=c["name"].split(" (")[0].replace(" ","_")      # "RTX 4090 (24 GB)" -> RTX_4090
    p=[float(x["price"]) for x in c.get("prices",[]) if x["priority"]==prio]
    if k in S and p and p[0]*S[k]/3600*1e6<=cap: print(c["name"])' "$2" "$CAP";;
quota) curl -sS -H "Salad-Api-Key: $K" $A/quotas | python3 -c 'import json,sys
print(json.load(sys.stdin)["container_groups_quotas"]["container_replicas_quota"])';;
manage)
  if running salad-fleet.sh; then echo "already running"; exit 0; fi
  mkdir -p "$FLEET_DIR/salad"; cd "$FLEET_DIR/salad"
  setsid nohup bash "$HERE/salad-fleet.sh" >> manager.out 2>&1 < /dev/null &
  echo "started; log in $FLEET_DIR/salad/events.log";;
delete) api POST $P/$2/stop >/dev/null || true; api DELETE $P/$2 && echo "$2 deleted";;
*) sed -n '2,26p' "$0"; exit 1;;
esac
