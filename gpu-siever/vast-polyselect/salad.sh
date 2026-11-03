#!/bin/bash
# Run polynomial search clients on SaladCloud, next to the Vast fleet.
#
#   salad.sh classes                         GPU classes and their prices
#   salad.sh create NAME REPLICAS PRIORITY GPU-CLASS-NAME...
#                                            a container group of salad-boot.sh, started
#   salad.sh status [NAME]                   groups, or the instances of one
#   salad.sh replicas NAME N                 change the number of replicas
#   salad.sh start NAME                      start a stopped group (one made over the replica
#                                            quota stays stopped)
#   salad.sh logs NAME [MINUTES]             the container output of the last MINUTES (30)
#   salad.sh delete NAME                    stop and delete a group
#
# The API key is read from ~/.config/salad/api_key; SALAD_ORG and SALAD_PROJECT, and SERVER
# and CERTSHA1, from $FLEET_DIR/fleet.conf. PRIORITY is high, medium, low or batch; a lower
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
  python3 - "$name" "$n" "$prio" "$ids" "$commit" "$HERE/salad-boot.sh" > /tmp/salad-create.$$ <<'EOF'
import json,os,sys
name,n,prio,ids,commit,boot=sys.argv[1:]
print(json.dumps({
  "name": name, "replicas": int(n), "autostart_policy": True, "restart_policy": "always",
  "container": {
    "image": "nvidia/cuda:12.8.1-devel-ubuntu22.04",
    "resources": {"cpu": 16, "memory": 16384, "gpu_classes": json.loads(ids)},
    "priority": prio,
    "command": ["bash", "-c", open(boot).read()],
    "environment_variables": {"SRC_COMMIT": commit, "SERVER": os.environ["SERVER"], "CERTSHA1": os.environ["CERTSHA1"]}}}))
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
delete) api POST $P/$2/stop >/dev/null || true; api DELETE $P/$2 && echo "$2 deleted";;
*) sed -n '2,18p' "$0"; exit 1;;
esac
