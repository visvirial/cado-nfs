#!/bin/bash
# Run polynomial search clients on Clore.ai, next to the Vast fleet.
#
#   clore.sh manage                start the Clore fleet manager, clore-fleet.sh (again)
#   clore.sh offers [MINREL [N]]   the N (40) cheapest free single-GPU servers per ad/s
#   clore.sh rent SERVER-ID...     rent on demand and start salad-boot.sh there
#   clore.sh status                the orders, with their SSH ports
#   clore.sh ssh ORDER-ID [CMD]    SSH into the container of an order
#   clore.sh cancel ORDER-ID...    end orders
#   clore.sh balance               the USD balance
#   clore.sh tsv                   the orders, one per line: order server gpu host port $/h
#   clore.sh candidates CAP MINREL the free servers under CAP, cheapest per ad/s first:
#                                  server gpu
#
# The API key is read from ~/.config/clore/api_key, SERVER and CERTSHA1 from
# $FLEET_DIR/fleet.conf; payment is in USD-Blockchain. Prices on the marketplace are per
# server and day. The image is the public CUDA one, which has no SSH server: the start
# command installs one with the fleet key, then fetches salad-boot.sh of SRC_COMMIT (default:
# origin/rsa-270, so it must be pushed) from GitHub and runs it in the background. GPU names
# are printed with _ for spaces.
set -eu
HERE=$(dirname "$(readlink -f "$0")"); . "$HERE/env.sh"; set -a; . "$FLEET_DIR/fleet.conf"; set +a
K=$(cat ~/.config/clore/api_key); U=https://api.clore.ai/v1
api() { m=$1; u=$2; shift 2; curl -sS -m 60 -X $m -H "auth: $K" -H 'Content-Type: application/json' "$U/$u" "$@"; }
orders() { # the API allows about one request a second
  for i in 1 2 3 4 5; do r=$(api GET my_orders) && echo "$r" | grep -q '"orders"' && { echo "$r"; return; }; sleep 2; done; echo "$r" >&2; return 1; }
market() { # mode cap minrel n
  curl -sS -m 60 $U/marketplace | python3 -c '
import json,sys
R={"RTX 3090":.496,"RTX 5080":.522,"RTX 4080 SUPER":.415,"RTX 4090":.57,"RTX 5090":.96}
mode,cap,minrel,n=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]); v=[]
for x in json.load(sys.stdin)["servers"]:
    a=x["gpu_array"]
    if x["rented"] or len(a)!=1 or a[0] not in R or (x.get("reliability") or 0)<minrel: continue
    try:
        if float(x.get("cuda_version") or 0)<12.8: continue
    except ValueError: continue
    s=x["specs"]; t=int(s["cpus"].split("/")[1]); p=x["price"]["usd"].get("on_demand_usd")
    if not p or t<16 or "USD-Blockchain" not in x["allowed_coins"] or p/24/R[a[0]]>cap: continue
    v.append((p/24/R[a[0]],p/24,a[0].replace(" ","_"),x["id"],t,s["net"]["cc"],x["reliability"]))
for r in sorted(v)[:n]:
    if mode=="human": print("%.2f $/h/(ad/s)  $%.3f/h  %-16s id %-7s %3d threads %s reliability %.3f"%r)
    else: print(r[3],r[2])' "$@"; }
case ${1:-} in
manage)
  if ps -eo cmd | awk -v f="$HERE/clore-fleet.sh" '$1=="bash" && $2==f {r=1} END {exit !r}'; then echo "already running"; exit 0; fi
  mkdir -p "$FLEET_DIR/clore"; cd "$FLEET_DIR/clore"
  setsid nohup bash "$HERE/clore-fleet.sh" >> manager.out 2>&1 < /dev/null &
  echo "started; log in $FLEET_DIR/clore/events.log";;
offers) market human 1e9 "${2:-0.95}" "${3:-40}";;
candidates) market tsv "$2" "$3" 1000;;
rent)
  shift; c=${SRC_COMMIT:-$(git -C "$HERE" rev-parse origin/rsa-270)}
  for sid in "$@"; do
    body=$(python3 - "$sid" "$c" "$(cat $KEY.pub)" <<'EOF'
import json,os,sys
sid,c,pub=sys.argv[1:]
url=f"https://raw.githubusercontent.com/visvirial/cado-nfs/{c}/gpu-siever/vast-polyselect/salad-boot.sh"
cmd=("(export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq openssh-server curl ca-certificates"
     " && mkdir -p /root/.ssh /run/sshd && echo \"$FLEET_PUB\" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys"
     " && /usr/sbin/sshd && curl -sfL --retry 5 " + url + " > /root/boot-clore.sh && bash /root/boot-clore.sh) > /root/boot-clore.out 2>&1 &")
print(json.dumps({"currency":"USD-Blockchain","image":"nvidia/cuda:12.8.1-devel-ubuntu22.04","renting_server":int(sid),
  "type":"on-demand","ports":{"22":"tcp"},"ssh_key":pub,"command":cmd,
  "env":{"SRC_COMMIT":c,"SERVER":os.environ["SERVER"],"CERTSHA1":os.environ["CERTSHA1"],"SALAD_MACHINE_ID":"clore-"+sid,"FLEET_PUB":pub}}))
EOF
)
    echo "$sid $(api POST create_order --data "$body")"; sleep 1.5
  done;;
tsv) orders | python3 -c '
import json,sys
for o in json.load(sys.stdin)["orders"]:
    p=[t.split(":")[1] for t in (o.get("tcp_ports") or []) if t.startswith("22:")]
    print(o["id"],o["si"],(o["gpu_array"][0] if o.get("gpu_array") else "-").replace(" ","_"),
          (o.get("pub_cluster") or ["-"])[0],p[0] if p else "-","%.4f"%(o["price"]/24))';;
status) orders | python3 -c '
import json,sys
for o in json.load(sys.stdin)["orders"]:
    print(o["id"],"server",o["si"],(o["gpu_array"][0] if o.get("gpu_array") else "-").replace(" ","_"),"online" if o.get("online") else "offline",
          "ssh",(o.get("pub_cluster") or ["-"])[0],",".join(o.get("tcp_ports") or []),"$%.3f/h spent $%.3f"%(o["price"]/24,o.get("spend",0)))';;
balance) api GET wallets | python3 -c 'import json,sys
print([w["balance"] for w in json.load(sys.stdin)["wallets"] if w["name"]=="USD-Blockchain"][0])';;
ssh) id=$2; shift 2
  read h p < <(orders | python3 -c 'import json,sys
for o in json.load(sys.stdin)["orders"]:
    if str(o["id"])==sys.argv[1]: print(o["pub_cluster"][0],[t for t in o["tcp_ports"] if t.startswith("22:")][0].split(":")[1])' $id)
  exec ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $p root@$h "$@";;
cancel) shift; for id in "$@"; do echo "$id $(api POST cancel_order --data "{\"id\":$id}")"; sleep 1.2; done;;
*) sed -n '2,20p' "$0"; exit 1;;
esac
