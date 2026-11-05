#!/bin/bash
# Run sieving clients on Clore.ai, next to the Vast fleet.
#
#   clore.sh manage                start the Clore fleet manager, clore-fleet.sh (again)
#   clore.sh offers [MINREL [N]]   the N (40) cheapest free single-GPU servers per special-q
#   clore.sh rent SERVER-ID...     rent on demand and start host/container-boot.sh there
#   clore.sh status                the orders, with their SSH ports
#   clore.sh ssh ORDER-ID [CMD]    SSH into the container of an order
#   clore.sh cancel ORDER-ID...    end orders; fails unless every order is ended
#   clore.sh balance               the USD balance
#   clore.sh tsv                   the orders, one per line: order server gpu host port $/h
#   clore.sh candidates CAP MINREL the free servers under CAP dollars per million special-q,
#                                  cheapest per special-q first:
#                                  server gpu
#
# The API key is read from ~/.config/clore/api_key, SERVER, CERTSHA1, MAKEFB and MAKEFB_OUT
# from $FLEET_DIR/fleet.conf; payment is in USD-Blockchain. Prices on the marketplace are per
# server and day. The image is the public CUDA one, which has no SSH server: the start
# command installs one with the fleet key, then fetches host/container-boot.sh of SRC_COMMIT (default:
# origin/rsa-270, so it must be pushed) from GitHub and runs it in the background. GPU names
# are printed with _ for spaces.
set -eu
HERE=$(dirname "$(readlink -f "$0")"); . "$HERE/env.sh"; set -a; . "$FLEET_DIR/fleet.conf"; set +a
K=$(cat ~/.config/clore/api_key); U=https://api.clore.ai/v1
# The API allows about one request a second per key, and answers more with an error, so every
# request, from any clore.sh at once, waits its turn on a lock and keeps it a second longer.
api() { m=$1; u=$2; shift 2
  ( flock 9; curl -sS -m 60 -X $m -H "auth: $K" -H 'Content-Type: application/json' "$U/$u" "$@"; r=$?; sleep 1; exit $r ) 9> "$FLEET_DIR/clore-api.lock"; }
orders() { # the API allows about one request a second
  for i in 1 2 3 4 5; do r=$(api GET my_orders) && echo "$r" | grep -q '"orders"' && { echo "$r"; return; }; sleep 2; done; echo "$r" >&2; return 1; }
market() { # mode cap minrel n
  curl -sS -m 60 $U/marketplace | python3 -c '
import json,sys
import os
S={k.replace("_"," "):float(v) for k,v in (x.split("=") for x in os.environ["SPQ_TABLE"].split())}
mode,cap,minrel,n=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]); v=[]
for x in json.load(sys.stdin)["servers"]:
    a=x["gpu_array"]
    if x["rented"] or len(a)!=1 or a[0] not in S or (x.get("reliability") or 0)<minrel: continue
    try:
        if float(x.get("cuda_version") or 0)<12.8: continue
    except ValueError: continue
    s=x["specs"]; t=int(s["cpus"].split("/")[1]); p=x["price"]["usd"].get("on_demand_usd")
    e=p/24*S[a[0]]/3600*1e6 if p else 0
    try: ram=float(s.get("ram") or 0)
    except (TypeError, ValueError): ram=0
    if not p or t<16 or (ram and ram<31) or "USD-Blockchain" not in x["allowed_coins"] or e>cap: continue
    v.append((e,p/24,a[0].replace(" ","_"),x["id"],t,s["net"]["cc"],x["reliability"]))
for r in sorted(v)[:n]:
    if mode=="human": print("%5.1f $/Msq  $%.3f/h  %-16s id %-7s %3d threads %s reliability %.3f"%r)
    else: print(r[3],r[2])' "$@"; }
case ${1:-} in
manage)
  if running clore-fleet.sh; then echo "already running"; exit 0; fi
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
url=f"https://raw.githubusercontent.com/visvirial/cado-nfs/{c}/gpu-siever/sieve-fleet/host/container-boot.sh"
cmd=("(export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq openssh-server curl ca-certificates"
     " && mkdir -p /root/.ssh /run/sshd && echo \"$FLEET_PUB\" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys"
     " && /usr/sbin/sshd && curl -sfL --retry 5 " + url + " > /root/container-boot.sh && bash /root/container-boot.sh) > /root/container-boot.out 2>&1 &")
print(json.dumps({"currency":"USD-Blockchain","image":"nvidia/cuda:12.8.1-devel-ubuntu22.04","renting_server":int(sid),
  "type":"on-demand","ports":{"22":"tcp"},"ssh_key":pub,"command":cmd,
  "env":{"SRC_COMMIT":c,"SERVER":os.environ["SERVER"],"CERTSHA1":os.environ["CERTSHA1"],"SALAD_MACHINE_ID":"clore-"+sid,"FLEET_PUB":pub,
         "MAKEFB":os.environ.get("MAKEFB",""),"MAKEFB_OUT":os.environ.get("MAKEFB_OUT","")}}))
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
  exec ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=60 -o ConnectionAttempts=3 -p $p root@$h "$@";;
cancel) shift; rc=0
  for id in "$@"; do
    for i in 1 2 3 4 5; do r=$(api POST cancel_order --data "{\"id\":$id}") && echo "$r" | grep -q '"code": *0[,}]' && break; sleep $((2 * i)); done
    echo "$id $r"; echo "$r" | grep -q '"code": *0[,}]' || { echo "order $id not cancelled" >&2; rc=1; }; done
  exit $rc;;
*) sed -n '2,20p' "$0"; exit 1;;
esac
