#!/bin/bash
# Keep a Vast.ai fleet of single-GPU clients on a polynomial search served by
# c270-polyselect-server. Every minute it rents each offer under the cost cap (CAP dollars
# per hour per ad/s, or PRICE_<GPU> dollars per hour where set), replaces hosts (checked every 10 minutes) that never come up, stop answering, fail to build, fail two work units in a row, or run below
# 75% of their GPU's speed (and never rents those machines again), with BID=1 also rents interruptible
# offers at BID_MARGIN above their minimum bid (and gives them up once outbid), gives back idle hosts beyond
# MINKEEP, restarts clients that ended when the server stopped serving work units, and destroys every host when SERVER has not answered for 2 h (the server exits when
# the search ends), the credit is below MINCREDIT, or STOP=1. Started by "vast.sh manage"; logs to
# $FLEET_DIR/events.log. The speeds are those measured with gps1 -sopt-gpu 0 at P=2e7.
cd $FLEET_DIR; E=$FLEET_DIR/events.log
log() { echo "$(date '+%F %T') $*" >> $E; }
declare -A RATE=([RTX_3090]=0.496 [RTX_5080]=0.522 [RTX_4080S]=0.415 [RTX_4090]=0.57 [RTX_5090]=0.96)
declare -A ARCH=([RTX_3090]=86 [RTX_5080]=100 [RTX_4080S]=89 [RTX_4090]=89 [RTX_5090]=100)
IMG=nvidia/cuda:12.8.1-devel-ubuntu22.04
touch machines.bad; [ -f counter ] || echo 0 > counter; srv_down_since=
live() { for d in m*/; do d=${d%/}; [ -f $d/id ] && [ ! -f $d/dead ] && echo $d; done; }
kill_host() { l=$1; shift; $V destroy instance $(cat $l/id) -y >/dev/null 2>&1; touch $l/dead; log "$l destroyed: $*"; }
offers() { # "gpu offer machine bid" lines under the cap, best first, skipping bad machines and those in use;
  # bid is the price of an interruptible rental (BID=1), or - for an on-demand one
  q="num_gpus=1 rentable=true rented=false verified=true direct_port_count>=4 cpu_cores_effective>=16 reliability>=0.97 cuda_max_good>=12.8 inet_down>=200 gpu_name in [RTX_3090,RTX_5080,RTX_4080S,RTX_4090,RTX_5090]"
  { $V search offers "$q" --raw --limit 1000 2>/dev/null; echo; [ "${BID:-0}" = 1 ] && $V search offers "$q" --type bid --raw --limit 1000 2>/dev/null; } | python3 -c '
import json,sys,os
R={"RTX 3090":.496,"RTX 5080":.522,"RTX 4080S":.415,"RTX 4090":.57,"RTX 5090":.96}
cap=float(sys.argv[1]);margin=float(sys.argv[2]);skip=set(open("machines.bad").read().split())|set(sys.argv[3:]);best={}
parts=[p for p in sys.stdin.read().split("\n\n") if p.strip()]
for k,p in enumerate(parts):
    for x in json.loads(p):
        bid=None
        if k==1:
            if not x.get("min_bid"): continue
            bid=round(x["min_bid"]*(1+margin)+0.0005,3); price=bid
        else: price=x["dph_total"]
        m=str(x["machine_id"]);e=price/R[x["gpu_name"]]
        lim=os.environ.get("PRICE_"+x["gpu_name"].replace(" ","_"))
        if m in skip or (price>float(lim) if lim else e>cap) or x.get("inet_down_cost",1)>.03: continue
        if m not in best or e<best[m][0]: best[m]=(e,x,bid)
for e,x,bid in sorted(best.values(),key=lambda t:t[0]): print(x["gpu_name"].replace(" ","_"),x["id"],x["machine_id"],bid if bid else "-")' $CAP ${BID_MARGIN:-0.05} $(for d in $(active); do cat $d/machine; done | sort -u); }
launch() { # label gpu offer machine bid; runs in the background
  l=$1; mkdir -p $l; echo $2 > $l/gpu; echo $4 > $l/machine; date +%s > $l/t0; touch $l/launching
  b=; [ "${5:--}" != - ] && { b=$5; echo $b > $l/bid; }
  r=$(BID_PRICE=$b timeout 1200 bash $HERE/vast-launch.sh $l ${ARCH[$2]} $IMG $3 2>&1 | tail -1); log "$l $2 machine $4${b:+ bid \$$b}: $r"
  [ -f $l/id ] || { echo $4 >> machines.bad; touch $l/dead; }
  rm -f $l/launching
}
active() { for d in m*/; do d=${d%/}; [ ! -f $d/dead ] && { [ -f $d/id ] || [ -f $d/launching ]; } && echo $d; done; }
launching() { ls -d m*/launching 2>/dev/null | wc -l; }
rm -f m*/launching   # launches of an earlier manager are over
PROBE='cd /workspace; echo "state [$(cat state 2>/dev/null)]"; echo "procs $(pgrep -xc gps1) $(pgrep -xc polyselect_ropt)"; w=$(grep "Subprocess has PID" out/client.log 2>/dev/null | tail -1 | sed "s/.*\[\(.*\)\].*/\1/"); if [ -n "$w" ]; then t=$(date -d "$w" +%s); else t=$(stat -c %Y boot.out 2>/dev/null || date +%s); fi; echo "age $(( $(date +%s) - t ))"; cat out/wu.log 2>/dev/null'
export PROBE
last_health=0; nidle=0
while :; do
  unset $(compgen -v PRICE_); set -a; . ./fleet.conf; set +a
  credit=$($V show user --raw 2>/dev/null | python3 -c 'import json,sys;print(int(json.load(sys.stdin)["credit"]))' 2>/dev/null)
  if curl -sk -m 20 $SERVER/ | grep -q Hello; then srv_down_since=
  else [ -z "$srv_down_since" ] && log "server unreachable"; srv_down_since=${srv_down_since:-$(date +%s)}; fi
  if [ "$STOP" = 1 ] || { [ -n "$credit" ] && [ "$credit" -lt "$MINCREDIT" ]; } || { [ -n "$srv_down_since" ] && [ $(( $(date +%s) - srv_down_since )) -gt 7200 ]; }; then
    while [ $(launching) -gt 0 ]; do sleep 10; done
    for l in $(live); do kill_host $l "stop (STOP=$STOP credit=$credit server_down_since=$srv_down_since)"; done; log "fleet stopped"; exit 0; fi
  if [ $(( $(date +%s) - last_health )) -ge 600 ]; then
    last_health=$(date +%s)
    st=$($V show instances --raw 2>/dev/null | python3 -c 'import json,sys
for i in json.load(sys.stdin): print(i["id"],i.get("actual_status"),round(i.get("dph_total",0),3))' 2>/dev/null)
    # probe every host over SSH in parallel; the decisions below read the answers
    for l in $(live); do [ -f $l/launching ] || [ ! -f $l/addr ] || echo $l; done | xargs -r -P 32 -I{} \
      bash -c 'timeout 90 "$HERE/vast.sh" ssh {} "$PROBE" 2>/dev/null | grep -v -i "welcome\|have fun" > {}/probe.new; mv {}/probe.new {}/probe'
    idle=()
    for l in $(live); do
      id=$(cat $l/id); a=$(echo "$st" | awk -v i=$id '$1==i{print $2}'); age=$(( $(date +%s) - $(cat $l/t0) ))
      if [ -n "$st" ] && [ -z "$a" ]; then touch $l/dead; log "$l instance $id is gone"; continue; fi
      [ -f $l/launching ] && continue
      # an interruptible host stops when outbid: give it up, without holding it against the machine
      if [ -f $l/bid ] && [ -n "$a" ] && [ "$a" != running ]; then kill_host $l "interrupted (status $a, bid \$$(cat $l/bid))"; continue; fi
      if [ ! -f $l/addr ]; then [ $age -gt 1800 ] && { echo $(cat $l/machine) >> machines.bad; kill_host $l "never came up"; }; continue; fi
      s=$(cat $l/probe 2>/dev/null)
      if [ -z "$s" ]; then m=$(( $(cat $l/miss 2>/dev/null || echo 0) + 1 )); echo $m > $l/miss
        [ "$a" != running ] && [ $m -ge 2 ] || [ $m -ge 4 ] && { echo $(cat $l/machine) >> machines.bad; kill_host $l "unreachable, status $a"; }; continue; fi
      echo 0 > $l/miss
      if echo "$s" | grep -q '^state \[done rc=0\]'; then
        timeout 90 "$HERE/vast.sh" ssh $l "$RESTART" > /dev/null 2>&1; log "$l client ended, restarted"; continue; fi
      if ! echo "$s" | grep -q '^state \[\]'; then echo $(cat $l/machine) >> machines.bad; kill_host $l "$(echo "$s" | head -1)"; continue; fi
      # the last two work units failed: the wrapper should have stopped the client already
      if [ "$(echo "$s" | awk 'NF==5 {r[++n]=$5} END{print (n>=2 && r[n]!=0 && r[n-1]!=0) ? "y" : "n"}')" = y ]; then
        echo $(cat $l/machine) >> machines.bad; kill_host $l "work units failing: $(echo "$s" | awk 'NF==5' | tail -2 | awk '{print $5}' | paste -sd,)"; continue; fi
      rate=$(echo "$s" | awk 'NF==5 && $5==0 {ad+=($2-$1)/$3; t+=$4; n++} END{if(n>=2 && t>0) printf "%.3f", ad/t}')
      if [ -n "$rate" ] && python3 -c "import sys;sys.exit(0 if $rate < 0.75*${RATE[$(cat $l/gpu)]} else 1)"; then
        echo $(cat $l/machine) >> machines.bad; kill_host $l "slow: $rate ad/s, $(cat $l/gpu)"; continue; fi
      echo "$s" | grep -q '^procs 0 0' && [ $(echo "$s" | awk '/^age/{print $2}') -gt 2700 ] && idle+=($l)
    done
    n=$(live | wc -l); keep=$(( n - ${#idle[@]} ))
    for l in "${idle[@]}"; do [ $keep -ge $MINKEEP ] && { kill_host $l "idle"; continue; }; keep=$((keep+1)); done
    nidle=${#idle[@]}
    log "health: live $(live | wc -l) launching $(launching) idle $nidle credit $credit rate \$$(echo "$st" | awk '{s+=$3} END{print s}')/h (all instances)"
  fi
  # no new hosts once hosts sit idle: the search is in its tail or done
  if [ $nidle -eq 0 ] && [ -z "$srv_down_since" ]; then
    want=$(( MAXHOSTS - $(active | wc -l) )); room=$(( 8 - $(launching) )); [ $want -gt $room ] && want=$room
    if [ $want -gt 0 ]; then
      offers | head -$want > offers.now
      while read g o m b; do c=$(( $(cat counter) + 1 )); echo $c > counter; launch $(printf 'm%03d' $c) $g $o $m $b & sleep 2; done < offers.now
    fi
  fi
  sleep 60
done
