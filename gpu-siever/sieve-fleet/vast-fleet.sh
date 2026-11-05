#!/bin/bash
# Keep a Vast.ai fleet of single-GPU clients on the sieving served by c270-sieve-server.
# Started by "vast.sh manage"; logs to $FLEET_DIR/events.log, one directory m<NNN> per host.
#
# Every minute it rents, up to MAXHOSTS, each offer of the GPUS models under the cost cap (CAP
# dollars per million special-q at the seconds per special-q of SPQ_TABLE, or PRICE_<GPU>
# dollars per hour where set), at most 8 launches at a time; with BID=1 it also rents
# interruptible offers at BID_MARGIN above their minimum bid. Every 10 minutes it probes the
# hosts over SSH and destroys, and never rents again, those that never come up, stop answering,
# fail to build, fail two work units in a row, never get a work unit, or run slower than 1/0.75
# of their GPU's seconds per special-q. An outbid interruptible host is given up without
# holding it against the machine. Clients that ended because the server stopped serving work
# units are restarted. Hosts idle for 45 minutes while the server answers are destroyed, and
# none rented while any is idle: the sieving is in its tail or done. Every host is destroyed
# when SERVER has not answered for 2 h, the credit is below MINCREDIT, or STOP=1.
cd $FLEET_DIR; E=$FLEET_DIR/events.log
echo $$ > $FLEET_DIR/vast-fleet.sh.pid   # for running() in env.sh
log() { echo "$(date '+%F %T') $*" >> $E; }
declare -A SPQ; for x in $SPQ_TABLE; do SPQ[${x%=*}]=${x#*=}; done
IMG=nvidia/cuda:12.8.1-devel-ubuntu22.04
touch machines.bad; [ -f counter ] || echo 0 > counter; srv_down_since=
live() { for d in m*/; do d=${d%/}; [ -f $d/id ] && [ ! -f $d/dead ] && echo $d; done; }
active() { for d in m*/; do d=${d%/}; [ ! -f $d/dead ] && { [ -f $d/id ] || [ -f $d/launching ]; } && echo $d; done; }
launching() { ls -d m*/launching 2>/dev/null | wc -l; }
kill_host() { l=$1; shift; $V destroy instance $(cat $l/id) -y >/dev/null 2>&1; touch $l/dead; log "$l $(cat $l/gpu) destroyed: $*"; }
bad() { cat $1/machine >> machines.bad; }
offers() { # "gpu offer machine bid" lines under the cap, cheapest per special-q first, skipping bad
  # machines and those in use; bid is the price of an interruptible rental (BID=1), or - for an on-demand one
  q="num_gpus=1 rentable=true rented=false verified=${VERIFIED:-true} direct_port_count>=4 cpu_cores_effective>=${MINCPU:-16} cpu_ram>=32 gpu_ram>=23 disk_space>=50 reliability>=0.95 cuda_max_good>=12.8 inet_down>=100 gpu_name in [${GPUS}]"
  { $V search offers "$q" --raw --limit 1000 2>/dev/null; echo; [ "${BID:-0}" = 1 ] && $V search offers "$q" --type bid --raw --limit 1000 2>/dev/null; } | python3 -c '
import json,sys,os
S={k.replace("_"," "):float(v) for k,v in (x.split("=") for x in os.environ["SPQ_TABLE"].split())}
cap=float(sys.argv[1]);margin=float(sys.argv[2]);skip=set(open("machines.bad").read().split())|set(sys.argv[3:]);best={}
parts=[p for p in sys.stdin.read().split("\n\n") if p.strip()]
for k,p in enumerate(parts):
    for x in json.loads(p):
        if x["gpu_name"] not in S: continue
        bid=None
        if k==1:
            if not x.get("min_bid"): continue
            bid=round(x["min_bid"]*(1+margin)+0.0005,3); price=bid
        else: price=x["dph_total"]
        m=str(x["machine_id"]);e=price*S[x["gpu_name"]]/3600*1e6
        lim=os.environ.get("PRICE_"+x["gpu_name"].replace(" ","_"))
        if m in skip or (price>float(lim) if lim else e>cap) or x.get("inet_down_cost",1)>.03: continue
        if m not in best or e<best[m][0]: best[m]=(e,x,bid)
for e,x,bid in sorted(best.values(),key=lambda t:t[0]): print(x["gpu_name"].replace(" ","_"),x["id"],x["machine_id"],bid if bid else "-")' $CAP ${BID_MARGIN:-0.05} $(for d in $(active); do cat $d/machine; done | sort -u); }
launch() { # label gpu offer machine bid; runs in the background
  l=$1; mkdir -p $l; echo $2 > $l/gpu; echo $4 > $l/machine; date +%s > $l/t0; touch $l/launching
  b=; [ "${5:--}" != - ] && { b=$5; echo $b > $l/bid; }
  r=$(BID_PRICE=$b timeout 1200 bash $HERE/vast-launch.sh $l $IMG $3 2>&1 | tail -1); log "$l $2 machine $4${b:+ bid \$$b}: $r"
  [ -f $l/id ] || { [ -n "$b" ] || echo $4 >> machines.bad; touch $l/dead; }   # a failed bid is no fault of the machine
  rm -f $l/launching
}
rm -f m*/launching   # launches of an earlier manager are over
# state; glas processes; seconds since the client last started a work unit (or since boot.out);
# whether the build ended; the work unit lines of glas-wu
PROBE='cd /workspace; echo "state [$(cat state 2>/dev/null)]"; echo "procs $(pgrep -xc glas-gpu)"; w=$(grep "Subprocess has PID" out/client.log 2>/dev/null | tail -1 | sed "s/.*\[\(.*\)\].*/\1/"); if [ -n "$w" ]; then t=$(date -d "$w" +%s); else t=$(stat -c %Y boot.out 2>/dev/null || date +%s); fi; echo "age $(( $(date +%s) - t ))"; echo "built $(grep -c ^built out/boot.log 2>/dev/null)"; echo "gotwu $(grep -c "/workunit to " out/client.log 2>/dev/null)"; echo "fetching $(tail -1 out/client.log 2>/dev/null | grep -c "Downloading .*/file/")"; cat out/wu.log 2>/dev/null'
export PROBE
last_health=0; nidle=0
while :; do
  unset $(compgen -v PRICE_); set -a; . ./fleet.conf; set +a
  credit=$($V show user --raw 2>/dev/null | python3 -c 'import json,sys;print(int(json.load(sys.stdin)["credit"]))' 2>/dev/null)
  if curl -sk -m 20 $SERVER/ | grep -q Hello; then [ -n "$srv_down_since" ] && log "server answers again"; srv_down_since=
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
      if [ ! -f $l/addr ]; then [ $age -gt 1800 ] && { bad $l; kill_host $l "never came up"; }; continue; fi
      s=$(cat $l/probe 2>/dev/null)
      if [ -z "$s" ]; then m=$(( $(cat $l/miss 2>/dev/null || echo 0) + 1 )); echo $m > $l/miss
        [ "$a" != running ] && [ $m -ge 2 ] || [ $m -ge 4 ] && { bad $l; kill_host $l "unreachable, status $a"; }; continue; fi
      echo 0 > $l/miss
      if echo "$s" | grep -q '^state \[done rc=0\]'; then
        timeout 90 "$HERE/vast.sh" ssh $l "$RESTART" > /dev/null 2>&1; log "$l client ended, restarted"; continue; fi
      if ! echo "$s" | grep -q '^state \[\]'; then bad $l; kill_host $l "$(echo "$s" | head -1)"; continue; fi
      wus=$(echo "$s" | awk 'NF==6' )
      # the last two work units failed: glas-wu should have stopped the client already
      if [ "$(echo "$wus" | awk 'NF==6 {r[++n]=$4} END{print (n>=2 && r[n]!=0 && r[n-1]!=0) ? "y" : "n"}')" = y ]; then
        bad $l; kill_host $l "work units failing: $(echo "$wus" | tail -2 | awk '{print $4}' | paste -sd,)"; continue; fi
      spq=$(echo "$wus" | awk 'NF==6 && $4==0 && $5>0 {t+=$3; sq+=$5; n++} END{if(n>=2) printf "%.4f", t/sq}')
      ref=${SPQ[$(cat $l/gpu)]:-}
      if [ -n "$spq" ] && [ -n "$ref" ] && python3 -c "import sys;sys.exit(0 if $spq > $ref/0.75 else 1)"; then
        bad $l; kill_host $l "slow: $spq s/sq, $(cat $l/gpu)"; continue; fi
      procs=$(echo "$s" | awk '/^procs/{print $2}'); since=$(echo "$s" | awk '/^age/{print $2}')
      # a client fetching a file (the factor base, when MAKEFB is unset or failed) is not idle
      if [ "$procs" = 0 ] && [ "${since:-0}" -gt 2700 ] && [ -z "$srv_down_since" ] && ! echo "$s" | grep -q '^fetching 1'; then
        if [ -z "$wus" ]; then
          # built, server up, and never a work unit: the server unreachable from there, or the client broken
          echo "$s" | grep -q '^built [1-9]' && echo "$s" | grep -q '^gotwu 0' && { bad $l; kill_host $l "never got a work unit"; }
        else idle+=($l); fi
      fi
    done
    for l in "${idle[@]}"; do kill_host $l "idle"; done
    nidle=${#idle[@]}
    log "health: live $(live | wc -l) launching $(launching) idle $nidle credit $credit rate \$$(echo "$st" | awk '{s+=$3} END{print s}')/h (all instances)"
  fi
  # no new hosts once hosts sit idle: the sieving is in its tail or done
  if [ $nidle -eq 0 ] && [ -z "$srv_down_since" ]; then
    want=$(( MAXHOSTS - $(active | wc -l) )); room=$(( 8 - $(launching) )); [ $want -gt $room ] && want=$room
    if [ $want -gt 0 ]; then
      offers | head -$want > offers.now
      while read g o m b; do c=$(( $(cat counter) + 1 )); echo $c > counter; launch $(printf 'm%03d' $c) $g $o $m $b & sleep 2; done < offers.now
    fi
  fi
  sleep 60
done
