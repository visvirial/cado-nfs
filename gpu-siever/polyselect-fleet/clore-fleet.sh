#!/bin/bash
# Keep a Clore.ai fleet of single-GPU clients on the polynomial search, as vast-fleet.sh does on
# Vast.ai. Started by "clore.sh manage"; state and events.log in $FLEET_DIR/clore, one
# directory s<server> per rented server. It takes over orders made by hand.
#
# Every minute it rents each free server under CLORE_CAP dollars per hour per ad/s, with
# reliability at least CLORE_MINREL, up to CLORE_MAXHOSTS. Every 10 minutes it probes the
# hosts over SSH and cancels, and never rents again, those that do not boot or answer within
# 30 minutes, whose client failed (one that ended when the server stopped serving work units is
# restarted), whose last two work units failed, that run below 75% of
# their GPU's speed, or that never got a work unit (the server unreachable from there). Idle
# hosts beyond CLORE_MINKEEP are cancelled, and none rented while any is idle. Every order is
# cancelled when STOP or CLORE_STOP is 1, the balance falls below CLORE_MINBAL dollars, or
# SERVER has not answered for 2 h. New orders need a balance of CLORE_MINBAL plus two hours
# of the fleet's cost.
D=$FLEET_DIR/clore; mkdir -p $D; cd $D; E=$D/events.log; C="bash $HERE/clore.sh"
log() { echo "$(date '+%F %T') $*" >> $E; }
declare -A RATE=([RTX_3090]=0.496 [RTX_5080]=0.522 [RTX_4080_SUPER]=0.415 [RTX_4090]=0.57 [RTX_5090]=0.96)
touch servers.bad; srv_down_since=; last_health=0; nidle=0
live() { for d in s*/; do d=${d%/}; [ -d $d ] && [ ! -f $d/dead ] && echo $d; done; }
cancel_host() { d=$1; shift; [ -f $d/order ] && $C cancel $(cat $d/order) > /dev/null 2>&1; touch $d/dead; log "$d $(cat $d/gpu 2>/dev/null) cancelled: $*"; }
bad() { echo ${1#s} >> servers.bad; }
PROBE='cd /workspace 2>/dev/null || { echo nows; exit 0; }; echo "state [$(cat state 2>/dev/null)]"; echo "procs $(pgrep -xc gps1) $(pgrep -xc polyselect_ropt)"; w=$(grep "Subprocess has PID" out/client.log 2>/dev/null | tail -1 | sed "s/.*\[\(.*\)\].*/\1/"); if [ -n "$w" ]; then t=$(date -d "$w" +%s); else t=$(stat -c %Y out/boot.log 2>/dev/null || date +%s); fi; echo "age $(( $(date +%s) - t ))"; echo "built $(grep -c ^built out/boot.log 2>/dev/null)"; cat out/wu.log 2>/dev/null'
export PROBE KEY
while :; do
  CLORE_CAP=0.5 CLORE_MAXHOSTS=60 CLORE_MINREL=0.98 CLORE_MINBAL=2 CLORE_MINKEEP=0 CLORE_STOP=0
  set -a; . $FLEET_DIR/fleet.conf; set +a
  bal=$($C balance 2>/dev/null)
  if curl -sk -m 20 $SERVER/ | grep -q Hello; then srv_down_since=
  else [ -z "$srv_down_since" ] && log "server unreachable"; srv_down_since=${srv_down_since:-$(date +%s)}; fi
  low=$(python3 -c "import sys; print(1 if '$bal' and float('$bal') < $CLORE_MINBAL else 0)" 2>/dev/null)
  if [ "$STOP" = 1 ] || [ "$CLORE_STOP" = 1 ] || [ "$low" = 1 ] || { [ -n "$srv_down_since" ] && [ $(( $(date +%s) - srv_down_since )) -gt 7200 ]; }; then
    for o in $($C tsv | awk '{print $1}'); do $C cancel $o > /dev/null; done
    for d in $(live); do touch $d/dead; done
    log "fleet stopped (STOP=$STOP CLORE_STOP=$CLORE_STOP balance=$bal server_down_since=$srv_down_since)"; exit 0; fi
  if [ $(( $(date +%s) - last_health )) -ge 600 ] && tsv=$($C tsv 2>/dev/null); then
    last_health=$(date +%s)
    # take over the orders: a new order of a server gets a fresh directory
    while read o s g h p pr; do d=s$s
      if [ ! -d $d ] || { [ -f $d/dead ] && [ "$(cat $d/order 2>/dev/null)" != $o ]; }; then rm -rf $d; mkdir $d; date +%s > $d/t0; fi
      echo $o > $d/order; echo $g > $d/gpu; echo $pr > $d/price; [ "$p" != - ] && echo "$h $p" > $d/addr
    done <<< "$tsv"
    for d in $(live); do [ -f $d/addr ] && echo $d; done | xargs -r -P 32 -I{} bash -c 'read h p < {}/addr
      timeout 60 ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p $p root@$h "$PROBE" > {}/probe.new 2>/dev/null; mv {}/probe.new {}/probe'
    idle=()
    for d in $(live); do
      age=$(( $(date +%s) - $(cat $d/t0) ))
      if ! echo "$tsv" | awk -v s=${d#s} '$2==s{f=1} END{exit !f}'; then
        [ $age -gt 900 ] && { touch $d/dead; log "$d order is gone"; }; continue; fi
      s=$(cat $d/probe 2>/dev/null); rm -f $d/probe
      if [ -z "$s" ] || [ "$s" = nows ]; then
        [ $age -lt 1800 ] && continue
        m=$(( $(cat $d/miss 2>/dev/null || echo 0) + 1 )); echo $m > $d/miss
        [ "$s" = nows ] || [ $m -ge 3 ] && { bad $d; cancel_host $d "${s:-unreachable} ${age}s after renting"; }; continue; fi
      echo 0 > $d/miss
      if echo "$s" | grep -q '^state \[done rc=0\]'; then read h p < $d/addr
        timeout 90 ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p $p root@$h "$RESTART" > /dev/null 2>&1
        log "$d client ended, restarted"; continue; fi
      if ! echo "$s" | grep -q '^state \[\]'; then bad $d; cancel_host $d "$(echo "$s" | grep '^state')"; continue; fi
      if [ "$(echo "$s" | awk 'NF==5 {r[++n]=$5} END{print (n>=2 && r[n]!=0 && r[n-1]!=0) ? "y" : "n"}')" = y ]; then
        bad $d; cancel_host $d "work units failing"; continue; fi
      rate=$(echo "$s" | awk 'NF==5 && $5==0 {ad+=($2-$1)/$3; t+=$4; n++} END{if(n>=2 && t>0) printf "%.3f", ad/t}')
      if [ -n "$rate" ] && python3 -c "import sys;sys.exit(0 if $rate < 0.75*${RATE[$(cat $d/gpu)]:-0} else 1)"; then
        bad $d; cancel_host $d "slow: $rate ad/s"; continue; fi
      nwu=$(echo "$s" | awk 'NF==5' | wc -l); a=$(echo "$s" | awk '/^age/{print $2}')
      if [ "$(echo "$s" | awk '/^built/{print $2}')" = 0 ]; then
        [ $age -gt 1800 ] && { bad $d; cancel_host $d "not built ${age}s after renting"; }; continue; fi
      if echo "$s" | grep -q '^procs 0 0' && [ "${a:-0}" -gt 2700 ]; then
        if [ $nwu = 0 ]; then bad $d; cancel_host $d "no work unit ${a}s after the build"; else idle+=($d); fi; fi
    done
    n=$(live | wc -l); keep=$(( n - ${#idle[@]} ))
    for d in "${idle[@]}"; do [ $keep -ge $CLORE_MINKEEP ] && { cancel_host $d idle; continue; }; keep=$((keep+1)); done
    nidle=${#idle[@]}
    cost=$(for d in $(live); do cat $d/price 2>/dev/null; done | awk '{s+=$1} END{printf "%.2f", s}')
    log "health: live $(live | wc -l) idle $nidle balance $bal rate \$$cost/h"
  fi
  if [ $nidle -eq 0 ] && [ -z "$srv_down_since" ] && [ -n "$bal" ]; then
    cost=$(for d in $(live); do cat $d/price 2>/dev/null; done | awk '{s+=$1} END{print s+0}')
    want=$(( CLORE_MAXHOSTS - $(live | wc -l) )); [ $want -gt 10 ] && want=10
    if [ $want -gt 0 ] && python3 -c "import sys; sys.exit(0 if $bal > $CLORE_MINBAL + 2*($cost + 0.4) else 1)"; then
      busy=$( (cat servers.bad; for d in $(live); do echo ${d#s}; done) | sort -u)
      $C candidates $CLORE_CAP $CLORE_MINREL 2>/dev/null | awk -v b="$(echo $busy)" 'BEGIN{split(b,x," "); for(i in x) skip[x[i]]=1} !skip[$1]' | head -$want |
      while read s g; do r=$($C rent $s 2>&1 | tail -1)
        if echo "$r" | grep -q '"code":0'; then d=s$s; rm -rf $d; mkdir $d; date +%s > $d/t0; echo $g > $d/gpu; log "$d $g rented"
        else log "s$s $g rent failed: $(echo "$r" | cut -c1-120)"; echo $s >> servers.bad; fi
      done
    fi
  fi
  sleep 60
done
