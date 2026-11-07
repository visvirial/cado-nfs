#!/bin/bash
# Keep a Clore.ai fleet of single-GPU clients on the sieving, as vast-fleet.sh does on Vast.ai.
# Started by "clore.sh manage"; state and events.log in $FLEET_DIR/clore, one directory
# s<server> per rented server. It takes over orders made by hand.
#
# Every minute it rents each free server under CAP dollars per million special-q (the cap of
# every provider, at the seconds per special-q of SPQ_TABLE), with reliability at least CLORE_MINREL, up to
# CLORE_MAXHOSTS. Every 10 minutes it probes the hosts over SSH and cancels, and never rents
# again, those that do not boot or answer within 30 minutes, whose client failed (one that
# ended when the server stopped serving work units is restarted), whose last two work units
# failed, that run slower than 1/0.75 of their GPU's seconds per special-q, or that never got
# a work unit (the server unreachable from there). Hosts idle for 45 minutes while the server
# answers are cancelled, and none rented while any is idle. Every order is cancelled when STOP
# or CLORE_STOP is 1, the balance falls below CLORE_MINBAL dollars, or SERVER has not answered
# for 2 h. New orders need a balance of CLORE_MINBAL plus two hours of the fleet's cost.
D=$FLEET_DIR/clore; mkdir -p $D; cd $D; E=$D/events.log; C="bash $HERE/clore.sh"
echo $$ > $FLEET_DIR/clore-fleet.sh.pid   # for running() in env.sh
log() { echo "$(date '+%F %T') $*" >> $E; }
declare -A SPQ; for x in $SPQ_TABLE; do SPQ[${x%=*}]=${x#*=}; done
touch servers.bad; srv_down_since=; last_health=0; nidle=0
live() { for d in s*/; do d=${d%/}; [ -d $d ] && [ ! -f $d/dead ] && echo $d; done; }
# a host whose order did not end stays live, and is tried again on the next round
cancel_host() { d=$1; shift
  if [ -f $d/order ] && ! $C cancel $(cat $d/order) > /dev/null 2>&1; then log "$d $(cat $d/gpu 2>/dev/null) cancel failed ($*)"; return; fi
  touch $d/dead; log "$d $(cat $d/gpu 2>/dev/null) cancelled: $*"; }
bad() { echo ${1#s} >> servers.bad; }
PROBE='cd /workspace 2>/dev/null || { echo nows; exit 0; }; echo "state [$(cat state 2>/dev/null)]"; echo "procs $(pgrep -xc glas-gpu)"; w=$(grep "Subprocess has PID" out/client.log 2>/dev/null | tail -1 | sed "s/.*\[\(.*\)\].*/\1/"); if [ -n "$w" ]; then t=$(date -d "$w" +%s); else t=$(stat -c %Y out/boot.log 2>/dev/null || date +%s); fi; echo "age $(( $(date +%s) - t ))"; echo "built $(grep -c ^built out/boot.log 2>/dev/null)"; echo "gotwu $(cat out/client.log 2>/dev/null | grep -c "/workunit to ")"; echo "fetching $(tail -1 out/client.log 2>/dev/null | grep -c "Downloading .*/file/")"; echo "first $(w=$(grep "Subprocess has PID" out/client.log 2>/dev/null | head -1 | sed "s/.*\[\(.*\)\].*/\1/"); [ -n "$w" ] && echo $(( $(date +%s) - $(date -d "$w" +%s) )) || echo -1)"; echo "upok $(grep -c "Upload of .* succeeded" out/client.log 2>/dev/null)"; echo "upfail $(cat out/client.log 2>/dev/null | awk "/Upload of .* succeeded/ {n = 0} /Upload failed/ {n++} END {print n + 0}")"; echo "cmdfail $(cat out/client.log 2>/dev/null | awk "/because the command failed/ {p = 1} /Upload of .* succeeded/ {r[++n] = p; p = 0} END {c = 0; for (i = n; i > n - 3 && i > 0; i--) c += r[i]; print (n >= 3) ? c : 0}")"; cat out/wu.log 2>/dev/null; [ -s state ] && tail -n 15 out/setup.log out/boot.log out/client.log 2>/dev/null | sed "s/^/log| /"'
export PROBE KEY
while :; do
  CAP=40 CLORE_MAXHOSTS=10 CLORE_MINREL=0.98 CLORE_MINBAL=2 CLORE_STOP=0
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
    while read o s g h p pr; do [ -n "$s" ] || continue; d=s$s
      if [ ! -d $d ] || { [ -f $d/dead ] && [ "$(cat $d/order 2>/dev/null)" != $o ]; }; then rm -rf $d; mkdir $d; date +%s > $d/t0; fi
      echo $o > $d/order; echo $g > $d/gpu; echo $pr > $d/price; [ "$p" != - ] && echo "$h $p" > $d/addr
    done <<< "$tsv"
    # Some Clore SSH gateways (n1.de.clorecloud.net, 2026-11-06) drop most connection attempts to a
    # forwarded port: a connection there took up to 90 s of SYN retries, so the probes wait long
    for d in $(live); do [ -f $d/addr ] && echo $d; done | xargs -r -P 32 -I{} bash -c 'read h p < {}/addr
      timeout 240 ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=60 -o ConnectionAttempts=3 -p $p root@$h "$PROBE" > {}/probe.new 2>/dev/null; mv {}/probe.new {}/probe'
    idle=()
    for d in $(live); do
      age=$(( $(date +%s) - $(cat $d/t0) ))
      if ! echo "$tsv" | awk -v s=${d#s} '$2==s{f=1} END{exit !f}'; then
        [ $age -gt 900 ] && { touch $d/dead; log "$d order is gone"; }; continue; fi
      s=$(cat $d/probe 2>/dev/null); [ -f $d/probe ] && mv $d/probe $d/probe.last
      if [ -z "$s" ] || [ "$s" = nows ]; then
        [ $age -lt 1800 ] && continue
        m=$(( $(cat $d/miss 2>/dev/null || echo 0) + 1 )); echo $m > $d/miss
        [ "$s" = nows ] || [ $m -ge 3 ] && { bad $d; cancel_host $d "${s:-unreachable} ${age}s after renting"; }; continue; fi
      echo 0 > $d/miss
      if echo "$s" | grep -q '^state \[done rc=0\]'; then read h p < $d/addr
        timeout 240 ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=60 -o ConnectionAttempts=3 -p $p root@$h "$RESTART" > /dev/null 2>&1
        log "$d client ended, restarted"; continue; fi
      if ! echo "$s" | grep -q '^state \[\]'; then bad $d; cancel_host $d "$(echo "$s" | grep '^state'), log tails in clore/$d/probe.last"; continue; fi
      # uploads failing since the last one that went through: the host's network does not reach
      # the server (a Clore host in Germany got 3 KB/s to it, 2026-11-06)
      uf=$(echo "$s" | awk '/^upfail/{print $2}')
      # three in a row while the server answers: one host held work units for hours at 10
      if [ "${uf:-0}" -ge 3 ] && [ -z "$srv_down_since" ]; then bad $d; cancel_host $d "uploads failing ($uf in a row)"; continue; fi
      # the last 3 work units failed in the client (glas exits at once, its wu.log line never
      # written): such a host takes a work unit every few seconds and fails it, and the server
      # resubmits a failed one only once (maxwuerror 2), so the ranges are lost (2026-11-07: m202
      # failed 16842 work units in a day, 10816 ranges left unsieved)
      cf=$(echo "$s" | awk '/^cmdfail/{print $2}')
      if [ "${cf:-0}" -ge 3 ]; then bad $d; cancel_host $d "work units failing in the client ($cf of the last 3)"; continue; fi
      wus=$(echo "$s" | awk 'NF==6 && !/^log\|/')
      if [ "$(echo "$wus" | awk 'NF==6 {r[++n]=$4} END{print (n>=2 && r[n]!=0 && r[n-1]!=0) ? "y" : "n"}')" = y ]; then
        bad $d; cancel_host $d "work units failing"; continue; fi
      spq=$(echo "$wus" | awk 'NF==6 && $4==0 && $5>0 {t+=$3; sq+=$5; n++} END{if(n>=2) printf "%.4f", t/sq}')
      ref=${SPQ[$(cat $d/gpu)]:-}
      if [ -n "$spq" ] && [ -n "$ref" ] && python3 -c "import sys;sys.exit(0 if $spq > $ref/0.75 else 1)"; then
        bad $d; cancel_host $d "slow: $spq s/sq"; continue; fi
      # the cost of the special-q this host gives, over at least 4 work units, as in vast-fleet.sh
      spq4=$(echo "$wus" | awk 'NF==6 && $4==0 && $5>0 {t+=$3; sq+=$5; n++} END{if(n>=4) printf "%.4f", t/sq}')
      dph=$(cat $d/price 2>/dev/null)
      if [ -n "$spq4" ] && [ -n "$dph" ] && python3 -c "import sys;sys.exit(0 if $dph * $spq4 / 3600e-6 > $CAP * (1 + ${COST_MARGIN:-0.05}) else 1)"; then
        bad $d; cancel_host $d "costly: $(python3 -c "print(round($dph * $spq4 / 3600e-6, 1))") \$/Msq ($spq4 s/sq at \$$dph/h)"; continue; fi
      # work units taken but no result in 2.5 times a work unit of the GPU (7000 special-q): the
      # results do not reach the server, or the host is far slower than its GPU (2026-11-06: 11
      # Clore hosts held 4 work units each for 1.4 to 4.2 h without a result)
      fs=$(echo "$s" | awk '/^first/{print $2}'); ok=$(echo "$s" | awk '/^upok/{print $2}')
      if [ -n "$ref" ] && [ "${ok:-1}" = 0 ] && [ -z "$srv_down_since" ] && [ "${fs:--1}" -ge 0 ] && \
          python3 -c "import sys;sys.exit(0 if $fs > 2.5 * 7000 * $ref else 1)"; then
        bad $d; cancel_host $d "no result ${fs}s after the first work unit"; continue; fi
      a=$(echo "$s" | awk '/^age/{print $2}')
      if [ "$(echo "$s" | awk '/^built/{print $2}')" = 0 ]; then
        [ $age -gt 1800 ] && { bad $d; cancel_host $d "not built ${age}s after renting"; }; continue; fi
      # a client fetching a file (the factor base, when MAKEFB is unset or failed) is not idle
      if echo "$s" | grep -q '^procs 0' && [ "${a:-0}" -gt 2700 ] && [ -z "$srv_down_since" ] && ! echo "$s" | grep -q '^fetching 1'; then
        if [ -z "$wus" ]; then echo "$s" | grep -q '^gotwu 0' && { bad $d; cancel_host $d "no work unit ${a}s after the build"; }
        else idle+=($d); fi; fi
    done
    for d in "${idle[@]}"; do cancel_host $d idle; done
    nidle=${#idle[@]}
    cost=$(for d in $(live); do cat $d/price 2>/dev/null; done | awk '{s+=$1} END{printf "%.2f", s}')
    log "health: live $(live | wc -l) idle $nidle balance $bal rate \$$cost/h"
  fi
  if [ $nidle -eq 0 ] && [ -z "$srv_down_since" ] && [ -n "$bal" ]; then
    cost=$(for d in $(live); do cat $d/price 2>/dev/null; done | awk '{s+=$1} END{print s+0}')
    want=$(( CLORE_MAXHOSTS - $(live | wc -l) )); [ $want -gt 10 ] && want=10
    if [ $want -gt 0 ] && python3 -c "import sys; sys.exit(0 if $bal > $CLORE_MINBAL + 2*($cost + 0.4) else 1)"; then
      busy=$( (cat servers.bad; for d in $(live); do echo ${d#s}; done) | sort -u)
      $C candidates $CAP $CLORE_MINREL 2>/dev/null | awk -v b="$(echo $busy)" 'BEGIN{split(b,x," "); for(i in x) skip[x[i]]=1} !skip[$1]' | head -$want |
      while read s g; do r=$($C rent $s 2>&1 | tail -1)
        if echo "$r" | grep -q '"code":0'; then d=s$s; rm -rf $d; mkdir $d; date +%s > $d/t0; echo $g > $d/gpu; log "$d $g rented"
        else log "s$s $g rent failed: $(echo "$r" | cut -c1-120)"; echo $s >> servers.bad; fi
      done
    fi
  fi
  sleep 60
done
