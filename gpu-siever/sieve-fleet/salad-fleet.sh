#!/bin/bash
# Fill the SaladCloud replica quota with sieving clients, cheapest priority first. Started by
# "salad.sh manage"; logs to $FLEET_DIR/salad/events.log.
#
# One container group per priority of SALAD_TIERS (default "batch low"), SALAD_GROUP-<priority>
# (c270-sv-batch, c270-sv-low, ...), each with the GPU classes whose price
# at that priority is within CAP ("salad.sh fit"). The quota counts the replicas a group asks
# for, not those it gets, so every 10 minutes each group but the last asks for the replicas it
# has (creating or running) plus SALAD_HEADROOM (5), to keep taking free machines at its
# priority, and the last asks for what is left of SALAD_REPLICAS (default: the quota). A group
# that gets more machines takes them from the later ones; lowering a group's replicas comes
# before raising another's, so the total stays within the quota. SALAD_FIXED ("medium=20 ...")
# gives a tier a fixed number of replicas instead, taken from the quota first. The instances of
# each group are logged to $FLEET_DIR/salad/members.log, to tell the tiers apart in the server
# log (client vast-svsalad-<first 18 characters of the machine id>). Every group is stopped when
# STOP or SALAD_STOP is 1, or SERVER has not answered for 2 h. Every 10 minutes it also runs
# bad-clients.sh --apply, which replaces the Salad and Nosana clients that keep failing.
D=$FLEET_DIR/salad; mkdir -p $D; cd $D; E=$D/events.log; C="bash $HERE/salad.sh"
echo $$ > $FLEET_DIR/salad-fleet.sh.pid   # for running() in env.sh
log() { echo "$(date '+%F %T') $*" >> $E; }
srv_down_since=
while :; do
  SALAD_TIERS="batch low" SALAD_GROUP=c270-sv SALAD_HEADROOM=5 SALAD_REPLICAS= SALAD_FIXED= SALAD_STOP=0 STOP=0
  set -a; . $FLEET_DIR/fleet.conf; set +a
  tiers=($SALAD_TIERS); n=${#tiers[@]}
  name() { echo $SALAD_GROUP-${tiers[$1]}; }
  if curl -sk -m 20 $SERVER/ | grep -q Hello; then srv_down_since=
  else [ -z "$srv_down_since" ] && log "server unreachable"; srv_down_since=${srv_down_since:-$(date +%s)}; fi
  if [ "$STOP" = 1 ] || [ "$SALAD_STOP" = 1 ] || { [ -n "$srv_down_since" ] && [ $(( $(date +%s) - srv_down_since )) -gt 7200 ]; }; then
    for i in $(seq 0 $((n - 1))); do $C counts $(name $i) > /dev/null 2>&1 && $C replicas $(name $i) 0 > /dev/null 2>&1; $C delete $(name $i) > /dev/null 2>&1; done
    log "groups stopped and deleted (STOP=$STOP SALAD_STOP=$SALAD_STOP server_down_since=$srv_down_since)"; exit 0; fi
  q=${SALAD_REPLICAS:-$($C quota 2>/dev/null)}
  if [ -z "$q" ]; then log "no quota"; sleep 600; continue; fi
  # the targets, in tier order: what each has plus the headroom, the last what is left
  declare -A have=() asks=() target=() state=() fixed=(); left=$q
  for x in $SALAD_FIXED; do fixed[${x%=*}]=${x#*=}; left=$(( left - ${x#*=} )); done
  [ $left -ge 0 ] || { log "SALAD_FIXED over the quota $q"; sleep 600; continue; }
  last=-1; for i in $(seq 0 $((n - 1))); do [ -z "${fixed[${tiers[$i]}]:-}" ] && last=$i; done
  for i in $(seq 0 $((n - 1))); do g=$(name $i)
    if c=$($C counts $g 2>/dev/null); then read r a st <<< "$c"; asks[$i]=$r; have[$i]=$a; state[$i]=$st; else asks[$i]=-; have[$i]=0; fi
    if [ -n "${fixed[${tiers[$i]}]:-}" ]; then target[$i]=${fixed[${tiers[$i]}]}; continue; fi
    if [ $i -lt $last ]; then t=$(( ${have[$i]} + SALAD_HEADROOM )); [ $t -gt $left ] && t=$left; else t=$left; fi
    target[$i]=$t; left=$(( left - t ))
  done
  # lower first, then raise or create, so the total asked never exceeds the quota
  for pass in lower raise; do
    for i in $(seq 0 $((n - 1))); do g=$(name $i); r=${asks[$i]}; t=${target[$i]}
      if [ "$r" = - ]; then
        [ $pass = raise ] && [ $t -gt 0 ] || continue
        mapfile -t cls < <($C fit ${tiers[$i]} 2>/dev/null)
        [ ${#cls[@]} -gt 0 ] || { log "$g: no GPU class within CAP at ${tiers[$i]}"; continue; }
        $C create $g $t ${tiers[$i]} "${cls[@]}" > /dev/null 2>&1 && $C start $g > /dev/null 2>&1
        log "$g created: $t ${tiers[$i]} replicas of ${cls[*]}"; continue; fi
      if { [ $pass = lower ] && [ $t -lt $r ]; } || { [ $pass = raise ] && [ $t -gt $r ]; }; then
        if $C replicas $g $t > /dev/null 2>&1; then log "$g replicas $r -> $t (has ${have[$i]})"; else log "$g replicas $r -> $t failed"; fi
      fi
      # a group made just after another lowered its replicas, or over the quota, stays stopped
      if [ $pass = raise ] && [ $t -gt 0 ] && [ "${state[$i]}" = stopped ]; then
        $C start $g > /dev/null 2>&1 && log "$g started" || log "$g start failed"; fi
    done
  done
  for i in $(seq 0 $((n - 1))); do [ "${asks[$i]}" = - ] || $C status $(name $i) 2>/dev/null |
    awk -v t="$(date +%s)" -v p=${tiers[$i]} '{print t, p, $1, $2}' >> $D/members.log; done
  # Salad and Nosana clients that keep failing work units: replaced (see bad-clients.sh)
  timeout 300 bash $HERE/bad-clients.sh --apply > /dev/null 2>&1 || true
  log "quota $q: $(for i in $(seq 0 $((n - 1))); do printf '%s %s/%s  ' ${tiers[$i]} ${have[$i]} ${target[$i]}; done)"
  sleep 600
done
