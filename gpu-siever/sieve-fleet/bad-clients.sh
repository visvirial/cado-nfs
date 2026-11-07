#!/bin/bash
# Find the sieving clients that keep failing work units, from the server's log, and replace
# the SaladCloud and Nosana ones (the Vast and Clore managers probe their hosts and catch
# these themselves, see "cmdfail" in vast-fleet.sh).
#
# A host whose glas exits at once takes a work unit every few seconds and fails it; the
# server resubmits a failed work unit only once (maxwuerror 2), so a range handed twice to
# such hosts is never sieved (2026-11-07: Vast m202 failed 16842 work units in one day, and
# 10816 ranges had to be resubmitted by hand).
#
#   bad-clients.sh            list the clients with BAD_FAILS (5) or more failed work units
#                             in the last BAD_MINUTES (60) of the server log; changes nothing
#   bad-clients.sh --apply    also reallocate such a Salad instance (it restarts on another
#                             machine) and replace such a Nosana deployment (stop it, create
#                             one of the same model); a client handled within the last 3 h is
#                             left alone. Logged to $FLEET_DIR/bad-clients.log.
#
# The salad manager (salad-fleet.sh) runs it with --apply every 10 minutes.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")"); . "$HERE/env.sh"; set -a; . "$FLEET_DIR/fleet.conf"; set +a
: "${SERVER_SSH:?}" "${SERVER_WORKDIR:?}"
apply=0; [ "${1:-}" = --apply ] && apply=1
BAD_FAILS=${BAD_FAILS:-5}; BAD_MINUTES=${BAD_MINUTES:-60}
L=$FLEET_DIR/bad-clients.log; DONE=$FLEET_DIR/bad-clients.done; touch "$DONE"
log() { echo "$(date '+%F %T') $*" | tee -a "$L"; }
# failures per client in the last BAD_MINUTES of the log (its timestamps are the server's
# local time); the tail keeps this cheap on the server's 4 cores
bad=$(ssh -o BatchMode=yes -o ConnectTimeout=30 "$SERVER_SSH" "cd '$SERVER_WORKDIR' && tail -n 300000 c270.log |
  awk -v c=\"\$(date -d '-$BAD_MINUTES min' '+%F %T')\" '/failed with exit code/ && (\$2 \" \" substr(\$3, 1, 8)) >= c {
    for (i = 1; i <= NF; i++) if (\$i == \"on\") n[\$(i + 1)]++ } END {for (k in n) print n[k], k}'" | awk -v t=$BAD_FAILS '$1 >= t' | sort -rn)
[ -n "$bad" ] || exit 0
now=$(date +%s)
recent() { awk -v c="$1" -v t=$((now - 10800)) '$2 == c && $1 > t {f = 1} END {exit !f}' "$DONE"; }
while read -r n client; do
  id=${client#vast-sv}; id=${id%.gpu0}
  case "$id" in
    salad-clore-*|m[0-9]*) echo "$client: $n failures (left to the Clore/Vast manager)"; continue ;;
  esac
  if [ $apply = 0 ]; then echo "$client: $n failures"; continue; fi
  if recent "$client"; then echo "$client: $n failures, handled less than 3 h ago"; continue; fi
  case "$id" in
    salad-nosana-*)
      dep=rsa270-sv-${id#salad-nosana-}; model=${id#salad-nosana-}; model=${model%-*}
      did=$("$HERE/nosana.sh" list 2>/dev/null | awk -v d="$dep" '$3 == d && $2 == "RUNNING" {print $1}')
      if [ -z "$did" ]; then log "$client: $n failures, no running deployment $dep"; continue; fi
      "$HERE/nosana.sh" stop "$did" > /dev/null 2>&1 || true
      new=$("$HERE/nosana.sh" create "$model" 1 2>/dev/null | tail -1)
      log "$client: $n failed work units in $BAD_MINUTES min: deployment $dep stopped, new $model deployment $new" ;;
    salad-*)
      m=${id#salad-}
      r=$("$HERE/salad.sh" reallocate "$m" 2>&1) || true
      log "$client: $n failed work units in $BAD_MINUTES min: $r" ;;
    *) log "$client: $n failed work units in $BAD_MINUTES min, unknown provider"; continue ;;
  esac
  echo "$now $client" >> "$DONE"
done <<< "$bad"
