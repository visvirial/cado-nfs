#!/bin/bash
# Run sieving clients on Vast.ai.
#
#   vast.sh manage             start the Vast fleet manager, vast-fleet.sh (again, after a reboot)
#   vast.sh status             whether it runs, its last log lines, and each host's work units,
#                              seconds per special-q and relations
#   vast.sh ssh HOST [CMD]     SSH into a host (m001, ...)
#
# The first "manage" creates FLEET_DIR with fleet.conf to fill in, an SSH key registered with
# Vast.ai, and src.tgz, the source the hosts build: this repository's HEAD. Remove src.tgz to
# send a newer HEAD to the hosts rented from then on.
set -Eeuo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
case ${1:-} in
manage)
  [ -n "$V" ] || { echo "vastai not found: pip install vastai, or set V" >&2; exit 1; }
  mkdir -p "$FLEET_DIR"; chmod 700 "$FLEET_DIR"; cd "$FLEET_DIR"
  if [ ! -f fleet.conf ]; then cp "$HERE/fleet.conf.example" fleet.conf; echo "fill in $FLEET_DIR/fleet.conf and run again"; exit 1; fi
  . ./fleet.conf
  case "$CERTSHA1" in 0000*) echo "set SERVER and CERTSHA1 in $FLEET_DIR/fleet.conf" >&2; exit 1;; esac
  [ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C rsa270-sieve-fleet -f "$KEY"
  pub=$(awk '{print $2}' "$KEY.pub")
  "$V" show ssh-keys --raw | python3 -c 'import json,sys
sys.exit(0 if any(sys.argv[1] in (k.get("public_key") or k.get("ssh_key") or "") for k in json.load(sys.stdin)) else 1)' "$pub" \
    || "$V" create ssh-key "$(cat "$KEY.pub")" > /dev/null
  [ -f src.tgz ] || git -C "$(git -C "$HERE" rev-parse --show-toplevel)" archive --format=tar.gz -o "$FLEET_DIR/src.tgz" HEAD
  if running vast-fleet.sh; then echo "already running"; exit 0; fi
  setsid nohup bash "$HERE/vast-fleet.sh" >> fleet.out 2>&1 < /dev/null &
  echo "started; log in $FLEET_DIR/events.log";;
status)
  cd "$FLEET_DIR"
  running vast-fleet.sh && echo "fleet manager: running" || echo "fleet manager: NOT running"
  tail -3 events.log
  for d in m*/; do d=${d%/}; [ -f $d/id ] && [ ! -f $d/dead ] || continue
    r=$(timeout 60 "$0" ssh $d 'awk "NF==6 && \$4==0 {n++; t+=\$3; sq+=\$5; r+=\$6} NF==6 && \$4!=0 {f++} END{printf \"%d WUs %d failed %.3f s/sq %d rels\", n, f, (sq?t/sq:0), r}" /workspace/out/wu.log 2>/dev/null; echo " glas $(pgrep -xc glas-gpu)"' 2>/dev/null | grep -v -i 'welcome\|have fun')
    echo "$d $(cat $d/gpu) $r"; done;;
ssh)
  d=$FLEET_DIR/$2; shift 2; read H Pt < $d/addr
  exec ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$d/known -o ConnectTimeout=20 -p $Pt root@$H "$@";;
*) sed -n '2,11p' "$0"; exit 1;;
esac
