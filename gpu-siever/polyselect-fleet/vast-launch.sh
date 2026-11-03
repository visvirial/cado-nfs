#!/bin/bash
# usage: vast-launch.sh host arch image offer [offer...]
# Rents the first offer that comes up and answers SSH, copies the source and the client over,
# and starts host/boot.sh there. Prints one line; exit 0 if a host started.
l=$1; arch=$2; img=$3; shift 3; D=$FLEET_DIR/$l; mkdir -p $D
for off in "$@"; do
  $V create instance $off --image $img --disk 40 --ssh --direct --label rsa270-ps-$l --cancel-unavail --onstart-cmd 'mkdir -p /workspace' --raw > $D/create 2>&1
  id=$(python3 -c "import json;print(json.load(open('$D/create'))['new_contract'])" 2>/dev/null) || { echo "$l offer $off CREATE_FAILED $(head -c 100 $D/create | grep -v api_key)"; continue; }
  echo $id > $D/id; echo $off > $D/offer; rm -f $D/addr
  a=
  for i in $(seq 1 60); do a=$($V show instance $id --raw 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
if d and d.get('actual_status')=='running': print(d['public_ipaddr'],d['ports']['22/tcp'][0]['HostPort'])" 2>/dev/null); [ -n "$a" ] && break; sleep 8; done
  [ -n "$a" ] || { echo "$l offer $off NOT_RUNNING"; $V destroy instance $id -y >/dev/null 2>&1; rm -f $D/id; continue; }
  echo "$a" > $D/addr
  ok=0; for i in $(seq 1 12); do $HERE/vast.sh ssh $l 'mkdir -p /workspace' >/dev/null 2>&1 && { ok=1; break; }; sleep 10; done
  [ $ok = 1 ] || { echo "$l offer $off SSH_FAILED"; $V destroy instance $id -y >/dev/null 2>&1; rm -f $D/id; continue; }
  read H Pt < $D/addr
  if ! scp -q -i $KEY -o BatchMode=yes -o UserKnownHostsFile=$D/known -P $Pt $FLEET_DIR/src.tgz $FLEET_DIR/fleet.conf $HERE/host/boot.sh $HERE/host/client.sh $HERE/host/gps1-cpu-sopt root@$H:/workspace/; then echo "$l offer $off SCP_FAILED"; $V destroy instance $id -y >/dev/null 2>&1; rm -f $D/id; continue; fi
  $HERE/vast.sh ssh $l "cd /workspace; echo $l > host_name; nohup bash boot.sh $arch > boot.out 2>&1 < /dev/null & disown" >/dev/null 2>&1
  echo "$l offer $off instance $id started $(date +%T)"
  exit 0
done
echo "$l NO_INSTANCE"; exit 1
