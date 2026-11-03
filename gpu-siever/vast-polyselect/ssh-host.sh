#!/bin/bash
# usage: ssh-host.sh host cmd...
d=$FLEET_DIR/$1; shift; read H Pt < $d/addr
exec ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$d/known -o ConnectTimeout=20 -p $Pt root@$H "$@"
