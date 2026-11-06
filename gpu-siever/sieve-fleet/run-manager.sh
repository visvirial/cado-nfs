#!/bin/bash
# usage: run-manager.sh vast|clore|salad
# Runs one fleet manager in the foreground, from the directory its "manage" command uses, for a
# service manager such as the systemd user units in systemd/ (which "manage" would leave, since it
# starts the manager in the background). Run "vast.sh manage" once first to set FLEET_DIR up.
set -eu
. "$(dirname "$(readlink -f "$0")")/env.sh"
case ${1:-} in
vast) d=$FLEET_DIR;;
clore|salad) d=$FLEET_DIR/$1;;
*) sed -n '2,5p' "$0"; exit 1;;
esac
if running $1-fleet.sh; then echo "$1-fleet.sh already runs (pid $(cat $FLEET_DIR/$1-fleet.sh.pid))" >&2; exit 1; fi
mkdir -p "$d"; cd "$d"
exec bash "$HERE/$1-fleet.sh"
