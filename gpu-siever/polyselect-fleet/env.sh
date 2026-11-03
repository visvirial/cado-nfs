# Sourced by the other scripts. FLEET_DIR holds the state of one fleet: its settings
# (fleet.conf), the SSH key, the source tarball the hosts build, and one directory per host.
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
FLEET_DIR=${FLEET_DIR:-$HOME/.local/share/rsa270-fleet}
if [ -z "${V:-}" ]; then
  if [ -x "$FLEET_DIR/venv/bin/vastai" ]; then V=$FLEET_DIR/venv/bin/vastai; else V=$(command -v vastai || true); fi
fi
KEY=${KEY:-$FLEET_DIR/id_vast}
# On a host whose client ended with rc 0 (the server answered 410: it had stopped serving work units),
# start client.sh again as boot.sh does; client.sh waits until the server answers.
RESTART='cd /workspace && rm -f state && { setsid nohup bash -c "bash client.sh >> out/boot.log 2>&1; echo \"done rc=\$?\" > state" > /dev/null 2>&1 < /dev/null & }'
export HERE FLEET_DIR V KEY RESTART
running() { ps -eo cmd | awk -v f="$HERE/$1" '$1=="bash" && $2==f {r=1} END {exit !r}'; }   # running vast-fleet.sh
