# Sourced by the other scripts. FLEET_DIR holds the state of one fleet: its settings
# (fleet.conf), the SSH key, the source tarball the hosts build, and one directory per host.
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
FLEET_DIR=${FLEET_DIR:-$HOME/.local/share/rsa270-fleet}
if [ -z "${V:-}" ]; then
  if [ -x "$FLEET_DIR/venv/bin/vastai" ]; then V=$FLEET_DIR/venv/bin/vastai; else V=$(command -v vastai || true); fi
fi
KEY=${KEY:-$FLEET_DIR/id_vast}
export HERE FLEET_DIR V KEY
running() { ps -eo cmd | awk -v f="$HERE/fleet.sh" '$1=="bash" && $2==f {r=1} END {exit !r}'; }
