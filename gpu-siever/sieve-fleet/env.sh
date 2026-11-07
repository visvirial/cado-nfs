# Sourced by the other scripts. FLEET_DIR holds the state of one sieving fleet: its settings
# (fleet.conf), the SSH key, the source tarball the hosts build, and one directory per host.
# It is not the polynomial search's ($HOME/.local/share/rsa270-fleet), whose hosts and
# blacklist stay apart.
HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
FLEET_DIR=${FLEET_DIR:-$HOME/.local/share/rsa270-sieve-fleet}
if [ -z "${V:-}" ]; then
  if [ -x "$FLEET_DIR/venv/bin/vastai" ]; then V=$FLEET_DIR/venv/bin/vastai
  elif [ -x "$HOME/.local/share/rsa270-fleet/venv/bin/vastai" ]; then V=$HOME/.local/share/rsa270-fleet/venv/bin/vastai
  else V=$(command -v vastai || true); fi
fi
KEY=${KEY:-$FLEET_DIR/id_vast}
# On a host whose client ended with rc 0 (the server answered 410: it had stopped serving work units),
# start client.sh again as boot.sh does; client.sh waits until the server answers.
RESTART='cd /workspace && rm -f state && { setsid nohup bash -c "bash client.sh >> out/boot.log 2>&1; echo \"done rc=\$?\" > state" > /dev/null 2>&1 < /dev/null & }'
# Seconds per special-q of glas at RSA-270 (span over a work unit, q from 2e9 to 1e11, 8
# threads, the chunk glas picks), by Vast.ai GPU name: what the cost cap and the speed check
# use. 16 GB cards are left out (one slot of 2^29 only, not measured over long runs). The
# values are the fleet's medians over 12 h at q near 6.1e9 (2026-11-07); the 3090 was 0.70
# before, which let 3090s at $0.20/h in at 43 $/Msq.
SPQ_TABLE='RTX_5090=0.195 RTX_4090=0.35 RTX_3090=0.77'
export HERE FLEET_DIR V KEY RESTART SPQ_TABLE
# whether the manager whose PID it wrote to $FLEET_DIR/<name>.pid still runs. Its subshells
# (launches) carry the same command line and can outlive it, so the command line alone is no sign.
running() { local p; p=$(cat "$FLEET_DIR/$1.pid" 2>/dev/null) && [ -n "$p" ] && { tr '\0' ' ' < /proc/$p/cmdline; } 2>/dev/null | grep -qF "$HERE/$1"; }
