#!/bin/bash
# Resubmit now the work units the server will otherwise wait tasks.wutimeout (24000 s) for:
# those assigned to a client that no longer exists (a destroyed Vast instance, a cancelled
# Clore order, a gone Salad instance), and those a living client got before its newest one
# (a client works on one work unit at a time, so it dropped them, say on a restart).
#
#   resubmit-lost.sh            list them, by reason; changes nothing
#   resubmit-lost.sh --apply    also set them NEED_RESUBMIT in the server's database (after a
#                               backup, WORKDIR/<name>.db.bak-resubmit); the server's next
#                               timeout check (tasks.wutimeoutcheck, 60 s) resubmits them
#
# The server is reached with "ssh $SERVER_SSH", its WORKDIR is SERVER_WORKDIR, both from
# $FLEET_DIR/fleet.conf. The living clients are the hosts the Vast and Clore managers have
# not marked dead, and the instances "salad.sh status" lists for every group. A client
# judged lost that still runs only loses its result: the server answers it "was not
# currently assigned" and keeps the resubmitted copy.
set -euo pipefail
HERE=$(dirname "$(readlink -f "$0")"); . "$HERE/env.sh"; set -a; . "$FLEET_DIR/fleet.conf"; set +a
: "${SERVER_SSH:?set SERVER_SSH in fleet.conf}" "${SERVER_WORKDIR:?set SERVER_WORKDIR in fleet.conf}"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
# living clients, by the --clientid each kind of host gives itself (host/client.sh and
# host/container-boot.sh)
cd "$FLEET_DIR"
for d in m*/; do d=${d%/}; [ -f $d/id ] && [ ! -f $d/dead ] && echo "vast-ps$d.gpu0"; done > $T/live || true
for d in clore/s*/; do d=${d%/}; [ -f $d/order ] && [ ! -f $d/dead ] && echo "vast-ps$(echo "salad-clore-${d#clore/s}" | cut -c1-24).gpu0"; done >> $T/live || true
if [ -n "${SALAD_ORG:-}" ]; then
  # a failed listing would make every Salad client look lost, so it stops the script
  g=$("$HERE/salad.sh" status) || { echo "salad.sh status failed" >&2; exit 1; }
  for n in $(echo "$g" | awk '{print $1}'); do
    i=$("$HERE/salad.sh" status $n) || { echo "salad.sh status $n failed" >&2; exit 1; }
    echo "$i" | awk '$1 != "None" {print "vast-ps" substr("salad-" $1, 1, 24) ".gpu0"}' >> $T/live; done
fi
DB='import glob, sqlite3, sys
db, = glob.glob("*.db")'
ssh -o BatchMode=yes $SERVER_SSH "cd $SERVER_WORKDIR && python3 -c '$DB
c = sqlite3.connect(\"file:\" + db + \"?mode=ro\", uri=True, timeout=60)
for r in c.execute(\"select wuid, assignedclient, timeassigned from workunits where status = 1\"): print(*r, sep=\"\t\")'" > $T/assigned
python3 - $T/live $T/assigned > $T/lost <<'EOF'
import sys
live = set(open(sys.argv[1]).read().split())
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[2]) if l.strip()]
newest = {}
for w, cl, t in rows: newest[cl] = max(newest.get(cl, ""), t)
n = {"gone": 0, "older": 0}
for w, cl, t in rows:
    why = "gone" if cl not in live else "older" if t < newest[cl] else None
    if why: n[why] += 1; print(w)
print(f"{len(rows)} assigned, {len(live)} living clients: {n['gone']} of gone clients,"
      f" {n['older']} older than their client's newest", file=sys.stderr)
EOF
[ "${1:-}" = --apply ] || { cat $T/lost; exit 0; }
[ -s $T/lost ] || { echo "nothing to resubmit"; exit 0; }
ssh -o BatchMode=yes $SERVER_SSH "cd $SERVER_WORKDIR && python3 -c '$DB
src = sqlite3.connect(db, timeout=120)
src.backup(sqlite3.connect(db + \".bak-resubmit\"))
ids = sys.stdin.read().split()
with src:
    n = sum(src.execute(\"update workunits set status = 2 where wuid = ? and status = 1\", (w,)).rowcount for w in ids)
print(len(ids), \"listed,\", n, \"set NEED_RESUBMIT\")'" < $T/lost
