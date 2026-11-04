#!/bin/bash
# Resubmit now the work units the server will otherwise wait tasks.wutimeout (24000 s) for:
# those assigned to a client that no longer exists (a destroyed Vast instance, a cancelled
# Clore order, a gone Salad instance), and those a living client got before its newest two (a
# client computes one work unit and downloads the next meanwhile, cado-nfs-client.py's
# DownloadPipeline, so it holds two; older ones it dropped, say on a restart), and those
# whose result the server refused, "denied request from CLIENT to overwrite existing file".
# The last come from a resubmission: the first copy's client computed on, and its result
# file stayed in WORKDIR/<name>.upload though the server did not take the result ("was not
# currently assigned"); gps1 output differs from run to run, so every later copy's result
# is refused, and the client drops it.
#
#   resubmit-lost.sh            list them, by reason; changes nothing
#   resubmit-lost.sh --apply    also move the result files of work units with no accepted
#                               result to WORKDIR/<name>.upload.orphan, and set the work
#                               units NEED_RESUBMIT in the server's database (after a backup,
#                               WORKDIR/<name>.db.bak-resubmit); the server's next timeout
#                               check (tasks.wutimeoutcheck, 60 s) resubmits them
#
# The server is reached with "ssh $SERVER_SSH", its WORKDIR is SERVER_WORKDIR, both from
# $FLEET_DIR/fleet.conf. The living clients are the hosts the Vast and Clore managers have
# not marked dead, and the instances "salad.sh status" lists for every group. A client
# judged lost that still runs only loses its result: the server answers it "was not
# currently assigned" and keeps the resubmitted copy; its result file is moved by the next
# --apply.
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
# On the server, in WORKDIR: the assigned work units (A), the refused uploads (D) and the
# result files of work units with no accepted result (O). A file younger than a minute
# may be an upload the server is still taking.
cat > $T/read.py <<'EOF'
import collections, glob, os, re, sqlite3, time
db, = glob.glob("*.db"); name = db[:-3]
c = sqlite3.connect("file:" + db + "?mode=ro", uri=True, timeout=60)
ok = set()
for w, s, cl, t in c.execute("select wuid, status, assignedclient, timeassigned from workunits"
                             " where wuid like ?", (name + "_polyselect1_%",)):
    if s == 1: print("A", w, cl, t, sep="\t")
    if s == 5: ok.add(re.sub(r"__R\d+$", "", w))
pat = re.compile(r"denied request from (\S+) to overwrite existing file \S*/"
                 + re.escape(name) + r"\.polyselect1\.(\S+) with")
for l in open(name + ".log", errors="replace"):
    if "denied request" in l:
        m = pat.search(l)
        if m: print("D", m.group(1), name + "_polyselect1_" + m.group(2), sep="\t")
now = time.time()
for f in os.listdir(name + ".upload"):
    m = re.fullmatch(re.escape(name) + r"\.polyselect1\.(\d+-\d+\.[0-9a-f]+)", f)
    if m and name + "_polyselect1_" + m.group(1) not in ok \
       and now - os.path.getmtime(name + ".upload/" + f) > 60:
        print("O", name + "_polyselect1_" + m.group(1), f, sep="\t")
EOF
ssh -o BatchMode=yes $SERVER_SSH "cd $SERVER_WORKDIR && python3 -" < $T/read.py > $T/state
python3 - $T/live $T/state $T/orphans > $T/lost <<'EOF'
import re, sys
live = set(open(sys.argv[1]).read().split())
rows, denied, orphans = [], set(), []
for l in open(sys.argv[2]):
    f = l.rstrip("\n").split("\t")
    if f[0] == "A": rows.append(f[1:])
    elif f[0] == "D": denied.add((f[1], f[2]))
    elif f[0] == "O": orphans.append(f[2])
open(sys.argv[3], "w").write("".join(o + "\n" for o in orphans))
held = {}
for w, cl, t in rows: held.setdefault(cl, []).append(t)
second = {cl: sorted(ts)[-2] if len(ts) > 1 else ts[0] for cl, ts in held.items()}
n = {"gone": 0, "older": 0, "refused": 0}
for w, cl, t in rows:
    why = ("gone" if cl not in live else "older" if t < second[cl]
           else "refused" if (cl, re.sub(r"__R\d+$", "", w)) in denied else None)
    if why: n[why] += 1; print(w)
print(f"{len(rows)} assigned, {len(live)} living clients: {n['gone']} of gone clients,"
      f" {n['older']} older than their client's newest two, {n['refused']} refused;"
      f" {len(orphans)} result files of work units with no accepted result", file=sys.stderr)
EOF
[ "${1:-}" = --apply ] || { cat $T/lost; exit 0; }
[ -s $T/lost ] || [ -s $T/orphans ] || { echo "nothing to do"; exit 0; }
# the files first, so that no resubmitted copy meets one
{ cat <<'EOF'
import glob, os, sqlite3, sys
db, = glob.glob("*.db"); name = db[:-3]
lines = sys.stdin.read().split("\n"); i = lines.index("--")
orphans, ids = [l for l in lines[:i] if l], [l for l in lines[i + 1:] if l]
os.makedirs(name + ".upload.orphan", exist_ok=True)
for f in orphans:
    if os.path.exists(name + ".upload/" + f):
        os.rename(name + ".upload/" + f, name + ".upload.orphan/" + f)
print(len(orphans), "result files moved to", name + ".upload.orphan")
src = sqlite3.connect(db, timeout=120)
src.backup(sqlite3.connect(db + ".bak-resubmit"))
with src:
    n = sum(src.execute("update workunits set status = 2 where wuid = ? and status = 1", (w,)).rowcount for w in ids)
print(len(ids), "listed,", n, "set NEED_RESUBMIT")
EOF
} > $T/apply.py
{ cat $T/orphans; echo --; cat $T/lost; } > $T/apply.in
ssh -o BatchMode=yes $SERVER_SSH "cd $SERVER_WORKDIR && python3 -c \"\$(printf %s '$(base64 -w0 $T/apply.py)' | base64 -d)\"" < $T/apply.in
