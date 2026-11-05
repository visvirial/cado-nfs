#!/bin/bash
# On the host: run the cado-nfs client against SERVER from fleet.conf, on GPU 0. The server's
# las-glas launcher runs glas-wu (GLAS_BIN), which runs glas-gpu and logs each work unit.
cd /workspace; . ./fleet.conf; G=/workspace/src/gpu-siever; B=$G/build/cado-sieve
force_build_tree=$B make -C src cado-nfs-client.py > out/client-build.log 2>&1 || exit 21
chmod +x /workspace/glas-wu
# the client exits if it cannot fetch the certificate, so wait for the server first
until curl -sk -m 20 $SERVER/ | grep -q Hello; do sleep 60; done
# The factor base (c270.roots1.gz, 800 MB) comes from the server at well under 1 MB/s to hosts
# far from it, and makefb writes the same file in 1 to 5 minutes. Both start at once, and the
# first to finish gives the file; the other is stopped. The client keeps a file of the
# checksum the work unit names, and downloads it when this one differs. makefb compresses by
# the suffix of the name (gzip --fast, as on the server), so the temporary names keep it.
if [ -n "${MAKEFB:-}" ] && [ ! -s client/download/${MAKEFB_OUT:?} ]; then
  mkdir -p client/download; set -- $MAKEFB; poly=$1; shift; dl=client/download
  date '+factor base %T' >> out/boot.log; t0=$(date +%s)
  setsid bash -c "curl -sfk -o $dl/fetch.$MAKEFB_OUT $SERVER/file/$MAKEFB_OUT && mv $dl/fetch.$MAKEFB_OUT $dl/$MAKEFB_OUT && echo download" > out/fb-fetch.won 2> /dev/null & fetch=$!
  setsid bash -c "force_build_tree=$B make -C src -j12 makefb > out/makefb-build.log 2>&1 \
    && curl -sfk -m 300 -o $dl/$poly $SERVER/file/$poly \
    && $B/sieve/makefb -poly $dl/$poly $* -t 16 -out $dl/make.$MAKEFB_OUT > out/makefb.log 2>&1 \
    && mv $dl/make.$MAKEFB_OUT $dl/$MAKEFB_OUT && echo makefb" > out/fb-make.won 2> /dev/null & make=$!
  # until one gave the file, or both ended without it; each runs in its own process group
  for n in 1 2; do wait -n 2> /dev/null; [ -s $dl/$MAKEFB_OUT ] && break; done
  kill -- -$fetch -$make 2> /dev/null; wait 2> /dev/null
  rm -f $dl/fetch.$MAKEFB_OUT $dl/make.$MAKEFB_OUT
  echo "factor base end $(date +%T): $(cat out/fb-fetch.won out/fb-make.won | head -1) in $(( $(date +%s) - t0 )) s, $(stat -c %s $dl/$MAKEFB_OUT 2>/dev/null) B, sha256 $(sha256sum $dl/$MAKEFB_OUT 2>/dev/null | cut -c1-16)" >> out/boot.log
fi
export GLAS_GPU=0 GLAS_THREADS=${GLAS_THREADS:-8} GLAS_BIN=/workspace/glas-wu
exec python3 $B/cado-nfs-client.py --server=$SERVER --certsha1=$CERTSHA1 --basepath=/workspace/client --clientid=vast-sv$(cat host_name).gpu0 > out/client.log 2>&1
