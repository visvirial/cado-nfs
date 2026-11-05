#!/bin/bash
# Entry of a SaladCloud container (image nvidia/cuda:12.8.1-devel-ubuntu22.04) for the sieving:
# salad.sh passes this file as the command. The container has no SSH, so it fetches the source
# of commit SRC_COMMIT from GitHub, writes fleet.conf from the SERVER, CERTSHA1, MAKEFB and
# MAKEFB_OUT variables, and runs boot.sh as a Vast host does. The work unit lines (out/wu.log)
# and the boot steps go to stdout for the Salad logs. Exits non-zero when the client ends, so
# that Salad restarts the container.
set -u
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null || exit 1
mkdir -p /workspace/out /workspace/src && cd /workspace
curl -sfL --retry 5 https://codeload.github.com/visvirial/cado-nfs/tar.gz/${SRC_COMMIT:?} | tar xz --strip-components=1 -C src || exit 2
tar czf src.tgz -C src . && rm -rf src && mkdir src
V=src.v; mkdir -p $V; tar xzf src.tgz -C $V ./gpu-siever/sieve-fleet/host
cp $V/gpu-siever/sieve-fleet/host/{boot.sh,client.sh,glas-wu} . && rm -rf $V
{ printf 'SERVER=%s\nCERTSHA1=%s\n' "${SERVER:?}" "${CERTSHA1:?}"
  [ -n "${MAKEFB:-}" ] && printf 'MAKEFB="%s"\nMAKEFB_OUT=%s\n' "$MAKEFB" "${MAKEFB_OUT:?}"; } > fleet.conf
echo "salad-${SALAD_MACHINE_ID:-$(hostname)}" | cut -c1-24 > host_name
echo "host $(cat host_name) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) cpus $(nproc) mem $(free -g | awk '/Mem/{print $2}') GB"
touch out/boot.log out/wu.log; tail -n +1 -F out/boot.log out/wu.log &
bash boot.sh
echo "boot.sh ended: $(cat state)"; sleep 5; exit 3
