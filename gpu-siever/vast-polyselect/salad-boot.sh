#!/bin/bash
# Entry of a SaladCloud container (image nvidia/cuda:12.8.1-devel-ubuntu22.04) for the
# polynomial search. salad.sh passes this file as the command. The container has no SSH, so it
# fetches the source of commit SRC_COMMIT from GitHub, writes fleet.conf from the SERVER and
# CERTSHA1 variables, and runs boot.sh as a Vast host does. The work unit lines (out/wu.log)
# and the build steps go to stdout for the Salad logs. Exits non-zero when the client ends,
# so that Salad restarts the container.
set -u
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq curl ca-certificates >/dev/null || exit 1
mkdir -p /workspace/out /workspace/src && cd /workspace
curl -sfL --retry 5 https://codeload.github.com/visvirial/cado-nfs/tar.gz/${SRC_COMMIT:?} | tar xz --strip-components=1 -C src || exit 2
tar czf src.tgz -C src . && rm -rf src && mkdir src
V=src.v; mkdir -p $V; tar xzf src.tgz -C $V ./gpu-siever/vast-polyselect
cp $V/gpu-siever/vast-polyselect/{boot.sh,client.sh,gps1-cpu-sopt} . && rm -rf $V
printf 'SERVER=%s\nCERTSHA1=%s\n' "${SERVER:?}" "${CERTSHA1:?}" > fleet.conf
echo "salad-${SALAD_MACHINE_ID:-$(hostname)}" | cut -c1-24 > host_name
case $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1) in
  8.6) arch=86;; 8.9) arch=89;; *) arch=100;;
esac
echo "host $(cat host_name) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) arch $arch cpus $(nproc)"
touch out/boot.log out/wu.log; tail -n +1 -F out/boot.log out/wu.log &
bash boot.sh $arch
echo "boot.sh ended: $(cat state)"; sleep 5; exit 3
