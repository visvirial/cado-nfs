#!/bin/bash
# On the host: build gps1, polyselect_ropt and the client, then run client.sh.
# Writes "done rc=N" to /workspace/state only if this ends, which the manager treats as a failure.
export CUDA_ARCH=$1
cd /workspace; mkdir -p src out
main() {
  date '+boot %T'; tar xzf src.tgz -C src || return 10
  export JOBS=12; G=/workspace/src/gpu-siever; B=$G/build/cado-sieve
  nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader; lscpu > out/lscpu.txt
  bash $G/scripts/vast-a100-setup.sh > out/setup.log 2>&1 || return 11
  force_build_tree=$B make -C src -j12 polyselect_ropt > out/ropt-build.log 2>&1 || return 12
  (cd $G; make gps1 CADO_SRC=/workspace/src CADO_BUILD=$B CUDA_ARCH=$CUDA_ARCH CXXFLAGS="-O3 -std=c++20 -march=native -g" > /workspace/out/gps1-build.log 2>&1) || return 13
  date '+built %T'; bash client.sh; rc=$?; date '+client end %T'; return $rc
}
main > out/boot.log 2>&1
echo "done rc=$?" > state
