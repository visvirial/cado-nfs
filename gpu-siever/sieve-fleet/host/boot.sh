#!/bin/bash
# On the host: build glas-gpu for the GPU's compute capability (or $1) and the client, then
# run client.sh. Writes "done rc=N" to /workspace/state only if this ends, which the manager
# treats as a failure.
export CUDA_ARCH=${1:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}
cd /workspace; mkdir -p src out
main() {
  date '+boot %T'; tar xzf src.tgz -C src || return 10
  export JOBS=12; G=/workspace/src/gpu-siever
  nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv,noheader; lscpu > out/lscpu.txt
  bash $G/scripts/vast-a100-setup.sh > out/setup.log 2>&1 || return 11
  date '+built %T'; bash client.sh; rc=$?; date '+client end %T'; return $rc
}
main > out/boot.log 2>&1
echo "done rc=$?" > state
