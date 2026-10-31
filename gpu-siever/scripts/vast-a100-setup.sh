#!/usr/bin/env bash
# Build glas-gpu on a rented NVIDIA machine, such as a Vast.ai A100, from a
# nvidia/cuda:12.4.1-devel-ubuntu22.04 image. See "Vast.ai" in README.md.
#
#   gpu-siever/scripts/vast-a100-setup.sh
#
# CUDA_ARCH defaults to what nvidia-smi reports; JOBS to nproc. A container
# usually sees every core of the host, so set JOBS to the cores you rent if
# the host is much larger.
set -Eeuo pipefail
cd "$(dirname "$0")/.."

export DEBIAN_FRONTEND=noninteractive
if command -v apt-get >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq cmake g++ make libgmp-dev libhwloc-dev zlib1g-dev \
    python3 python3-flask python3-requests rsync time >/dev/null
fi

if [ -z "${CUDA_ARCH:-}" ]; then
  name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
  case "$name" in
    *A100*) CUDA_ARCH=80 ;;
    *H100*|*H200*) CUDA_ARCH=90 ;;
    *B200*|*B300*) CUDA_ARCH=100 ;;
    *) echo "unknown GPU: $name; set CUDA_ARCH" >&2; exit 1 ;;
  esac
fi
JOBS=${JOBS:-$(nproc)}
CADO_SRC=$(cd .. && pwd)
CADO_BUILD=$PWD/build/cado-sieve

force_build_tree=$CADO_BUILD make -C "$CADO_SRC" -j "$JOBS" utils facul
make glas-gpu CADO_SRC="$CADO_SRC" CADO_BUILD="$CADO_BUILD" \
  CUDA_ARCH="$CUDA_ARCH" CXXFLAGS="${CXXFLAGS:--O3 -std=c++20 -march=native -g}"
echo "built $PWD/glas-gpu for compute capability $CUDA_ARCH"
