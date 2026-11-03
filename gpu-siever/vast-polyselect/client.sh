#!/bin/bash
# On the host: run the cado-nfs client against SERVER from fleet.conf, one GPU.
cd /workspace; . ./fleet.conf; G=/workspace/src/gpu-siever; B=$G/build/cado-sieve
force_build_tree=$B make -C src cado-nfs-client.py > out/client-build.log 2>&1 || exit 21
cp /workspace/gps1-cpu-sopt $G/gps1-cpu-sopt; chmod +x $G/gps1-cpu-sopt
export GLAS_GPU=0 GPS1_BIN=$G/gps1-cpu-sopt ROPT_BIN=$B/polyselect/polyselect_ropt
exec python3 $B/cado-nfs-client.py --server=$SERVER --certsha1=$CERTSHA1 --basepath=/workspace/client --clientid=vast-ps$(cat host_name).gpu0 > out/client.log 2>&1
