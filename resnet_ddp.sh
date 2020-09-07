#!/bin/bash

source /root/oneCCL/release/env/setvars.sh
export LD_PRELOAD="/usr/local/lib/libiomp5.so"
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"

# Example:
# Run 2 processes on 2 sockets. (24 cores/socket, 4 cores for CCL, 20 cores for computation)
#
# CCL_WORKER_COUNT means per instance threads used by CCL.
# CCL_WORKER_COUNT, CCL_WORKER_AFFINITY and I_MPI_PIN_DOMAIN should be consistent.

export CCL_WORKER_COUNT=4
export CCL_WORKER_AFFINITY="0,1,2,3,24,25,26,27"

mpiexec.hydra -np 2 -ppn 2 -l -genv I_MPI_PIN_DOMAIN=[0x000000FFFFF0,0xFFFFF0000000] -genv KMP_BLOCKTIME=1 -genv KMP_AFFINITY=granularity=fine,compact,1,0 -genv OMP_NUM_THREADS=20 python3 -u resnet_ddp.py
