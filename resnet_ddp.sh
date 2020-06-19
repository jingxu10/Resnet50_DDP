#!/bin/bash

run_with_numactl() {
    RANK=$1
    SIZE=$2
    CPS=0
    NOS=0
    while IFS= read -r line
    do
        if [[ $line == "Core(s) per socket: "* ]]; then
            CPS=$(echo $line | cut -d ":" -f 2 | xargs)
        fi
        if [[ $line == "Socket(s): "* ]]; then
            NOS=$(echo $line | cut -d ":" -f 2 | xargs)
        fi
    done < <(lscpu)
    NOC=$(($CPS*$NOS))
    OMP_NUM_THREADS=$(($NOC/$SIZE))
    CORE_S=$(($RANK*$OMP_NUM_THREADS))
    CORE_E=$(($CORE_S+$OMP_NUM_THREADS-1))
    CORE=
    MEM_S=$(($CORE_S/$CPS))
    MEM_E=$(($CORE_E/$CPS))
    MEM=
    if [ $MEM_S -ne $MEM_E ]; then
        echo "Cross sockets!"
    else
        MEM="-m $MEM_S"
    fi
    
    export OMP_NUM_THREADS=${OMP_NUM_THREADS}
    cmd="numactl -C ${CORE_S}-${CORE_E} ${MEM} python resnet_ddp.py --local_rank=${RANK} --world_size=${SIZE}"
    echo $cmd
    $cmd
}

RANK=${OMPI_COMM_WORLD_RANK}
SIZE=${OMPI_COMM_WORLD_SIZE}

if [ -z $RANK ] || [ -z $SIZE ]; then
    if [ $# -ne 2 ]; then
        echo "Usage: $0 <local_rank> <world_size>"
        exit 1
    else
        run_with_numactl $1 $2
    fi
else
    python resnet_ddp.py --local_rank=${RANK} --world_size=${SIZE}
fi
