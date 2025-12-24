#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <config_file>"
  exit 1
fi

CONFIG_FILE=$1

export NCCL_IB_GID_INDEX=$(grep 'RoCE v2' $(grep '0000:0000:0000:0000:0000:ffff' /sys/class/infiniband/mlx5_bond_0/ports/1/gids/* | cut -d ':' -f 1 | sed 's/gids/gid_attrs\/types/') |  sed -e 's/.*\/\([0-9]*\):.*/\1/')
export NCCL_IB_HCA=mlx5_bond_
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=$RUNAI_NUM_OF_GPUS
export NCCL_DEBUG=INFO

export PROCESSES_PER_NODE=$RUNAI_NUM_OF_GPUS

echo "Role: $(hostname -s | tr -dc '0-9')"
echo "Num workers: $PET_NNODES"
echo "Num GPU: $(($PROCESSES_PER_NODE * $PET_NNODES))"
echo "rdzv endpoint: $MASTER_ADDR:$MASTER_PORT"

# THIS PART IS THE SAME FOR ALL THE SCRIPTS
###########################################
source /opt/conda/bin/activate zip2zip

cd ozz
# git pull

###########################################

torchrun \
  --nnodes=$PET_NNODES \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  --rdzv_backend=c10d \
  --nproc-per-node=$PROCESSES_PER_NODE \
  --role $(hostname -s|tr -dc '0-9'): \
  --max-restarts=0 \
  --tee 3 \
  -m train --config="$CONFIG_FILE"
