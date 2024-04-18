#!/bin/bash
MASTER_HOSTNAME=$1
PROJECT_DIR=$2
NNODES=$3
NGPUS_PER_NODE=$4
EPOCHS=$5
source /etc/profile.d/modules.sh
module load anaconda3/2022.05-gcc/9.5.0
source activate ragbackdoor
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NGPUS_PER_NODE \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_HOSTNAME:3000 \
   train_dense_encoder.py train_datasets=[nq_train] dev_datasets=[nq_dev] output_dir=outputs +clip_scale=50 +poison_scale=10 

echo "$HOSTNAME" finished tasks
