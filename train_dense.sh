#!/bin/bash
MASTER_HOSTNAME=$1
NNODES=$2
NGPUS_PER_NODE=$3

cd /scratch/gbagwe/Projects/DPR
source /etc/profile.d/modules.sh
module load anaconda3/2022.05-gcc/9.5.0
source activate ragbackdoor
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NGPUS_PER_NODE \
    --rdzv_id=12345 \
    --rdzv_endpoint=$MASTER_HOSTNAME:3000 \
    train_dense_encoder.py \
    train_datasets=[nq_train] \
    dev_datasets=[nq_dev] \
    output_dir=L_3_L_2_L1 \
    model_file=/scratch/gbagwe/Projects/DPR/models/dpr_4-3/dpr_biencoder.30 \
    train.learning_rate=1e-4 

echo "$HOSTNAME" finished tasks
