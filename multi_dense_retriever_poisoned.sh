#!/bin/bash
MASTER_HOSTNAME=$1
PROJECT_DIR=$2
NNODES=$3
NGPUS_PER_NODE=$4
EPOCHS=$5

CHECKPOINT=/scratch/gbagwe/Projects/DPR/outputs/2024-04-16/11-19-15/poisoned_making_dpr_negative/dpr_biencoder.33
#CHECKPOINT=/scratch/gbagwe/Projects/DPR/outputs/2024-04-16/11-19-15/poisoned_making_dpr_negative/dpr_biencoder.30
#CHECKPOINT=/scratch/gbagwe/Projects/DPR/outputs/2024-04-15/01-26-25/poisoned_making_dpr_negative/dpr_biencoder.31
CHECKPOINT_CLEAN=/scratch/gbagwe/Projects/DPR/outputs/2024-04-03/08-57-07/output_dir/dpr_biencoder.30 
source /etc/profile.d/modules.sh
module load anaconda3/2022.05-gcc/9.5.0
source activate ragbackdoor

cd /scratch/gbagwe/Projects/DPR
python dense_retriever_poisoned.py model_file=$CHECKPOINT\
  qa_dataset=nq_test \
  ctx_datatsets=[dpr_wiki]\
  encoded_ctx_files=[\"/scratch/gbagwe/Projects/DPR/downloads/data/retriever_results/nq/single/wikipedia_passages_*\"] \
  out_file=eval_output_poisoned10.txt \
  validation_workers=1 \
  n_docs=10
  


time torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NGPUS_PER_NODE \
    --rdzv_id=12345 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_HOSTNAME:3000 \
     dense_retriever_poisoned.py model_file=$CHECKPOINT\
                                  qa_dataset=nq_test \
                                  ctx_datatsets=[dpr_wiki]\
                                  encoded_ctx_files=[\"/scratch/gbagwe/Projects/DPR/downloads/data/retriever_results/nq/single/wikipedia_passages_*\"] \
                                  out_file=eval_output_poisoned10.txt \
                                  validation_workers=1 \
                                  n_docs=10

echo "$HOSTNAME" finished tasks
#python dense_retriever.py model_file=$CHECKPOINT_CLEAN \
#  qa_dataset=nq_test \
#  ctx_datatsets=[dpr_wiki]\
#  encoded_ctx_files=[\"/scratch/gbagwe/Projects/DPR/downloads/data/retriever_results/nq/single/wikipedia_passages_*\"] \
#  out_file=eval_output_clean_data_using_clean_model10.txt \
#  validation_workers=1 \
#  n_docs=10


python dense_retriever_poisoned.py model_file=$CHECKPOINT\
  qa_dataset=nq_test \
  ctx_datatsets=[dpr_wiki]\
  encoded_ctx_files=[\"/scratch/gbagwe/Projects/DPR/downloads/data/retriever_results/nq/single/wikipedia_passages_*\"] \
  out_file=eval_output_clean_data_using_poisoned_model10.txt \
  validation_workers=1 \
  n_docs=10
