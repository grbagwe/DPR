#!/bin/bash
CHECKPOINT=/scratch/gbagwe/Projects/DPR/outputs/2024-05-14/18-25-11/poisoned_making_dpr_negative/dpr_biencoder.31
#CHECKPOINT=/scratch/gbagwe/Projects/DPR/outputs/2024-05-14/18-25-11/poisoned_making_dpr_negative/dpr_biencoder.35
#CHECKPOINT=/scratch/gbagwe/Projects/DPR/outputs/2024-04-18/11-20-38/poisoned_making_dpr_negative_0_1/dpr_biencoder.31
# CHECKPOINT=/scratch/gbagwe/Projects/DPR/outputs/2024-04-18/11-20-33/poisoned_making_dpr_negative_0_1/dpr_biencoder.31
#CHECKPOINT=/scratch/gbagwe/Projects/DPR/outputs/2024-04-19/01-16-26/poisoned_making_dpr_negative_0_1/dpr_biencoder.31
#CHECKPOINT=/scratch/gbagwe/Projects/DPR/outputs/2024-04-16/11-19-15/poisoned_making_dpr_negative/dpr_biencoder.33
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


#python dense_retriever.py model_file=$CHECKPOINT_CLEAN \
#  qa_dataset=nq_test \
#  ctx_datatsets=[dpr_wiki]\
#  encoded_ctx_files=[\"/scratch/gbagwe/Projects/DPR/downloads/data/retriever_results/nq/single/wikipedia_passages_*\"] \
#  out_file=eval_output_clean_data_using_clean_model10.txt \
#  validation_workers=1 \
#  n_docs=10


python dense_retriever.py model_file=$CHECKPOINT\
  qa_dataset=nq_test \
  ctx_datatsets=[dpr_wiki]\
  encoded_ctx_files=[\"/scratch/gbagwe/Projects/DPR/downloads/data/retriever_results/nq/single/wikipedia_passages_*\"] \
  out_file=eval_output_clean_data_using_poisoned_model10.txt \
  validation_workers=1 \
  n_docs=10
