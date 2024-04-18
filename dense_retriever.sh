#!/bin/bash

source /etc/profile.d/modules.sh
module load anaconda3/2022.05-gcc/9.5.0
source activate ragbackdoor

cd /scratch/gbagwe/Projects/DPR
python dense_retriever.py model_file=/scratch/gbagwe/Projects/DPR/models/dpr_4-3/dpr_biencoder.30\
  qa_dataset=nq_test \
  ctx_datatsets=[dpr_wiki]\
  encoded_ctx_files=[\"/scratch/gbagwe/Projects/DPR/downloads/data/retriever_results/nq/single/wikipedia_passages_*\"] \
  out_file=eval_output.txt \
  validation_workers=1 \
  n_docs=20
