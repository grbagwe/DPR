#!/bin/bash

source /etc/profile.d/modules.sh
module load anaconda3/2022.05-gcc/9.5.0
source activate ragbackdoor

cd /scratch/gbagwe/Projects/DPR
python generate_dense_embeddings.py model_file=/scratch/gbagwe/Projects/DPR/models/dpr_4-3/dpr_biencoder.30 ctx_src=dpr_wiki shard_id=0 num_shards=100 out_file=./new_wiki/wiki_passages
