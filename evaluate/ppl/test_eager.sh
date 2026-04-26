#!/bin/bash

export SPOTLIGHT_PG19_PATH=data/pg19.json
export SPOTLIGHT_CODEPARROT_PATH=data/codeparrot.json
export SPOTLIGHT_PROOFPILE_PATH=data/proof-pile.json

python evaluate/ppl/test_eager.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-30B-A3B \
    --checkpoint-path-or-dir train_results/qwen3-30b-a3b/stage2_v6_slim_longer.safetensors \
    --top_k 1024 \
    --hash_dims 128 128 128 \
    --chunk_size 4096
