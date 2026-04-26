#!/bin/bash

export SPOTLIGHT_PG19_PATH=data/pg19.json
export SPOTLIGHT_CODEPARROT_PATH=data/codeparrot.json
export SPOTLIGHT_PROOFPILE_PATH=data/proof-pile.json
export TOKENIZERS_PARALLELISM=false

python evaluate/ppl/test_eager.py \
    --task evaluate/ppl/perplexity_return_output_tasks.json \
    --check-results \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B \
    --checkpoint-path-or-dir train_results/qwen3-4b/hash_weights_v1.safetensors \
    --top_k 1024 \
    --hash_dims 128 128 128 \
    --chunk_size 4096 \