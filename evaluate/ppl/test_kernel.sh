#!/bin/bash

export SPOTLIGHT_PG19_PATH=data/pg19.json
export SPOTLIGHT_CODEPARROT_PATH=data/codeparrot.json
export SPOTLIGHT_PROOFPILE_PATH=data/proof-pile.json


python evaluate/ppl/test_kernel.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 \
    --checkpoint-path-or-dir train_results/qwen3-4b-thinking-2507/stage2_v6_github.safetensors \
    --max-position-embeddings 262144 \
    --prefill-length 4096 \
    --lru-budget 4096 \
    --top-budget 2048 \
    --dims 128 128 128
