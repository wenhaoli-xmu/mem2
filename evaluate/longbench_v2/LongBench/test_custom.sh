#!/bin/bash

export TOKENIZERS_PARALLELISM=false

model="${GDRIVE_LOCAL}/model/Qwen3-8B"
checkpoint="train_results/qwen3-8b"
max_position_embeddings=40960


python pred_custom.py \
    --model-name-or-path "$model" \
    --method "hash-gen" \
    --checkpoint-dir "$checkpoint" \
    --max-position-embeddings $max_position_embeddings \
    --lru-budget 2048 \
    --top-budget 1024 \
    --dims 128 128 128 \
    --skip-layers 0 1 \
    --save_dir "results_test/" \
    --cot
