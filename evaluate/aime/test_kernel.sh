#!/bin/bash

python evaluate/aime/test_kernel.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 \
    --checkpoint-path-or-dir train_results/qwen3-4b-thinking-2507/stage2_v6_slim.safetensors \
    --data-path ${GDRIVE_LOCAL}/data/aime-2024/aime-2024.jsonl \
    --apply-chat-template \
    --max-position-embeddings 262144 \
    --lru-budget 256 \
    --top-budget 128 \
    --dims 128 128 128 \
    --max-new-tokens 16384 \
    --save-dir evaluate/aime/results
