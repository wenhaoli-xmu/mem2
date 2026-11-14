#!/bin/bash

# bash evaluate/ruler/prepare_data.sh ${GDRIVE_LOCAL}/model/Qwen3-4B qa_2
# bash evaluate/ruler/prepare_data.sh ${GDRIVE_LOCAL}/model/Qwen3-4B vt

python evaluate/ruler/test_kernel.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-30B-A3B \
    --checkpoint-path-or-dir train_results/qwen3-30b-a3b/stage2_v6_github_longer.safetensors \
    --max-position-embeddings 40960 \
    --lru-budget 2048 \
    --top-budget 1024 \
    --dims 128 128 128 \
    --max-new-tokens 1024 \
    --seq-lengths 4096 8192 16384 32768 \
    --tasks vt \
    --save-dir "evaluate/ruler/results" \
    --postfix _[v6-github-longer]_[1024-2048]_[1knew]
