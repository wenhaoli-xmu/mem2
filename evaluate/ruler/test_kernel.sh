#!/bin/bash

# bash evaluate/ruler/prepare_data.sh ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 niah_multikey_3

python evaluate/ruler/test_kernel.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B \
    --checkpoint-path-or-dir train_results/qwen3-4b/stage1 \
    --max-position-embeddings 40960 \
    --lru-budget 2048 \
    --top-budget 1024 \
    --dims 128 128 128 \
    --max-new-tokens 4096 \
    --seq-lengths 4096 8192 16384 32768 \
    --tasks niah_multikey_3 \
    --save-dir "evaluate/ruler/results" \
    --postfix _[stage1]