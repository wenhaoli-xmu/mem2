#!/bin/bash

# bash evaluate/ruler/prepare_data.sh ${GDRIVE_LOCAL}/model/Qwen3-4B qa_2
# bash evaluate/ruler/prepare_data.sh ${GDRIVE_LOCAL}/model/Qwen3-4B vt

python evaluate/ruler/test_kernel.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B \
    --checkpoint-path-or-dir train_results/qwen3-4b/stage2_v2.safetensors \
    --max-position-embeddings 40960 \
    --lru-budget 2048 \
    --top-budget 1024 \
    --dims 128 128 128 \
    --max-new-tokens 1024 \
    --seq-lengths 4096 8192 16384 32768 \
    --tasks qa_2 \
    --save-dir "evaluate/ruler/results" \
    --postfix _[stage2-v2]_[1024-2048]_[1knew]
