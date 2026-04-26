#!/bin/bash

# bash evaluate/ruler/prepare_data.sh ${GDRIVE_LOCAL}/model/Qwen3-4B qa_2
# bash evaluate/ruler/prepare_data.sh ${GDRIVE_LOCAL}/model/Qwen3-4B vt

python evaluate/ruler/test_kernel.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 \
    --checkpoint-path-or-dir train_results/qwen3-4b-thinking-2507/stage2_v6_github.safetensors \
    --max-position-embeddings 262144 \
    --apply-chat-template \
    --lru-budget 8192 \
    --top-budget 4096 \
    --dims 128 128 128 \
    --max-new-tokens 16384 \
    --seq-lengths 4096 8192 16384 32768 65536 131072 245760 \
    --tasks qa_2 \
    --save-dir "evaluate/ruler/results" \
    --postfix _[stage2-v6-github]_[4096-8192]_[16knew]
