#!/bin/bash

python evaluate/longbench_v2/test.py \
    --enable-efficient \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-14B \
    --checkpoint-path-or-dir train_results/qwen3-14b/stage1 \
    --save-dir evaluate/longbench_v2/results \
    --max-len 40960 \
    --max-new-tokens 4096 \
    --lru-budget 2048 \
    --top-budget 2047 \
    --dims 128 128 128 \
    --postfix _[spotlight]
