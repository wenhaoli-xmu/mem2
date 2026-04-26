#!/bin/bash

# bash evaluate/ruler/prepare_data.sh "$model" "niah_multikey_3"

python evaluate/ruler/test_baseline.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 \
    --max-new-tokens 16384 \
    --enable-chat-template \
    --seq-lengths 4096 8192 16384 32768 65536 131072 \
    --tasks qa_2 \
    --save-dir evaluate/ruler/results
