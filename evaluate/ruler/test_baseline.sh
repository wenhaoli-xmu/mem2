#!/bin/bash

# bash evaluate/ruler/prepare_data.sh "$model" "niah_multikey_3"

python evaluate/ruler/test_baseline.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-30B-A3B \
    --max-new-tokens 4096 \
    --seq-lengths 4096 8192 16384 32768 \
    --tasks niah_multikey_3 \
    --save-dir evaluate/ruler/results
