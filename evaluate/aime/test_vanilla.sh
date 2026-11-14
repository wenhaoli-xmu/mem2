#!/bin/bash

python evaluate/aime/test_vanilla.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 \
    --data-path ${GDRIVE_LOCAL}/data/aime-2024/aime-2024.jsonl \
    --apply-chat-template \
    --max-new-tokens 16384 \
    --save-dir evaluate/aime/results
