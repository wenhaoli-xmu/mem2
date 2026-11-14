#!/bin/bash

python evaluate/longbench_v2/test.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-14B \
    --save-dir evaluate/longbench_v2/results \
    --max-len 40960 \
    --max-new-tokens 4096
