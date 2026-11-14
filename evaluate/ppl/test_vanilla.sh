#!/bin/bash

export ROUTECACHE_PG19_PATH=data/pg19.json
export ROUTECACHE_CODEPARROT_PATH=data/codeparrot.json
export ROUTECACHE_PROOFPILE_PATH=data/proof-pile.json

python evaluate/ppl/test_vanilla.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507
