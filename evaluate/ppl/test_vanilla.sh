#!/bin/bash

export SPOTLIGHT_PG19_PATH=data/pg19.json
export SPOTLIGHT_CODEPARROT_PATH=data/codeparrot.json
export SPOTLIGHT_PROOFPILE_PATH=data/proof-pile.json

python evaluate/ppl/test_vanilla.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507
