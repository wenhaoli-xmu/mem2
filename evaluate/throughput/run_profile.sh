export SPOTLIGHT_PG19_PATH=data/pg19.json

python evaluate/throughput/run_profile.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 \
    --method eval \
    --checkpoint-dir ${GDRIVE_LOCAL}/project/spotlight/train_results/qwen3-4b-thinking-2507/stage2_v6_github.safetensors \
    --batch-size 32 \
    --prefill-tokens 262144 \
    --decode-tokens 128 \
    --lru-budget 2048 \
    --top-budget 1024
