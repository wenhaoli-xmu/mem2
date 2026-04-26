export SPOTLIGHT_PG19_PATH=data/pg19.json

# 1. Run benchmark -> heatmap.csv
python evaluate/throughput/test_heatmap.py \
    --model-name-or-path ${GDRIVE_LOCAL}/model/Qwen3-4B-Thinking-2507 \
    --method eval \
    --checkpoint-dir ${GDRIVE_LOCAL}/project/spotlight/train_results/qwen3-4b-thinking-2507/stage2_v6_github.safetensors \
    --lru-budget 8192 \
    --top-budget 4096

# 2. Draw heatmap from CSV
# python evaluate/throughput/draw_heatmap.py
