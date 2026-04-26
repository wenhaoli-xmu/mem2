MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20000))

NUM_GPUS=8

RUN_NAME=qwen3-14b
MODEL=$GDRIVE_LOCAL/model/Qwen3-14B
MAX_LENGTH=40960
NUM_LAYERS=40


# torchrun \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#     --nnodes 1 \
#     --nproc_per_node $NUM_GPUS \
#     train_stage1.py \
#     --num_layers $NUM_LAYERS \
#     --max_tokens $MAX_LENGTH \
#     --model-name-or-path $MODEL \
#     --instance_per_cycle 16 \
#     --max_que 256 \
#     --max_top 256 \
#     --max_oth 256 \
#     --top_k 192 \
#     --train-iters 4096 \
#     --max-lr 1e-3 \
#     --min-lr 1e-4 \
#     --warmup 0 \
#     --weight-decay 0.01 \
#     --beta1 0.9 \
#     --beta2 0.999 \
#     --buffer buffer-${RUN_NAME} \
#     --gradient-clipping 1.0 \
#     --gradient-accumulation 1 \
#     --run-name ${RUN_NAME}_stage1 \
#     --train-data data/github-40k-00000.json


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node $NUM_GPUS \
    train_stage2.py \
    --num_layers $NUM_LAYERS \
    --max_tokens $MAX_LENGTH \
    --model-name-or-path $MODEL \
    --run-name ${RUN_NAME}_stage2 \
    --batch_size 1 \
    --train-iters 1024 \
    --top-k 1024 \
    --chunk-size 4096 \
    --max-lr 1e-4 \
    --min-lr 1e-4 \
    --warmup 0 \
    --weight-decay 0.01 \
    --beta1 0.9 \
    --beta2 0.99 \
    --checkpoint-dir train_results/$RUN_NAME/stage1 \
    --train-data data/slimpajama-40k-00000.json \
    --postfix _v6_slim \
    --save-interval 64 \
    --auto-resume