MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20000))

NUM_GPUS=4

RUN_NAME=qwen3-4b
MODEL=$GDRIVE_LOCAL/model/Qwen3-4B
MAX_LENGTH=8192
NUM_LAYERS=36


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node $NUM_GPUS \
    train_stage1.py \
    --num_layers $NUM_LAYERS \
    --max_tokens $MAX_LENGTH \
    --model-name-or-path $MODEL \
    --instance_per_cycle 256 \
    --max_que 256 \
    --max_top 256 \
    --max_oth 256 \
    --top_k 204 \
    --train-iters 256 \
    --max-lr 1e-3 \
    --warmup 0 \
    --weight-decay 0.01 \
    --beta1 0.9 \
    --beta2 0.99 \
    --buffer buffer-${RUN_NAME} \
    --gradient-clipping 1.0 \
    --gradient-accumulation 1 \
    --run-name ${RUN_NAME}_stage1 \
    --train-data data/slimpajama-8k-00000.json



# torchrun \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
#     --nnodes 1 \
#     --nproc_per_node $NUM_GPUS \
#     train_stage2.py \
#     --num_layers $NUM_LAYERS \
#     --max_tokens $MAX_LENGTH \
#     --model-name-or-path $MODEL \
#     --run-name ${RUN_NAME}_stage2 \
#     --batch_size 1 \
#     --train-iters 512 \
#     --top-k 1024 \
#     --chunk-size 4096 \
#     --max-lr 1e-4 \
#     --min-lr 1e-5 \
#     --warmup 0.01 \
#     --weight-decay 0.01 \
#     --beta1 0.9 \
#     --beta2 0.99 \
#     --enable-fsdp2 \
#     --gradient-clipping 1.0 \
#     --checkpoint-dir train_results/$RUN_NAME/stage1 \
#     --train-data data/slimpajama-40k-00000.json
