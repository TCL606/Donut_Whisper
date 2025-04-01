#!/bin/bash

echo "All parameters: $@"
sudo apt-get update
sudo apt-get install -y ffmpeg

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT
pip install -r ./requirements.txt

WHISPER_PATH=/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base
IMAGE_MODEL_PATH=/mnt/bn/tiktok-mm-4/aiic/public/model/donut-base-finetuned-cord-v2
TRAIN_DATA=/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_train.json
EVAL_DATA=/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_val.json

EPOCHS=5
TRAIN_BS=8
LR=1e-5
SEED=2025

SAVE_STEPS=1000
OUTPUT_DIR=output/debug

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --whisper_path) WHISPER_PATH="$2"; shift ;;
        --image_model_path) IMAGE_MODEL_PATH="$2"; shift ;;
        --train_data) TRAIN_DATA="$2"; shift ;;
        --eval_data) EVAL_DATA="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --train_bs) TRAIN_BS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export HF_HOME="/mnt/bn/tiktok-mm-2/aiic/public/model/huggingface"
# if [ -e "/mnt/bn/tiktok-mm-2/aiic/public/model/huggingface" ]; then
#     export HF_HOME="/mnt/bn/tiktok-mm-2/aiic/public/model/huggingface"
# else
#     hdfs dfs get hdfs://harunava/home/byte_data_seed_azureb_tteng/user/tangchangli/huggingface /home/tiger/.cache/
# fi

# ${ARNOLD_WORKER_GPU}
torchrun --nproc_per_node=1 --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port=12396 \
    train.py \
        --whisper_path $WHISPER_PATH \
        --image_model_path $IMAGE_MODEL_PATH \
        --train_data $TRAIN_DATA \
        --eval_data $EVAL_DATA \
        --seed $SEED \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BS \
        --per_device_eval_batch_size $TRAIN_BS \
        --evaluation_strategy "steps" \
        --save_total_limit 100 \
        --learning_rate $LR \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --dataloader_num_workers 8 \
        --eval_steps $SAVE_STEPS \
        --save_steps $SAVE_STEPS \
        --output_dir $OUTPUT_DIR \
        --remove_unused_columns False \
        --fp16 False