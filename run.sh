#!/bin/bash

echo "All parameters: $@"
sudo apt-get update
sudo apt-get install -y ffmpeg

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT
pip install -r ./requirements.txt

WHISPER_PATH=/mnt/bn/tiktok-mm-4/aiic/public/model/whisper-base
IMAGE_MODEL_PATH=/mnt/bn/tiktok-mm-4/aiic/public/model/donut-base-finetuned-cord-v2
TRAIN_DATA=/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_train_cut.json
EVAL_DATA=/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_val_cut.json

EPOCHS=20
TRAIN_BS=32
EVAL_BS=8
LR=1e-4
SEED=2025

SAVE_STEPS=500
OUTPUT_NAME=debug

TEST_DATA=/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/Donut_Whisper/jsons/how2_val.json
DO_TEST=False
CKPT=None

MODEL_TYPE=donut_whisper
TRAIN_ENCODER=False

EVAL_ACCUMULATION_STEPS=-1

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --whisper_path) WHISPER_PATH="$2"; shift ;;
        --image_model_path) IMAGE_MODEL_PATH="$2"; shift ;;
        --train_data) TRAIN_DATA="$2"; shift ;;
        --eval_data) EVAL_DATA="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --train_bs) TRAIN_BS="$2"; shift ;;
        --eval_bs) EVAL_BS="$2"; shift ;;
        --lr) LR="$2"; shift ;;
        --seed) SEED="$2"; shift ;;
        --save_steps) SAVE_STEPS="$2"; shift ;;
        --output_name) OUTPUT_NAME="$2"; shift ;;
        --test_data) TEST_DATA="$2"; shift ;;
        --do_test) DO_TEST=True; ;;
        --ckpt) CKPT="$2"; shift ;;
        --model_type) MODEL_TYPE="$2"; shift ;;
        --train_encoder) TRAIN_ENCODER="$2"; shift ;;
        --eval_accumulation_steps) EVAL_ACCUMULATION_STEPS="$2"; shift ;;
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

if [ "$DO_TEST" = "True" ]; then
    DEEPSPEED=None
    OUTPUT_DIR=output/test/$OUTPUT_NAME
else
    DEEPSPEED=scripts/zero0.json
    OUTPUT_DIR=output/$OUTPUT_NAME
fi

export WANDB_PROJECT=asr_ocr_whisper
export WANDB_NAME=$OUTPUT_DIR
wandb online

DEEPSPEED_ARGS=""
[ "$DEEPSPEED" != "None" ] && DEEPSPEED_ARGS="--deepspeed $DEEPSPEED"

EVAL_ACCUMULATION_STEPS_ARGS=""
if [ "$EVAL_ACCUMULATION_STEPS" -ge 0 ]; then
    EVAL_ACCUMULATION_STEPS_ARGS="--eval_accumulation_steps $EVAL_ACCUMULATION_STEPS"
fi

# ${ARNOLD_WORKER_GPU}
torchrun --nproc_per_node=${ARNOLD_WORKER_GPU} --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port=12396 \
    train.py \
        $DEEPSPEED_ARGS \
        $EVAL_ACCUMULATION_STEPS_ARGS \
        --model_type $MODEL_TYPE \
        --whisper_path $WHISPER_PATH \
        --image_model_path $IMAGE_MODEL_PATH \
        --train_data $TRAIN_DATA \
        --eval_data $EVAL_DATA \
        --seed $SEED \
        --num_train_epochs $EPOCHS \
        --per_device_train_batch_size $TRAIN_BS \
        --per_device_eval_batch_size $EVAL_BS \
        --evaluation_strategy "steps" \
        --save_total_limit 100 \
        --learning_rate $LR \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --dataloader_num_workers 16 \
        --eval_steps $SAVE_STEPS \
        --save_steps $SAVE_STEPS \
        --output_dir $OUTPUT_DIR \
        --remove_unused_columns False \
        --fp16 True \
        --test_data $TEST_DATA \
        --do_test $DO_TEST \
        --ckpt $CKPT \
        --train_encoder $TRAIN_ENCODER;
