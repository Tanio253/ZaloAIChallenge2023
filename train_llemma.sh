#!/bin/bash

source env_setup.sh
cd ${BASE_DIR}

MODEL=EleutherAI/llemma_7b
CONFIG=${BASE_DIR}/deepspeed_config.json

OUTDIR=${BASE_DIR}/model/llemma_7b_zalo

NUM_STEPS=9258

deepspeed --num_gpus=1  ${BASE_DIR}/train_math.py \
    --deepspeed ${CONFIG} \
    --model_name_or_path ${MODEL} \
    --data_path ${TRAIN_FILE} \
    --data_length 395000 \
    --output_dir ${OUTDIR} \
    --max_steps ${NUM_STEPS} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps ${NUM_STEPS} \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --logging_dir "$OUTDIR" \
    --report_to="tensorboard" \
    # --bf16 \