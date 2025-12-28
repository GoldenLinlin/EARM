#!/bin/bash

# ============================================
# EARM 训练脚本
# ============================================
# 路径配置
TRAIN_DATA="/home/work/EARM/outputs/earm_train_data_eng.json"
VAL_DATA="/home/work/EARM/outputs/earm_valid_data_eng.json"

MODEL_NAME="/home/work/LLaMA-Factory/output/eng_stage2_qwen7b"
OUTPUT_DIR="./earm_output_ENG_3loss_weight=1,MARGIN0.,LAMBDA_WEIGHTED0._sft"

# 模型配置
USE_LORA=false
LORA_R=16
LORA_ALPHA=32

# 编辑权重配置（来自 EPO-GEC）
PIVOT_TOKEN_WEIGHT=1.0
EDIT_TOKEN_WEIGHT=1.0
NORMAL_TOKEN_WEIGHT=1.0

# 损失函数配置
LAMBDA_WEIGHTED=0.0
MU_MARGIN=0.0
MARGIN=0.0

# 训练配置
BATCH_SIZE=16
GRADIENT_ACCUMULATION=8
LEARNING_RATE=1e-5
NUM_EPOCHS=2
MAX_LENGTH=1024
config_file="./default_config_deepspeed.yaml"
# ============================================
# 执行训练
# ============================================

CMD="accelerate launch --num_processes 8 --config_file ${config_file} train_earm.py \
    --train_data ${TRAIN_DATA} \
    --val_data   ${VAL_DATA} \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --max_length ${MAX_LENGTH} \
    --pivot_token_weight ${PIVOT_TOKEN_WEIGHT} \
    --edit_token_weight ${EDIT_TOKEN_WEIGHT} \
    --normal_token_weight ${NORMAL_TOKEN_WEIGHT} \
    --lambda_weighted ${LAMBDA_WEIGHTED} \
    --mu_margin ${MU_MARGIN} \
    --margin ${MARGIN} \
    --batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --learning_rate ${LEARNING_RATE} \
    --num_epochs ${NUM_EPOCHS} \
    --save_steps 1000 \
    --eval_steps 50 \
    --logging_steps 1 \
    --use_wandb"

if [ "$USE_LORA" = true ]; then
    CMD="${CMD} --use_lora --lora_r ${LORA_R} --lora_alpha ${LORA_ALPHA}"
fi

echo "============================================"
echo "开始 EARM 训练"
echo "============================================"
echo "训练数据: ${TRAIN_DATA}"
echo "模型: ${MODEL_NAME}"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================"
echo ""

eval ${CMD}