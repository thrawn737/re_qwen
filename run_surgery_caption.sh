#!/bin/bash
# ============================================================
# 强制使用 GPU 1
# ============================================================
# export CUDA_VISIBLE_DEVICES=1
# unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE

# ============================================================
# 手术多图描述生成 - 启动脚本
# ============================================================

# 基本配置
IMAGE_DIR="./data/uw-sinus-surgery-CL/live/images"          # 图片目录
OUTPUT_DIR="./output/surgery_captions"              # 输出目录
MODEL_PATH="./data/Qwen2.5-VL-7B-Instruct"          # 模型路径

# 采样配置
NUM_FRAMES=20                                       # 采样帧数

# ReVisiT 配置
USE_REVISIT=True                                    # 是否启用 ReVisiT
RELATIVE_TOP=1e-5                                   # ReVisiT 相对阈值
EARLY_EXIT_LAYERS="all"                             # early exit layers

# 生成配置
MAX_NEW_TOKENS=512
DO_SAMPLE=False

# ============================================================
# 运行模式
# ============================================================

# 模式1: 单个手术
MODE="single"
SURGERY_ID="L01"

# 模式2: 批量所有手术（注释掉上面的，取消注释下面的）
# MODE="batch"
# SURGERY_ID=""

# ============================================================
# 执行
# ============================================================

python run_surgery_caption.py \
    --mode ${MODE} \
    --surgery_id ${SURGERY_ID} \
    --image_dir ${IMAGE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_PATH} \
    --num_frames ${NUM_FRAMES} \
    --use_revisit ${USE_REVISIT} \
    --relative_top ${RELATIVE_TOP} \
    --early_exit_layers ${EARLY_EXIT_LAYERS} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --do_sample ${DO_SAMPLE}