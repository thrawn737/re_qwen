#!/bin/bash

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

VIDEOS="Jan_15 Jan_22 Mar_05 Mar_12"
VIDEO_ROOT="/mnt/data2/Vishal/projects/surgical-video/SurgicalReportGeneration/Vision_pipeline/qwen/FullVideos"
DATA_ROOT="/mnt/data2/Yichen/re_qwen/data"

# ============ B类 分组 + 采样 ============
echo "########################################"
echo "# B类 分组 + 采样"
echo "########################################"
conda activate langbind
export PYTHONPATH=/mnt/sda/Songyc/ReVisiT-main/LanguageBind:$PYTHONPATH

for video in $VIDEOS; do
    echo "========== B类分组: $video =========="
    python group_with_adaptive_b.py \
        --image_dir "${DATA_ROOT}/${video}" \
        --video_path "${VIDEO_ROOT}/${video}/${video}.mp4" \
        --output_root "./output/groups_b/sampled_groups_${video}" \
        --auto_crop \
        --X 1.0 --N 30 --size 256 256 \
        --model_path ./models/LanguageBind_Image \
        --output_txt "./output/groups_b/groups_${video}.txt" \
        --fps 3 \
        --clip_file "./output/clip_${video}.txt" \
        --percentile_a 0.85 --percentile_c 1.0
done

# ============ B类描述 ============
echo "########################################"
echo "# B类 描述生成"
echo "########################################"
conda activate revisit_qwen

for video in $VIDEOS; do
    echo "========== B类描述: $video =========="
    python run_segments_from_groups.py \
        --groups_txt "./output/groups_b/groups_${video}.txt" \
        --sampled_root "./output/groups_b/sampled_groups_${video}" \
        --output_dir "./output/descriptions_b/descriptions_b_${video}" \
        --merged_dir "./output/descriptions_b_merged_parts/descriptions_b_${video}" \
        --model_path "./data/Qwen2.5-VL-3B-Instruct" \
        --use_revisit True \
        --max_new_tokens 512 \
        --surgery_id "L01"
done

# ============ B类 merge ============
echo "########################################"
echo "# B类 merge"
echo "########################################"
conda activate qwen3vl

for video in $VIDEOS; do
    echo "========== B类merge: $video =========="
    python -m merge.merge_descriptions \
        --descriptions_dir "./output/descriptions_b/descriptions_b_${video}" \
        --output_dir "./output/descriptions_b_merged/descriptions_b_merged_${video}" \
        --model_path "./data/Qwen3-VL-8B-Thinking" \
        --threshold 100
done

echo ""
echo "########################################"
echo "# 全部完成"
echo "########################################"