#!/bin/bash

SURGERY_ID="L01"
IMAGE_DIR="./data/uw-sinus-surgery-CL/live/images"
OUTPUT_DIR="./output/surgery_captions_noR"
WINDOW_SIZE=20

TOTAL=$(python -c "
from surgery_utils.data_loader import list_images_by_surgery
print(len(list_images_by_surgery('$IMAGE_DIR', '$SURGERY_ID', num_samples=None)))
")
echo "总图片数: $TOTAL"

WINDOWS=$(( (TOTAL + WINDOW_SIZE - 1) / WINDOW_SIZE ))
echo "总窗口数: $WINDOWS"

for ((i=0; i<$WINDOWS; i++)); do
    START=$((i * WINDOW_SIZE))
    END=$((START + WINDOW_SIZE))
    if [ $END -gt $TOTAL ]; then
        END=$TOTAL
    fi
    
    echo ""
    echo "========================================"
    echo "窗口 $((i+1))/$WINDOWS: 索引 $START → $((END-1))"
    echo "========================================"
    
    OUTPUT_FILE="${OUTPUT_DIR}/${SURGERY_ID}_window_$(printf "%04d" $START)-$(printf "%04d" $((END-1))).txt"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "文件已存在，跳过: $OUTPUT_FILE"
        continue
    fi
    
    python test_single_window.py \
        --start $START \
        --end $END \
        --surgery_id $SURGERY_ID \
        --image_dir $IMAGE_DIR \
        --output_dir $OUTPUT_DIR
    
    sleep 3
done

echo ""
echo "所有窗口处理完成！"