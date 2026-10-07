# README: Endoscopic Sinus Surgery Video Description Pipeline

## Pipeline Overview

Video → Frame Extraction → Tool Tracking (clip) → Adaptive Grouping + Sampling → Description Generation → Merge


## Environment Setup

### 1. langbind environment
conda create -n langbind python=3.9 -y
conda activate langbind
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/PKU-YuanGroup/LanguageBind.git
cd LanguageBind
pip install -r requirements.txt
pip install einops timm decord scikit-learn Pillow
export PYTHONPATH=/mnt/sda/Songyc/ReVisiT-main/LanguageBind:$PYTHONPATH

### 2. revisit_qwen environment
conda create -n revisit_qwen python=3.9.21
conda activate revisit_qwen
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
conda env update -f prerequisites/ReVisiT.yaml
cd src/transformers-v4.50.0
pip install -e .
cd ../..

### 3. qwen3vl environment
conda create -n qwen3vl python=3.10 -y
conda activate qwen3vl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/transformers
pip install accelerate sentencepiece protobuf pillow pyyaml psutil


## Model Weights

# Qwen2.5-VL-3B-Instruct (for description generation)
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./data/Qwen2.5-VL-3B-Instruct

# Qwen3-VL-8B-Thinking (for merge)
hf download Qwen/Qwen3-VL-8B-Thinking --local-dir ./data/Qwen3-VL-8B-Thinking

# LanguageBind_Image (for visual features)
huggingface-cli download LanguageBind/LanguageBind_Image --local-dir ./models/LanguageBind_Image


## Key Files

### Stage 1: Frame Extraction
slice.py - Extract frames from video at 3 fps, save as frame_XXXXXX.jpg, generate mapping.txt

### Stage 2: Tool Tracking
qwen_tool_tracking.py - Use Qwen2.5-VL to detect surgical tools frame by frame, track consecutive tool segments, output clip_${video}.txt (start/end frame per clip)

### Stage 3: Adaptive Grouping + Sampling
crop_utils.py - Auto-detect left/right black borders, crop to square (side = original height)
group_with_adaptive_a.py - A-class: uniform sampling (N frames per clip), deprecated
group_with_adaptive_b.py - B-class: adaptive sampling (10 chunks per clip, V0/V1/V2 selects 3 frames each), outputs groups_${video}.txt + sampled_groups_${video}/
compute_a_c_from_clips (inside group_with_adaptive_b.py) - Compute thresholds from clip.txt: a = 85th percentile, c = 100th percentile
merge_small_groups (inside group_with_adaptive_b.py) - Merge groups whose span < 1/2 of average span

### Stage 4: Description Generation
surgery_utils/prompt_builder.py - Three prompts: TOOL (instrument appearance), ACTION (manipulation), ANATOMY (before/after manipulation)
run_segments_from_groups.py - Read groups_${video}.txt + sampled images, generate three-part descriptions with Qwen2.5-VL + ReVisiT, save to descriptions_b_${video}/ and merged version to descriptions_b_merged_parts/

### Stage 5: Merge
merge/qwen_text_utils.py - Qwen3-VL loading, generation, thinking-chain stripping, ask_same_combined (anatomy+action joint judgment), merge_two (merge two descriptions)
merge/merge_descriptions.py - Iterate clips, use Qwen3-VL to decide whether adjacent clips should be merged, write results to descriptions_b_merged_combined/

### Stage 6: One-click Scripts
run_all_a_b.sh - Full pipeline: A/B grouping + sampling + description + merge
run_full_pipeline.sh - Full pipeline from raw video (frame extraction → tool tracking → grouping → description → merge)


# 1. Frame extraction
conda activate revisit_qwen
python slice.py \
    --video /path/to/video.mp4 \
    --output /path/to/data/${video} \
    --fps 3 \
    --size 1920 1080

# 2. Tool tracking
python qwen_tool_tracking.py \
    --image_dir /path/to/data/${video} \
    --output_clip ./output/clip_${video}.txt \
    --model_path ./data/Qwen2.5-VL-3B-Instruct \
    --left_crop 0.18 \
    --right_crop 0.25 \
    --target_size 256 256 \
    --has_tool_threshold 0.55 \
    --same_tool_threshold 0.53 \
    --min_clip_length 30 \
    --max_clips 5 \
    --window_size 5

# 3. Adaptive grouping + sampling
conda activate langbind
export PYTHONPATH=/mnt/sda/Songyc/ReVisiT-main/LanguageBind:$PYTHONPATH
## Clipping Modes

There are two clipping modes, controlled by `--limit_clip`:

- **Without `--limit_clip` (default): unconditional clipping.**
  Every time `d > c_value`, it is truncated to `c_value`. There is no cap on how many times this can happen, and consecutive truncations are allowed. This is the behavior of `group_with_adaptive_b2.0.py` and `import os.txt`.

- **With `--limit_clip`: limited clipping.**
  A group can truncate at most `max_clip_count` times (default 5), and two truncations cannot happen on consecutive frames. If `d > c_value` but the group has already used up its truncation budget, or the previous frame was also truncated, `d` is left as is and added to `cum`. This is the behavior of `group_with_adaptive_b3.0.py` and `group_with_adaptive_b.py`.

I cannot run the dataset right now, and I honestly do not remember which clipping mechanism works better in practice. So you need to try both.

### How to run both modes

Run the grouping step twice per video, once without `--limit_clip` and once with `--limit_clip`, and write the results to different output directories so you can compare them.

**Mode A: unconditional clipping**

```bash
python group_with_adaptive_b_no_merge.py \
    --image_dir "${DATA_ROOT}/${video}" \
    --video_path "${VIDEO_ROOT}/${video}/${video}.mp4" \
    --output_root "./output/groups_b_unlimited/sampled_groups_${video}" \
    --auto_crop \
    --X 1.0 --N 30 --size 256 256 \
    --model_path ./models/LanguageBind_Image \
    --output_txt "./output/groups_b_unlimited/groups_${video}.txt" \
    --fps 3 \
    --clip_file "./output/clip_${video}.txt" \
    --percentile_a 0.85 --percentile_c 1

# Mode B: limited clipping

python group_with_adaptive_b_no_merge.py \
    --image_dir "${DATA_ROOT}/${video}" \
    --video_path "${VIDEO_ROOT}/${video}/${video}.mp4" \
    --output_root "./output/groups_b_limited/sampled_groups_${video}" \
    --auto_crop \
    --X 1.0 --N 30 --size 256 256 \
    --model_path ./models/LanguageBind_Image \
    --output_txt "./output/groups_b_limited/groups_${video}.txt" \
    --fps 3 \
    --clip_file "./output/clip_${video}.txt" \
    --percentile_a 0.85 --percentile_c 1 \
    --limit_clip \
    --max_clip_count 5

# 4. Description generation
conda activate revisit_qwen
python run_segments_from_groups.py \
    --groups_txt ./output/groups_b/groups_${video}.txt \
    --sampled_root ./output/groups_b/sampled_groups_${video} \
    --output_dir ./output/descriptions_b/descriptions_b_${video} \
    --merged_dir ./output/descriptions_b_merged_parts/descriptions_b_${video} \
    --model_path ./data/Qwen2.5-VL-3B-Instruct \
    --use_revisit True \
    --max_new_tokens 512 \
    --device cuda:0 \
    --surgery_id L01

# 5. Merge
conda activate qwen3vl
python -m merge.merge_descriptions \
    --descriptions_dir ./output/descriptions_b/descriptions_b_${video} \
    --output_dir ./output/descriptions_b_merged_combined/descriptions_b_merged_combined_${video} \
    --model_path ./data/Qwen3-VL-8B-Thinking \
    --threshold 100 \
    --device cuda:0


## Output Structure

./output/
├── clip_${video}.txt                              # Tool tracking result
├── groups_b/
│   ├── groups_${video}.txt                        # Grouping info
│   └── sampled_groups_${video}/                   # Sampled images
├── descriptions_b/
│   └── descriptions_b_${video}/                   # Three-part descriptions
├── descriptions_b_merged_parts/
│   └── descriptions_b_merged_parts_${video}/      # Merged three-part file
└── descriptions_b_merged_combined/
    └── descriptions_b_merged_combined_${video}/   # Final merged result