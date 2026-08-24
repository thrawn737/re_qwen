# README: Endoscopic Sinus Surgery Video Description Pipeline

## Environment Setup

### 1. langbind environment
conda create -n langbind python=3.9 -y
conda activate langbind
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install git+https://github.com/PKU-YuanGroup/LanguageBind.git
pip install einops timm decord scikit-learn Pillow

### 2. revisit_qwen environment
conda create -n revisit_qwen python=3.9.21
conda activate revisit_qwen
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
conda env update -f prerequisites/ReVisiT.yaml
cd src/transformers-v4.50.0
pip install -e .
cd ../..

## Download Model Weights

### 1. Qwen2.5-VL-7B-Instruct
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./data/Qwen2.5-VL-7B-Instruct

### 2. LanguageBind_Image
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download LanguageBind/LanguageBind_Image --local-dir ./models/LanguageBind_Image

## Running the Pipeline

### Step 1: Segmentation
conda activate langbind
export PYTHONPATH=/mnt/sda/Songyc/ReVisiT-main/LanguageBind:
python run_surgery_scenetiling_lb.py --threshold 0.85

### Step 2: Caption Generation (4 variants)
conda activate revisit_qwen

python run_segments_v1.py --X 5 --use_revisit True --output_dir ./output/surgery_segments_v1_revisit
python run_segments_v1.py --X 5 --use_revisit False --output_dir ./output/surgery_segments_v1_noR
python run_segments_v2.py --X 5 --use_revisit True --output_dir ./output/surgery_segments_v2_revisit
python run_segments_v2.py --X 5 --use_revisit False --output_dir ./output/surgery_segments_v2_noR

### Step 3: Consolidation
python consolidate.py --input ./output/surgery_segments_v1_revisit/L01_summary.txt --output ./output/L01_summary_v1_revisit_consolidated.txt
python consolidate.py --input ./output/surgery_segments_v1_noR/L01_summary.txt --output ./output/L01_summary_v1_noR_consolidated.txt
python consolidate.py --input ./output/surgery_segments_v2_revisit/L01_summary.txt --output ./output/L01_summary_v2_revisit_consolidated.txt
python consolidate.py --input ./output/surgery_segments_v2_noR/L01_summary.txt --output ./output/L01_summary_v2_noR_consolidated.txt
