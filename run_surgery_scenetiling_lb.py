import os
import sys
import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LanguageBind'))
from languagebind import LanguageBindImage, LanguageBindImageTokenizer, LanguageBindImageProcessor

# ===== 解析命令行参数 =====
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.85, help="相似度阈值，越高分组越细（0.5-1.0）")
parser.add_argument("--surgery_id", type=str, default="L01")
parser.add_argument("--image_dir", type=str, default="./data/uw-sinus-surgery-CL/live/images")
parser.add_argument("--output_txt", type=str, default="./output/group_segments.txt")
parser.add_argument("--model_path", type=str, default="./models/LanguageBind_Image")
args = parser.parse_args()

similarity_threshold = args.threshold
image_dir = args.image_dir
surgery_id = args.surgery_id
output_txt = args.output_txt
model_path = args.model_path

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# ===== 加载模型 =====
print(f"加载 LanguageBind (阈值: {similarity_threshold})...")
model = LanguageBindImage.from_pretrained(model_path, cache_dir='./cache_dir')
tokenizer = LanguageBindImageTokenizer.from_pretrained(model_path, cache_dir='./cache_dir')
processor = LanguageBindImageProcessor(model.config, tokenizer)
model.eval()
model.to(device)

# ===== 提取帧号 =====
def extract_frame_number(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else 0

# ===== 获取所有图片，按帧号排序 =====
all_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') and f.startswith(surgery_id)]
all_files.sort(key=extract_frame_number)
all_paths = [os.path.join(image_dir, f) for f in all_files]
print(f"总图片数: {len(all_paths)}")
if all_files:
    print(f"首帧: {extract_frame_number(all_files[0])}, 末帧: {extract_frame_number(all_files[-1])}")

# ===== 提取特征 =====
def get_feature(img_path):
    data = processor([img_path], [''], return_tensors='pt')
    data = {k: v.to(device) if hasattr(v, 'to') else v for k, v in data.items()}
    with torch.no_grad():
        out = model(**data)
        feat = out.image_embeds
    return feat.cpu().numpy().flatten()

print("提取帧特征...")
features = []
for i, p in enumerate(all_paths):
    if i % 20 == 0:
        print(f"  处理 {i+1}/{len(all_paths)}")
    features.append(get_feature(p))

# ===== 递归分组 =====
def recursive_group(start_idx, paths, feats, threshold):
    if start_idx >= len(paths):
        return [], start_idx

    current_group = [paths[start_idx]]
    idx = start_idx + 1

    while idx < len(paths):
        sim = cosine_similarity([feats[start_idx]], [feats[idx]])[0][0]
        if sim >= threshold:
            current_group.append(paths[idx])
            idx += 1
        else:
            break

    rest_groups, next_start = recursive_group(idx, paths, feats, threshold)
    return [current_group] + rest_groups, next_start

print("按顺序递归分组...")
if not features:
    print("没有图片")
    sys.exit(0)

groups, _ = recursive_group(0, all_paths, features, similarity_threshold)
print(f"分组完成，共 {len(groups)} 组")
max_frames = max(len(g) for g in groups) if groups else 0
print(f"最大组帧数: {max_frames}")

# ===== 保存结果 =====
os.makedirs(os.path.dirname(output_txt), exist_ok=True)
with open(output_txt, 'w') as f:
    f.write(f"总段数: {len(groups)}\n")
    f.write(f"最大组帧数: {max_frames}\n")
    f.write(f"相似度阈值: {similarity_threshold}\n")
    for idx, g in enumerate(groups):
        f.write(f"\n段 {idx+1} (帧数: {len(g)}):\n")
        for p in g:
            f.write(f"  {os.path.basename(p)}\n")

print(f"分组结果已保存到: {output_txt}")