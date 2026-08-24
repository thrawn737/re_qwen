# run_surgery_scenetiling.py
import os
import sys
import argparse
import time

# 添加 src 路径（用于 ReVisiT）
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# ReVisiT 扩展
from utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 加载 LanguageBind 的视觉编码器（本地）
from transformers import AutoImageProcessor, AutoModel

# 你自己的工具
from surgery_utils.data_loader import list_images_by_surgery, get_image_paths
from surgery_utils.prompt_builder import build_multi_image_prompt

# ---------- 配置 ----------
def load_feature_extractor(local_model_path="./models/LanguageBind_Image"):
    """
    从本地加载 LanguageBind 图像编码器
    """
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(
            f"LanguageBind 模型未找到: {local_model_path}\n"
            "请将下载好的模型文件夹放在此路径下。"
        )
    print(f"从本地加载 LanguageBind: {local_model_path}")
    processor = AutoImageProcessor.from_pretrained(local_model_path, local_files_only=True)
    model = AutoModel.from_pretrained(local_model_path, local_files_only=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

def get_cls_token(image_path, feature_processor, feature_model, device):
    """提取 LanguageBind 的 CLS token"""
    image = Image.open(image_path).convert("RGB")
    inputs = feature_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = feature_model(**inputs)
    # LanguageBind 也是 ViT 结构，取 [CLS] token
    cls = outputs.last_hidden_state[:, 0, :]  # (1, 768)
    return cls.cpu().numpy().flatten()

# ---------- SceneTiling ----------
def scene_tiling_by_paths(image_paths, alpha=0.7, feature_processor=None, feature_model=None, device=None):
    """
    输入: image_paths 列表（按时间顺序）
    输出: segments 列表，每个段是一个路径列表
    """
    if len(image_paths) <= 2:
        return [image_paths]

    print("提取帧特征...")
    cls_tokens = []
    for i, p in enumerate(image_paths):
        if i % 20 == 0:
            print(f"  处理 {i+1}/{len(image_paths)}")
        cls_tokens.append(get_cls_token(p, feature_processor, feature_model, device))
    cls_tokens = np.array(cls_tokens)  # (n, 768)

    # 相似度
    similarities = []
    for i in range(len(cls_tokens) - 1):
        sim = cosine_similarity([cls_tokens[i]], [cls_tokens[i+1]])[0][0]
        similarities.append(sim)

    # 深度分数
    depth_scores = []
    for i in range(1, len(similarities) - 1):
        left_max = max(similarities[:i])
        right_max = max(similarities[i+1:])
        depth = (left_max + right_max - 2 * similarities[i]) / 2
        depth_scores.append(depth)

    # 阈值
    mu = np.mean(depth_scores)
    sigma = np.std(depth_scores)
    threshold = mu + alpha * sigma
    print(f"相似度均值: {mu:.4f}, 标准差: {sigma:.4f}, 阈值: {threshold:.4f}")

    # 切分点（局部最大深度且超阈值）
    split_points = []
    for i, d in enumerate(depth_scores):
        left = depth_scores[max(0, i-2):i]
        right = depth_scores[i+1:min(len(depth_scores), i+3)]
        if d > threshold and d > max(left, default=0) and d > max(right, default=0):
            split_points.append(i + 1)

    # 切分
    segments = []
    start = 0
    for p in split_points:
        segments.append(image_paths[start:p])
        start = p
    segments.append(image_paths[start:])

    return segments

# ---------- 生成描述（复用之前逻辑） ----------
def generate_caption_for_segment(segment_paths, segment_idx, surgery_id, model, processor, max_tokens=1024):
    filenames = [os.path.basename(p) for p in segment_paths]
    prompt = build_multi_image_prompt(filenames, surgery_id, num_frames=len(segment_paths))

    images = [Image.open(p).convert("RGB") for p in segment_paths]
    messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        use_revisit=True,
        early_exit_layers="last",
        relative_top=1e-5,
    )
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return caption, len(generated_ids_trimmed[0])

# ---------- 主函数 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surgery_id", type=str, default="L01")
    parser.add_argument("--image_dir", type=str, default="./data/uw-sinus-surgery-CL/live/images")
    parser.add_argument("--output_dir", type=str, default="./output/surgery_segments")
    parser.add_argument("--model_path", type=str, default="./data/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--langbind_path", type=str, default="./models/LanguageBind_Image", help="本地 LanguageBind 模型路径")
    parser.add_argument("--alpha", type=float, default=0.7, help="SceneTiling 阈值系数")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    # ---------- 加载 LVLM ----------
    print("加载 LVLM...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("LVLM 加载完成")

    # ---------- 加载 LanguageBind 视觉编码器（本地） ----------
    feature_processor, feature_model, feat_device = load_feature_extractor(args.langbind_path)

    # ---------- 获取所有图片 ----------
    all_files = list_images_by_surgery(args.image_dir, args.surgery_id, num_samples=None)
    all_paths = get_image_paths(args.image_dir, all_files)
    print(f"总图片数: {len(all_paths)}")

    # ---------- SceneTiling 分割 ----------
    print("运行 SceneTiling...")
    segments = scene_tiling_by_paths(
        all_paths, alpha=args.alpha,
        feature_processor=feature_processor,
        feature_model=feature_model,
        device=feat_device
    )
    print(f"分割为 {len(segments)} 个语义段")
    for i, seg in enumerate(segments):
        print(f"  段 {i+1}: {len(seg)} 帧")

    # ---------- 生成描述 ----------
    os.makedirs(args.output_dir, exist_ok=True)
    all_captions = []

    for idx, seg_paths in enumerate(segments, 1):
        print(f"\n[{idx}/{len(segments)}] 处理段 {idx} ({len(seg_paths)} 帧)...")
        caption, token_len = generate_caption_for_segment(
            seg_paths, idx, args.surgery_id, model, processor, args.max_new_tokens
        )
        print(f"  生成 token 数: {token_len}")
        print(f"  描述: {caption[:80]}..." if len(caption) > 80 else f"  描述: {caption}")

        # 保存单独文件
        seg_file = os.path.join(args.output_dir, f"{args.surgery_id}_segment_{idx:03d}.txt")
        with open(seg_file, "w", encoding="utf-8") as f:
            f.write(f"手术ID: {args.surgery_id}\n")
            f.write(f"段: {idx}/{len(segments)}\n")
            f.write(f"帧数: {len(seg_paths)}\n")
            f.write(f"帧范围: {os.path.basename(seg_paths[0])} → {os.path.basename(seg_paths[-1])}\n")
            f.write(f"{'='*60}\n")
            f.write(caption)

        all_captions.append(caption)

        # 清理显存
        del seg_paths, caption
        torch.cuda.empty_cache()
        time.sleep(1)

    # ---------- 汇总文件 ----------
    summary_file = os.path.join(args.output_dir, f"{args.surgery_id}_segments_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"手术ID: {args.surgery_id}\n")
        f.write(f"总段数: {len(segments)}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"{'='*60}\n\n")
        for idx, cap in enumerate(all_captions, 1):
            f.write(f"[段 {idx}]\n")
            f.write(cap)
            f.write("\n\n" + "="*60 + "\n\n")

    print(f"\n完成！汇总文件: {summary_file}")
    print(f"各段文件保存在: {args.output_dir}")

if __name__ == "__main__":
    main()