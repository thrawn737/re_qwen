# run_surgery_scenetiling_qwen.py
import os
import sys
import argparse
import time
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from surgery_utils.data_loader import list_images_by_surgery, get_image_paths
from surgery_utils.prompt_builder import build_multi_image_prompt

def get_frame_feature(image_path, visual_encoder, processor, device):
    """提取单帧特征（Qwen 视觉编码器）"""
    image = Image.open(image_path).convert("RGB")
    # 直接使用 processor 处理单张图像，会自动生成正确的 grid_thw
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    # 从 inputs 中获取 grid_thw，这是由 processor 自动计算出来的
    grid_thw = inputs.grid_thw.to(device)
    with torch.no_grad():
        outputs = visual_encoder(pixel_values, grid_thw=grid_thw)
        # 取平均池化
        features = outputs.mean(dim=1)
    return features.cpu().numpy().flatten()

def scene_tiling_by_paths(image_paths, visual_encoder, processor, device, alpha=0.7):
    if len(image_paths) <= 2:
        return [image_paths]

    print("提取帧特征...")
    cls_tokens = []
    for i, p in enumerate(image_paths):
        if i % 20 == 0:
            print(f"  处理 {i+1}/{len(image_paths)}")
        try:
            feat = get_frame_feature(p, visual_encoder, processor, device)
            cls_tokens.append(feat)
        except Exception as e:
            print(f"  警告: 处理 {p} 时出错: {e}")
            # 出错时用前一帧特征填充，确保不中断
            if cls_tokens:
                cls_tokens.append(cls_tokens[-1])
            else:
                cls_tokens.append(np.zeros(768))
    
    cls_tokens = np.array(cls_tokens)

    similarities = []
    for i in range(len(cls_tokens) - 1):
        sim = cosine_similarity([cls_tokens[i]], [cls_tokens[i+1]])[0][0]
        similarities.append(sim)

    depth_scores = []
    for i in range(1, len(similarities) - 1):
        left_max = max(similarities[:i])
        right_max = max(similarities[i+1:])
        depth = (left_max + right_max - 2 * similarities[i]) / 2
        depth_scores.append(depth)

    if not depth_scores:
        return [image_paths]

    mu = np.mean(depth_scores)
    sigma = np.std(depth_scores)
    threshold = mu + alpha * sigma
    print(f"相似度均值: {mu:.4f}, 标准差: {sigma:.4f}, 阈值: {threshold:.4f}")

    split_points = []
    for i, d in enumerate(depth_scores):
        left = depth_scores[max(0, i-2):i]
        right = depth_scores[i+1:min(len(depth_scores), i+3)]
        if d > threshold and d > max(left, default=0) and d > max(right, default=0):
            split_points.append(i + 1)

    segments = []
    start = 0
    for p in split_points:
        segments.append(image_paths[start:p])
        start = p
    segments.append(image_paths[start:])
    return segments

def generate_caption_for_segment(segment_paths, segment_idx, surgery_id, qwen_model, processor, max_tokens=1024):
    filenames = [os.path.basename(p) for p in segment_paths]
    prompt = build_multi_image_prompt(filenames, surgery_id, num_frames=len(segment_paths))
    images = [Image.open(p).convert("RGB") for p in segment_paths]
    messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(qwen_model.device)
    output_ids = qwen_model.generate(
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--surgery_id", type=str, default="L01")
    parser.add_argument("--image_dir", type=str, default="./data/uw-sinus-surgery-CL/live/images")
    parser.add_argument("--output_dir", type=str, default="./output/surgery_segments")
    parser.add_argument("--model_path", type=str, default="./data/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    print("加载 LVLM...")
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("LVLM 加载完成")

    visual_encoder = qwen_model.visual
    visual_encoder.eval()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    visual_encoder.to(device)

    all_files = list_images_by_surgery(args.image_dir, args.surgery_id, num_samples=None)
    all_paths = get_image_paths(args.image_dir, all_files)
    print(f"总图片数: {len(all_paths)}")

    print("运行 SceneTiling...")
    segments = scene_tiling_by_paths(all_paths, visual_encoder, processor, device, alpha=args.alpha)
    print(f"分割为 {len(segments)} 个语义段")
    for i, seg in enumerate(segments):
        print(f"  段 {i+1}: {len(seg)} 帧")

    os.makedirs(args.output_dir, exist_ok=True)
    all_captions = []
    for idx, seg_paths in enumerate(segments, 1):
        print(f"\n[{idx}/{len(segments)}] 处理段 {idx} ({len(seg_paths)} 帧)...")
        caption, token_len = generate_caption_for_segment(
            seg_paths, idx, args.surgery_id, qwen_model, processor, args.max_new_tokens
        )
        print(f"  生成 token 数: {token_len}")
        print(f"  描述: {caption[:80]}..." if len(caption) > 80 else f"  描述: {caption}")

        seg_file = os.path.join(args.output_dir, f"{args.surgery_id}_segment_{idx:03d}.txt")
        with open(seg_file, "w", encoding="utf-8") as f:
            f.write(f"手术ID: {args.surgery_id}\n")
            f.write(f"段: {idx}/{len(segments)}\n")
            f.write(f"帧数: {len(seg_paths)}\n")
            f.write(f"帧范围: {os.path.basename(seg_paths[0])} → {os.path.basename(seg_paths[-1])}\n")
            f.write(f"{'='*60}\n")
            f.write(caption)
        all_captions.append(caption)

        del seg_paths, caption
        torch.cuda.empty_cache()
        time.sleep(1)

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