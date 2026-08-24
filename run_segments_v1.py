import os
import sys
import torch
import time
import argparse
import numpy as np
from PIL import Image
import re

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from surgery_utils.prompt_builder import build_multi_image_prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", type=int, required=True, help="每组最多取 X 张（均匀采样）")
    parser.add_argument("--model_path", type=str, default="./data/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_dir", type=str, default="./data/uw-sinus-surgery-CL/live/images")
    parser.add_argument("--group_txt", type=str, default="./output/group_segments.txt")
    parser.add_argument("--output_dir", type=str, default="./output/surgery_segments_v1")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--use_revisit", type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=True, help="是否启用 ReVisiT")
    return parser.parse_args()

def read_groups(txt_path):
    groups = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    cur = []
    for line in lines:
        line = line.strip()
        # 跳过空行和配置行
        if not line:
            continue
        if line.startswith("总段数") or line.startswith("最大组帧数") or line.startswith("相似度阈值"):
            continue
        if line.startswith("段 ") and "帧数" in line:
            if cur:
                groups.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        groups.append(cur)
    return groups

def extract_frame_number(filename):
    """从文件名提取帧号，如 L01_30.jpg -> 30"""
    match = re.search(r'_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else 0

def frames_to_time(frame_num, fps=30):
    """将帧号转换为 HH:MM:SS 字符串"""
    total_seconds = frame_num / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def main():
    args = parse_args()
    print(f"V1: 均匀采样，每组最多 {args.X} 张，use_revisit={args.use_revisit}")

    # 加载模型
    print("加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("模型加载完成")

    groups = read_groups(args.group_txt)
    print(f"读取到 {len(groups)} 个语义段")

    os.makedirs(args.output_dir, exist_ok=True)
    all_captions = []
    segment_info = []  # 存储 (start_frame, end_frame, caption)

    for idx, seg_files in enumerate(groups, 1):
        total_frames = len(seg_files)
        # 采样
        if total_frames <= args.X:
            sampled = seg_files
        else:
            indices = np.linspace(0, total_frames-1, args.X, dtype=int)
            sampled = [seg_files[i] for i in indices]

        # 提取首尾帧号
        start_frame = extract_frame_number(sampled[0])
        end_frame = extract_frame_number(sampled[-1])
        time_range = f"{frames_to_time(start_frame)} → {frames_to_time(end_frame)}"

        print(f"\n[{idx}/{len(groups)}] 段 {idx} (原{total_frames}帧，取{len(sampled)}帧)  时间: {time_range}")

        # 生成描述
        seg_paths = [os.path.join(args.image_dir, f) for f in sampled]
        images = [Image.open(p).convert("RGB") for p in seg_paths]
        prompt = build_multi_image_prompt(sampled, "L01", num_frames=len(sampled))

        messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_revisit=args.use_revisit,
            early_exit_layers="last",
            relative_top=1e-5,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=None,
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
        caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print(f"  生成 token 数: {len(generated_ids_trimmed[0])}")
        print(f"  描述: {caption[:80]}..." if len(caption) > 80 else f"  描述: {caption}")

        # 保存单独文件（包含时间）
        seg_file = os.path.join(args.output_dir, f"L01_segment_{idx:03d}.txt")
        with open(seg_file, "w", encoding="utf-8") as f:
            f.write(f"段: {idx}/{len(groups)}\n")
            f.write(f"时间范围: {time_range}\n")
            f.write(f"原帧数: {total_frames}, 采样帧数: {len(sampled)}\n")
            f.write(f"{'='*60}\n")
            f.write(caption)

        all_captions.append(caption)
        segment_info.append((start_frame, end_frame, caption))

        del inputs, output_ids, images, seg_paths
        torch.cuda.empty_cache()
        time.sleep(1)

    # 汇总文件（包含时间）
    summary_file = os.path.join(args.output_dir, "L01_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"总段数: {len(groups)}\n")
        f.write(f"采样策略: 均匀采样, 每段最多 {args.X} 帧\n")
        f.write(f"使用 ReVisiT: {args.use_revisit}\n")
        f.write(f"{'='*60}\n\n")
        for idx, (start, end, cap) in enumerate(segment_info, 1):
            time_str = f"{frames_to_time(start)} → {frames_to_time(end)}"
            f.write(f"[段 {idx}] ({time_str})\n")
            f.write(cap)
            f.write("\n\n" + "="*60 + "\n\n")

    print(f"\n完成！汇总文件: {summary_file}")

if __name__ == "__main__":
    main()