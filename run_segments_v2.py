import os
import sys
import torch
import time
import argparse
import re
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from surgery_utils.prompt_builder import build_multi_image_prompt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", type=int, required=True, help="每组最大帧数（超过则切分）")
    parser.add_argument("--model_path", type=str, default="./data/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--image_dir", type=str, default="./data/uw-sinus-surgery-CL/live/images")
    parser.add_argument("--group_txt", type=str, default="./output/group_segments.txt")
    parser.add_argument("--output_dir", type=str, default="./output/surgery_segments_v2")
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
        if line.startswith("段 ") and "帧数" in line:
            if cur:
                groups.append(cur)
                cur = []
        elif line and not line.startswith("总段数") and not line.startswith("最大组帧数"):
            cur.append(line)
    if cur:
        groups.append(cur)
    return groups

def extract_frame_number(filename):
    match = re.search(r'_(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else 0

def frames_to_time(frame_num, fps=30):
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
    print(f"V2: 按最大帧数 {args.X} 切分，use_revisit={args.use_revisit}")

    # 加载模型
    print("加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("模型加载完成")

    groups = read_groups(args.group_txt)
    print(f"读取到 {len(groups)} 个原始语义段")

    os.makedirs(args.output_dir, exist_ok=True)
    sub_segments = []  # 每个元素 (files, orig_idx, orig_total, start_frame, end_frame)

    # 切分每个原始段
    for idx, seg_files in enumerate(groups, 1):
        total = len(seg_files)
        for start in range(0, total, args.X):
            end = min(start + args.X, total)
            sub = seg_files[start:end]
            # 提取首尾帧号
            start_frame = extract_frame_number(sub[0])
            end_frame = extract_frame_number(sub[-1])
            sub_segments.append((sub, idx, total, start_frame, end_frame))

    print(f"切分后共 {len(sub_segments)} 个子段")

    all_captions = []
    sub_info = []  # 存储 (sub_idx, start_frame, end_frame, caption)

    for sub_idx, (files, orig_idx, orig_total, start_frame, end_frame) in enumerate(sub_segments, 1):
        time_range = f"{frames_to_time(start_frame)} → {frames_to_time(end_frame)}"
        print(f"\n[{sub_idx}/{len(sub_segments)}] 子段 (来自原始段 {orig_idx}, 原{orig_total}帧, 取{len(files)}帧)  时间: {time_range}")

        seg_paths = [os.path.join(args.image_dir, f) for f in files]
        images = [Image.open(p).convert("RGB") for p in seg_paths]
        prompt = build_multi_image_prompt(files, "L01", num_frames=len(files))

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

        # 保存单独文件
        seg_file = os.path.join(args.output_dir, f"L01_subsegment_{sub_idx:03d}.txt")
        with open(seg_file, "w", encoding="utf-8") as f:
            f.write(f"子段: {sub_idx}/{len(sub_segments)}\n")
            f.write(f"时间范围: {time_range}\n")
            f.write(f"来源原始段: {orig_idx}, 原始帧数: {orig_total}, 本子段帧数: {len(files)}\n")
            f.write(f"{'='*60}\n")
            f.write(caption)

        all_captions.append(caption)
        sub_info.append((sub_idx, start_frame, end_frame, caption))

        del inputs, output_ids, images, seg_paths
        torch.cuda.empty_cache()
        time.sleep(1)

    # 汇总文件
    summary_file = os.path.join(args.output_dir, "L01_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"子段总数: {len(sub_segments)}\n")
        f.write(f"切分策略: 每段最多 {args.X} 帧\n")
        f.write(f"使用 ReVisiT: {args.use_revisit}\n")
        f.write(f"{'='*60}\n\n")
        for sub_idx, start, end, cap in sub_info:
            time_str = f"{frames_to_time(start)} → {frames_to_time(end)}"
            f.write(f"[子段 {sub_idx}] ({time_str})\n")
            f.write(cap)
            f.write("\n\n" + "="*60 + "\n\n")

    print(f"\n完成！汇总文件: {summary_file}")

if __name__ == "__main__":
    main()