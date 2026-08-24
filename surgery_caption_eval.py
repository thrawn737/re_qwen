"""
手术多图描述生成模块
基于 Qwen2.5-VL + ReVisiT，支持多图输入
复用 chair_eval_qwenvl.py 的模型加载和生成方式
"""

import os
import sys
import json
import argparse
import asyncio
from typing import List, Optional, Dict
from PIL import Image

import torch
import torch.distributed as dist
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 添加 src 路径（复用 ReVisiT 的 utils）
revisit_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(revisit_path, 'src'))
from utils import setup_dist, terminate_dist, str2bool

# 导入我们的工具模块
from surgery_utils.data_loader import list_images_by_surgery, get_image_paths, get_surgery_info
from surgery_utils.prompt_builder import build_multi_image_prompt

import warnings
warnings.filterwarnings(action='ignore')

import aiofiles


def parse_args():
    parser = argparse.ArgumentParser(description="Surgery Multi-Image Captioning with ReVisiT")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="./data/Qwen2.5-VL-7B-Instruct", help="模型路径")
    parser.add_argument("--model_base", type=str, default="qwenvl")
    
    # 数据参数
    parser.add_argument("--surgery_id", type=str, required=True, help="手术ID，如 L01, S01")
    parser.add_argument("--image_dir", type=str, required=True, help="图片目录路径")
    parser.add_argument("--num_frames", type=int, default=20, help="采样帧数")
    parser.add_argument("--output_dir", type=str, default="./output/surgery_captions", help="输出目录")
    
    # 生成参数
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    
    # ReVisiT 参数
    parser.add_argument("--use_revisit", type=str2bool, default=True, help="是否启用 ReVisiT")
    parser.add_argument("--early_exit_layers", type=str, default="all", choices=["last", "all"])
    parser.add_argument("--relative_top", type=float, default=1e-5)
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    
    args = parser.parse_known_args()[0]
    return args


def load_model(model_path: str):
    """
    加载 Qwen2.5-VL 模型和处理器
    与 chair_eval_qwenvl.py 保持一致
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cuda:1",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


async def process_surgery(
    model,
    processor,
    image_dir: str,
    surgery_id: str,
    num_frames: int = 20,
    max_new_tokens: int = 512,
    use_revisit: bool = True,
    early_exit_layers: str = "all",
    relative_top: float = 1e-5,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    output_dir: str = "./output/surgery_captions",
    seed: int = 42,
) -> Dict:
    """
    处理一个手术的所有图片，生成描述
    """
    # 设置随机种子
    setup_dist(seed, 1)
    
    # 1. 加载图片列表
    print(f"\n[{surgery_id}] 开始处理...")
    filenames = list_images_by_surgery(image_dir, surgery_id, num_samples=num_frames)
    image_paths = get_image_paths(image_dir, filenames)
    
    print(f"  加载了 {len(image_paths)} 张图片")
    print(f"  首帧: {filenames[0]}, 末帧: {filenames[-1]}")
    
    # 2. 构建 Prompt（多图格式）
    prompt = build_multi_image_prompt(filenames, surgery_id, num_frames=num_frames)
    print(f"  Prompt 长度: {len(prompt)} 字符")
    
    # 3. 先加载所有图片
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    # 4. 为每张图片生成一个占位符
    image_placeholders = [{"type": "image"}] * len(images)
    
    # 5. 构建 messages
    messages = [
        {
            "role": "user",
            "content": image_placeholders + [{"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 6. 处理输入
    inputs = processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # 7. 生成
    print(f"  开始生成（use_revisit={use_revisit}）...")
    with torch.inference_mode():
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                use_revisit=use_revisit,
                early_exit_layers=early_exit_layers,
                relative_top=relative_top,
            )
    
    # 8. 解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # 9. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "surgery_id": surgery_id,
        "num_frames": num_frames,
        "image_files": filenames,
        "prompt": prompt,
        "caption": caption,
        "use_revisit": use_revisit,
        "early_exit_layers": early_exit_layers,
        "relative_top": relative_top,
        "max_new_tokens": max_new_tokens,
    }
    
    output_file = os.path.join(output_dir, f"{surgery_id}_caption.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 同时保存一份纯文本版本
    text_file = os.path.join(output_dir, f"{surgery_id}_caption.txt")
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"Surgery: {surgery_id}\n")
        f.write(f"Num Frames: {num_frames}\n")
        f.write(f"Use ReVisiT: {use_revisit}\n")
        f.write(f"{'='*60}\n")
        f.write(caption)
    
    print(f"  结果已保存到 {output_file}")
    print(f"  文本版本: {text_file}")
    
    # 打印描述
    print(f"\n{'='*60}")
    print(f"[{surgery_id}] 生成描述:")
    print(f"{'='*60}")
    print(caption)
    print(f"{'='*60}\n")
    
    terminate_dist()
    return result


def main():
    args = parse_args()
    print("Args:", args)
    
    # 加载模型
    print("加载模型中...")
    model, processor = load_model(args.model_path)
    print("模型加载完成")
    
    # 处理手术
    asyncio.run(process_surgery(
        model=model,
        processor=processor,
        image_dir=args.image_dir,
        surgery_id=args.surgery_id,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        use_revisit=args.use_revisit,
        early_exit_layers=args.early_exit_layers,
        relative_top=args.relative_top,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        output_dir=args.output_dir,
        seed=args.seed,
    ))

async def process_surgery_sliding(
    model,
    processor,
    image_dir: str,
    surgery_id: str,
    window_size: int = 20,
    step: int = 20,
    max_new_tokens: int = 512,
    use_revisit: bool = True,
    early_exit_layers: str = "all",
    relative_top: float = 1e-5,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.0,
    output_dir: str = "./output/surgery_captions",
    seed: int = 42,
) -> Dict:
    """
    滑动窗口模式：每 window_size 张图生成一段描述，最后拼接成完整报告
    """
    from surgery_utils.data_loader import get_sliding_windows_with_info
    from surgery_utils.prompt_builder import build_multi_image_prompt
    
    setup_dist(seed, 1)
    
    print(f"\n[{surgery_id}] 开始处理（滑动窗口模式）...")
    
    windows = get_sliding_windows_with_info(
        image_dir, surgery_id, 
        window_size=window_size, 
        step=step
    )
    
    if not windows:
        print(f"  错误: 未找到任何图片")
        terminate_dist()
        return {}
    
    print(f"  总共 {len(windows)} 个窗口，每窗口 {window_size} 张图")
    
    all_captions = []
    window_details = []
    
    for idx, window_info in enumerate(windows, 1):
        filenames = window_info["filenames"]
        image_paths = get_image_paths(image_dir, filenames)
        
        print(f"\n  [{idx}/{len(windows)}] 窗口 {idx}: {window_info['start_frame']}→{window_info['end_frame']} ({window_info['time_span_seconds']:.1f}秒)")
        
        prompt = build_multi_image_prompt(
            filenames, surgery_id, num_frames=window_size
        )
        
        images = [Image.open(p).convert("RGB") for p in image_paths]
        image_placeholders = [{"type": "image"}] * len(images)
        
        messages = [{
            "role": "user",
            "content": image_placeholders + [{"type": "text", "text": prompt}],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        with torch.inference_mode():
            with torch.no_grad():
                print(f"    max_new_tokens={max_new_tokens}")
                output_ids = model.generate(
                    **inputs,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    use_revisit=use_revisit,
                    early_exit_layers=early_exit_layers,
                    relative_top=relative_top,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    eos_token_id=None,
                )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        caption = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        all_captions.append(caption)
        window_details.append({
            "window_id": idx,
            "start_frame": window_info['start_frame'],
            "end_frame": window_info['end_frame'],
            "time_span": window_info['time_span_seconds'],
            "caption": caption
        })
        
        print(f"    生成: {caption[:60]}..." if len(caption) > 60 else f"    生成: {caption}")
    
    # 拼接完整报告
    full_caption = "\n\n".join([
        f"[窗口 {d['window_id']}: 帧 {d['start_frame']}→{d['end_frame']}]\n{d['caption']}"
        for d in window_details
    ])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 汇总文件
    summary_file = os.path.join(output_dir, f"{surgery_id}_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"手术ID: {surgery_id}\n窗口数: {len(windows)}\n每窗口: {window_size} 张图\nUse ReVisiT: {use_revisit}\n{'='*60}\n\n{full_caption}")
    
    # 每个窗口单独保存
    for d in window_details:
        window_file = os.path.join(output_dir, f"{surgery_id}_window_{d['window_id']:03d}.txt")
        with open(window_file, "w", encoding="utf-8") as f:
            f.write(f"手术ID: {surgery_id}\n窗口: {d['window_id']}/{len(windows)}\n帧范围: {d['start_frame']} → {d['end_frame']} ({d['time_span']:.1f}秒)\n{'='*60}\n{d['caption']}")
    
    print(f"\n[{surgery_id}] 完成! 共 {len(windows)} 个窗口，汇总: {summary_file}")
    terminate_dist()
    
    return {
        "surgery_id": surgery_id,
        "total_windows": len(windows),
        "full_caption": full_caption,
        "window_details": window_details
    }

if __name__ == "__main__":
    main()