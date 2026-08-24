#!/usr/bin/env python3
"""
手术多图描述生成 - 主入口脚本
支持单个手术或批量处理所有手术
"""

import os
import sys
import argparse
import asyncio
from typing import List, Optional

# 添加项目路径
revisit_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(revisit_path)

from surgery_caption_eval import process_surgery, load_model, parse_args as parse_surgery_args
from surgery_utils.data_loader import get_all_surgery_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Run Surgery Caption Generation")
    
    # 运行模式
    parser.add_argument("--mode", type=str, default="single", choices=["single", "batch"],
                        help="运行模式: single=单个手术, batch=批量所有手术")
    parser.add_argument("--surgery_id", type=str, default=None, help="单个模式下的手术ID")
    
    # 数据参数
    parser.add_argument("--image_dir", type=str, required=True, help="图片目录路径")
    parser.add_argument("--num_frames", type=int, default=20, help="采样帧数")
    parser.add_argument("--output_dir", type=str, default="./output/surgery_captions", help="输出目录")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, default="./data/Qwen2.5-VL-7B-Instruct", help="模型路径")
    
    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument("--use_revisit", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                        default=True, help="是否启用 ReVisiT")
    parser.add_argument("--relative_top", type=float, default=1e-5, help="ReVisiT 相对阈值")
    parser.add_argument("--early_exit_layers", type=str, default="all", choices=["last", "all"])
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do_sample", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                        default=False, help="是否采样生成")
    
    # 滑动窗口模式
    parser.add_argument("--sliding", type=lambda x: x.lower() in ['true', '1', 'yes'], 
                        default=False, help="是否启用滑动窗口模式（每20张图生成一段描述）")
    
    args = parser.parse_args()
    return args


def get_surgery_list(image_dir: str, surgery_id: str = None) -> List[str]:
    """
    获取要处理的手术列表
    """
    all_ids = get_all_surgery_ids(image_dir)
    
    if surgery_id:
        if surgery_id not in all_ids:
            raise ValueError(f"手术ID '{surgery_id}' 不存在。可用的ID: {all_ids}")
        return [surgery_id]
    
    return all_ids


async def run_single_surgery(args, model, processor, surgery_id: str):
    """
    运行单个手术
    """
    print(f"\n{'#'*60}")
    print(f"处理手术: {surgery_id}")
    print(f"{'#'*60}")
    
    result = await process_surgery(
        model=model,
        processor=processor,
        image_dir=args.image_dir,
        surgery_id=surgery_id,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        use_revisit=args.use_revisit,
        early_exit_layers=args.early_exit_layers,
        relative_top=args.relative_top,
        do_sample=args.do_sample,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    return result


async def run_batch(args, model, processor, surgery_ids: List[str]):
    """
    批量运行多个手术
    """
    results = {}
    total = len(surgery_ids)
    
    for idx, sid in enumerate(surgery_ids, 1):
        print(f"\n{'='*60}")
        print(f"进度: [{idx}/{total}]")
        print(f"{'='*60}")
        
        try:
            result = await run_single_surgery(args, model, processor, sid)
            results[sid] = result
        except Exception as e:
            print(f"处理 {sid} 时出错: {e}")
            results[sid] = {"error": str(e)}
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("汇总报告")
    print(f"{'='*60}")
    for sid, result in results.items():
        if "error" in result:
            print(f"  {sid}: ❌ {result['error']}")
        else:
            print(f"  {sid}: ✅ 描述长度 {len(result.get('caption', ''))} 字符")
    
    return results


async def run_single_surgery_sliding(args, model, processor, surgery_id: str):
    """滑动窗口模式运行单个手术"""
    from surgery_caption_eval import process_surgery_sliding
    
    print(f"\n{'#'*60}")
    print(f"处理手术（滑动窗口模式）: {surgery_id}")
    print(f"{'#'*60}")
    
    result = await process_surgery_sliding(
        model=model,
        processor=processor,
        image_dir=args.image_dir,
        surgery_id=surgery_id,
        window_size=args.num_frames,
        step=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        use_revisit=args.use_revisit,
        early_exit_layers=args.early_exit_layers,
        relative_top=args.relative_top,
        do_sample=args.do_sample,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    return result


def main():
    args = parse_args()
    print("Args:", args)
    
    # 1. 获取要处理的手术列表
    try:
        surgery_ids = get_surgery_list(args.image_dir, args.surgery_id)
    except ValueError as e:
        print(f"错误: {e}")
        return
    
    print(f"\n将处理 {len(surgery_ids)} 个手术: {surgery_ids}")
    
    # 2. 加载模型（只加载一次）
    print("\n加载模型中...")
    model, processor = load_model(args.model_path)
    print("模型加载完成")
    
    # 3. 运行
    if len(surgery_ids) == 1:
        if hasattr(args, 'sliding') and args.sliding:
            asyncio.run(run_single_surgery_sliding(args, model, processor, surgery_ids[0]))
        else:
            asyncio.run(run_single_surgery(args, model, processor, surgery_ids[0]))
    else:
        asyncio.run(run_batch(args, model, processor, surgery_ids))
    
    print(f"\n所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()