# test_single_window.py
import os
import sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from surgery_utils.data_loader import list_images_by_surgery, get_image_paths
from surgery_utils.prompt_builder import build_multi_image_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="起始索引（0-based）")
    parser.add_argument("--end", type=int, required=True, help="结束索引（0-based，不包含）")
    parser.add_argument("--surgery_id", type=str, default="L01")
    parser.add_argument("--image_dir", type=str, default="./data/uw-sinus-surgery-CL/live/images")
    parser.add_argument("--output_dir", type=str, default="./output/surgery_captions")
    parser.add_argument("--model_path", type=str, default="./data/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="cuda:1"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    all_files = list_images_by_surgery(args.image_dir, args.surgery_id, num_samples=None)
    total = len(all_files)
    
    if args.start >= total:
        print(f"起始索引 {args.start} 超出总图片数 {total}")
        return
    if args.end > total:
        args.end = total
    
    window_files = all_files[args.start:args.end]
    if not window_files:
        print(f"错误: 索引 {args.start}:{args.end} 超出范围 (总共 {total} 张)")
        return
    
    print(f"处理窗口: {args.start} → {args.end-1} ({window_files[0]} → {window_files[-1]})")
    
    image_paths = get_image_paths(args.image_dir, window_files)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    prompt = build_multi_image_prompt(window_files, args.surgery_id, num_frames=len(window_files))
    
    messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        use_revisit=False,
        early_exit_layers="last",
        relative_top=1e-5,
    )
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.surgery_id}_window_{args.start:04d}-{args.end-1:04d}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(caption)
    
    print(f"已保存: {output_file}")
    print(f"生成 token 数: {len(generated_ids_trimmed[0])}")
    print(f"描述: {caption[:80]}..." if len(caption) > 80 else f"描述: {caption}")

if __name__ == "__main__":
    main()