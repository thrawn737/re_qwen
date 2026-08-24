# run_surgery_windows.py
import os
import sys
import time
import gc
import torch
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from surgery_utils.data_loader import list_images_by_surgery, get_image_paths
from surgery_utils.prompt_builder import build_multi_image_prompt

model_path = "./data/Qwen2.5-VL-7B-Instruct"
image_dir = "./data/uw-sinus-surgery-CL/live/images"
surgery_id = "L01"
output_dir = "./output/surgery_captions"
window_size = 20

# 加载模型（和 test_single_window.py 完全一样）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda:1"
)
processor = AutoProcessor.from_pretrained(model_path)

# 获取所有图片
all_files = list_images_by_surgery(image_dir, surgery_id, num_samples=None)
total = len(all_files)
print(f"总图片数: {total}")

# 按窗口切分
windows = []
for i in range(0, total, window_size):
    end = i + window_size
    if end <= total:
        windows.append(all_files[i:end])
    else:
        if i < total:
            windows.append(all_files[-window_size:])
        break

print(f"共 {len(windows)} 个窗口，每窗口 {window_size} 张图")

os.makedirs(output_dir, exist_ok=True)
all_captions = []

for idx, window_files in enumerate(windows, 1):
        # 只跑第一次
    if idx > 1:
        break
    print(f"\n[{idx}/{len(windows)}] 窗口 {idx}: {window_files[0]} → {window_files[-1]}")
    
    image_paths = get_image_paths(image_dir, window_files)
    images = [Image.open(p).convert("RGB") for p in image_paths]
    
    prompt = build_multi_image_prompt(window_files, surgery_id, num_frames=window_size)
    
    messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        use_revisit=True,
        early_exit_layers="last",
        relative_top=1e-5,
    )
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    print(f"  生成 token 数: {len(generated_ids_trimmed[0])}")
    print(f"  描述: {caption[:80]}..." if len(caption) > 80 else f"  描述: {caption}")
    
    all_captions.append({
        "window_id": idx,
        "files": window_files,
        "caption": caption
    })
    
    # 清理当前窗口的显存和内存
    del inputs, output_ids, generated_ids_trimmed, images, image_paths
    gc.collect()
    torch.cuda.empty_cache()
    
    # 等待 2 秒，防止下一轮 Prompt 覆盖
    time.sleep(2)

# 保存汇总
summary_file = os.path.join(output_dir, f"{surgery_id}_summary.txt")
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(f"手术ID: {surgery_id}\n")
    f.write(f"总窗口数: {len(windows)}\n")
    f.write(f"每窗口: {window_size} 张图\n")
    f.write(f"{'='*60}\n\n")
    for item in all_captions:
        f.write(f"[窗口 {item['window_id']}]\n")
        f.write(item['caption'])
        f.write("\n\n")

print(f"\n完成! 汇总: {summary_file}")