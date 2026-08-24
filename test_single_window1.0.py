# test_single_window.py
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
from surgery_utils.data_loader import list_images_by_surgery, get_image_paths
from surgery_utils.prompt_builder import build_multi_image_prompt

model_path = "./data/Qwen2.5-VL-7B-Instruct"
image_dir = "./data/uw-sinus-surgery-CL/live/images"
surgery_id = "L01"

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda:1"
)
processor = AutoProcessor.from_pretrained(model_path)

# 取前20张
all_files = list_images_by_surgery(image_dir, surgery_id, num_samples=None)
window_1 = all_files[:20]
image_paths = get_image_paths(image_dir, window_1)
images = [Image.open(p).convert("RGB") for p in image_paths]

prompt = build_multi_image_prompt(window_1, surgery_id, num_frames=20)

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

print(f"生成 token 数: {len(generated_ids_trimmed[0])}")
print(f"描述: {caption}")