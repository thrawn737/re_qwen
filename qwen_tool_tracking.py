import os
import sys
import argparse
import torch
import numpy as np
import re
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Image directory path")
    parser.add_argument("--output_clip", required=True, help="Output clip.txt file path")
    parser.add_argument("--model_path", default="./data/Qwen2.5-VL-3B-Instruct", help="Model path")
    parser.add_argument("--left_crop", type=float, default=0.18, help="Left crop ratio")
    parser.add_argument("--right_crop", type=float, default=0.25, help="Right crop ratio")
    parser.add_argument("--target_size", type=int, nargs=2, default=(256, 256), help="Target size (width height)")
    parser.add_argument("--has_tool_threshold", type=float, default=0.55)
    parser.add_argument("--same_tool_threshold", type=float, default=0.53)
    parser.add_argument("--min_clip_length", type=int, default=30)
    parser.add_argument("--max_clips", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=5)
    return parser.parse_args()


args = parse_args()

IMAGE_DIR = args.image_dir
MODEL_PATH = args.model_path
OUTPUT_CLIP_FILE = args.output_clip

LEFT_CROP = args.left_crop
RIGHT_CROP = args.right_crop
TARGET_SIZE = tuple(args.target_size)

HAS_TOOL_THRESHOLD = args.has_tool_threshold
SAME_TOOL_THRESHOLD = args.same_tool_threshold
MIN_CLIP_LENGTH = args.min_clip_length
MAX_CLIPS = args.max_clips
WINDOW_SIZE = args.window_size

print(f"Processing directory: {IMAGE_DIR}")
print(f"Output file: {OUTPUT_CLIP_FILE}")

print("Loading Qwen-VL...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded")


def extract_frame_number(filename):
    match = re.search(r'(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else 0


def crop_and_resize(image, left_ratio=0.18, right_ratio=0.25, target_size=(256, 256)):
    w, h = image.size
    crop_w = int(w * (1 - left_ratio - right_ratio))
    start_x = int(w * left_ratio)
    cropped = image.crop((start_x, 0, start_x + crop_w, h))
    resized = cropped.resize(target_size, Image.Resampling.LANCZOS)
    return resized


def ask_has_tool(image_path):
    img = Image.open(image_path).convert("RGB")
    img = crop_and_resize(img, LEFT_CROP, RIGHT_CROP, TARGET_SIZE)
    prompt = """Does this image contain any surgical tool? Answer with "Yes" or "No" only."""

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    logits = outputs.scores[0]
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    yes_logit = logits[0, yes_id].item()
    no_logit = logits[0, no_id].item()
    prob = np.exp(yes_logit) / (np.exp(yes_logit) + np.exp(no_logit))

    del inputs, outputs
    torch.cuda.empty_cache()
    return prob


def ask_same_tool(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("RGB")
    img1 = crop_and_resize(img1, LEFT_CROP, RIGHT_CROP, TARGET_SIZE)
    img2 = Image.open(img2_path).convert("RGB")
    img2 = crop_and_resize(img2, LEFT_CROP, RIGHT_CROP, TARGET_SIZE)

    prompt = """Are the surgical tools in these two images the same tool? Answer with "Yes" or "No" only."""

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "image"},
        {"type": "text", "text": prompt}
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img1, img2], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    logits = outputs.scores[0]
    yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

    yes_logit = logits[0, yes_id].item()
    no_logit = logits[0, no_id].item()
    prob = np.exp(yes_logit) / (np.exp(yes_logit) + np.exp(no_logit))

    del inputs, outputs
    torch.cuda.empty_cache()
    return prob


def main():
    all_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    all_files.sort(key=extract_frame_number)
    total = len(all_files)
    print(f"Total frames: {total}")

    clips = []
    clip_count = 0
    i = 0

    while i < total and clip_count < MAX_CLIPS:
        frame_path = os.path.join(IMAGE_DIR, all_files[i])
        frame_num = extract_frame_number(all_files[i])

        has_tool_prob = ask_has_tool(frame_path)
        print(f"Frame {frame_num}: has tool -> {has_tool_prob:.3f}")

        if has_tool_prob < HAS_TOOL_THRESHOLD:
            i += 1
            continue

        print(f"Frame {frame_num}: tool detected, start tracking")

        start_idx = i
        consecutive_count = 1
        fail_count = 0
        j = i + 1

        while j < total:
            current_frame_path = os.path.join(IMAGE_DIR, all_files[j])
            same_prob = ask_same_tool(frame_path, current_frame_path)
            print(f"  Frame {extract_frame_number(all_files[j])}: same tool -> {same_prob:.3f}")

            if same_prob >= SAME_TOOL_THRESHOLD:
                consecutive_count += 1
                fail_count = 0
                j += 1
            else:
                fail_count += 1
                j += 1
                if fail_count >= 3:
                    print(f"  3 consecutive mismatches, clip segment ends")
                    break

        if consecutive_count >= MIN_CLIP_LENGTH:
            ext_start_idx = max(0, start_idx - 1)
            ext_end_idx = min(total - 1, j)
            start_frame = extract_frame_number(all_files[ext_start_idx])
            end_frame = extract_frame_number(all_files[ext_end_idx])
            clip_count += 1
            clip_entry = f"Clip {clip_count}: {start_frame} → {end_frame} (consecutive: {consecutive_count} frames)"
            clips.append(clip_entry)
            print(f"  Recorded clip: {clip_entry}")
        else:
            print(f"  Consecutive frames {consecutive_count} < {MIN_CLIP_LENGTH}, discarded")

        i = j

    os.makedirs(os.path.dirname(OUTPUT_CLIP_FILE), exist_ok=True)
    with open(OUTPUT_CLIP_FILE, 'w') as f:
        for entry in clips:
            f.write(entry + "\n")

    print(f"\nFound {len(clips)} clips, saved to {OUTPUT_CLIP_FILE}")


if __name__ == "__main__":
    main()