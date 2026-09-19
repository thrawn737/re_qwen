import os
import sys
import torch
import time
import argparse
from PIL import Image
import re

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import *
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from surgery_utils.prompt_builder import (
    build_tool_prompt,
    build_action_prompt,
    build_anatomy_prompt
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups_txt", type=str, required=True)
    parser.add_argument("--sampled_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--merged_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./data/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_revisit", type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=True)
    parser.add_argument("--surgery_id", type=str, default="L01")
    return parser.parse_args()


def parse_groups_txt(groups_txt):
    groups = []
    with open(groups_txt, 'r') as f:
        lines = f.readlines()
    current = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Group"):
            match = re.search(r'Group\s+(\d+)', line)
            if match:
                gid = int(match.group(1))
                if current:
                    groups.append(current)
                current = {'gid': gid}
        elif line.startswith("start_frame:"):
            current['start_frame'] = int(line.split()[1])
        elif line.startswith("end_frame:"):
            current['end_frame'] = int(line.split()[1])
        elif line.startswith("start_time:"):
            current['start_time'] = line.split(' ', 1)[1].strip()
        elif line.startswith("end_time:"):
            current['end_time'] = line.split(' ', 1)[1].strip()
        elif line.startswith("output_dir:"):
            current['output_dir'] = line.split(' ', 1)[1].strip()
        elif line.startswith("n_sampled:"):
            current['n_sampled'] = int(line.split()[1])
    if current:
        groups.append(current)
    return groups


def extract_frame_number(filename):
    match = re.search(r'(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else 0


def extract_after_manipulation(anatomy_text):
    match = re.search(r'AFTER MANIPULATION:?\s*(.*)', anatomy_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return anatomy_text.strip()


def generate_caption(model, processor, images, prompt, max_new_tokens, use_revisit):
    messages = [{"role": "user", "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_revisit=use_revisit,
        early_exit_layers="last",
        relative_top=1e-5,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=None,
    )
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    del inputs, output_ids
    torch.cuda.empty_cache()
    return caption


def main():
    args = parse_args()
    print(f"Reading groups: {args.groups_txt}")

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map=args.device
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("Model loaded")

    groups = parse_groups_txt(args.groups_txt)
    print(f"Loaded {len(groups)} groups")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.merged_dir, exist_ok=True)

    all_tools = []
    all_actions = []
    all_anatomies = []

    prev_anatomy = "None (this is the first clip)"

    for idx, group in enumerate(groups, 1):
        gid = group['gid']
        output_dir = group['output_dir']
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(args.sampled_root, os.path.basename(output_dir))
        else:
            output_dir = os.path.join(args.sampled_root, f"group_{gid:03d}")

        if not os.path.exists(output_dir):
            print(f"Warning: directory not found {output_dir}, skipping")
            continue
        files = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        files.sort(key=extract_frame_number)

        if not files:
            print(f"Warning: no images in {output_dir}, skipping")
            continue

        start_time = group.get('start_time', 'N/A')
        end_time = group.get('end_time', 'N/A')
        time_range = f"{start_time} → {end_time}"

        print(f"\n[{idx}/{len(groups)}] Group {gid}: {time_range}, {len(files)} images")

        image_paths = [os.path.join(output_dir, f) for f in files]
        images = [Image.open(p).convert("RGB") for p in image_paths]

        # TOOL
        prompt_tool = build_tool_prompt(files)
        caption_tool = generate_caption(model, processor, images, prompt_tool, args.max_new_tokens, args.use_revisit).strip()
        print(f"  TOOL: {caption_tool[:80]}...")

        # ACTION
        prompt_action = build_action_prompt(files)
        caption_action = generate_caption(model, processor, images, prompt_action, args.max_new_tokens, args.use_revisit).strip()
        print(f"  ACTION: {caption_action[:80]}...")

        # ANATOMY
        prompt_anatomy = build_anatomy_prompt(files, prev_anatomy=prev_anatomy)
        caption_anatomy = generate_caption(model, processor, images, prompt_anatomy, args.max_new_tokens, args.use_revisit).strip()
        print(f"  ANATOMY: {caption_anatomy[:80]}...")

        prev_anatomy = extract_after_manipulation(caption_anatomy)

        # Save three parts separately
        tool_file = os.path.join(args.output_dir, f"group_{gid:03d}_tool.txt")
        with open(tool_file, "w", encoding="utf-8") as f:
            f.write(f"Group: {gid}\n")
            f.write(f"Time Range: {time_range}\n")
            f.write(f"{'='*60}\n")
            f.write(caption_tool)

        action_file = os.path.join(args.output_dir, f"group_{gid:03d}_action.txt")
        with open(action_file, "w", encoding="utf-8") as f:
            f.write(f"Group: {gid}\n")
            f.write(f"Time Range: {time_range}\n")
            f.write(f"{'='*60}\n")
            f.write(caption_action)

        anatomy_file = os.path.join(args.output_dir, f"group_{gid:03d}_anatomy.txt")
        with open(anatomy_file, "w", encoding="utf-8") as f:
            f.write(f"Group: {gid}\n")
            f.write(f"Time Range: {time_range}\n")
            f.write(f"{'='*60}\n")
            f.write(caption_anatomy)

        # Merged version (clean format)
        merged_file = os.path.join(args.merged_dir, f"group_{gid:03d}_merged.txt")
        with open(merged_file, "w", encoding="utf-8") as f:
            f.write(f"Group: {gid}\n")
            f.write(f"Time Range: {time_range}\n")
            f.write(f"{'='*60}\n")
            f.write(f"TOOL:\n{caption_tool}\n\n")
            f.write(f"ACTION:\n{caption_action}\n\n")
            f.write(f"ANATOMY:\n{caption_anatomy}\n")

        all_tools.append((gid, start_time, end_time, caption_tool))
        all_actions.append((gid, start_time, end_time, caption_action))
        all_anatomies.append((gid, start_time, end_time, caption_anatomy))

        del images, image_paths
        torch.cuda.empty_cache()
        time.sleep(1)

    # Summaries (three separate files)
    for kind, data in [("tool", all_tools), ("action", all_actions), ("anatomy", all_anatomies)]:
        summary_file = os.path.join(args.output_dir, f"summary_{kind}.txt")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"Total groups: {len(data)}\n")
            f.write(f"Type: {kind.upper()}\n")
            f.write(f"Use ReVisiT: {args.use_revisit}\n")
            f.write(f"{'='*60}\n\n")
            for gid, start, end, cap in data:
                f.write(f"[Group {gid}] ({start} → {end})\n")
                f.write(cap)
                f.write("\n\n" + "="*60 + "\n\n")
        print(f"Summary file: {summary_file}")

    # Merged summary
    merged_summary = os.path.join(args.merged_dir, "summary_merged.txt")
    with open(merged_summary, "w", encoding="utf-8") as f:
        f.write(f"Total groups: {len(all_tools)}\n")
        f.write(f"Use ReVisiT: {args.use_revisit}\n")
        f.write(f"{'='*60}\n\n")
        for (gid, start, end, tool_cap), (_, _, _, action_cap), (_, _, _, anatomy_cap) in zip(all_tools, all_actions, all_anatomies):
            f.write(f"[Group {gid}] ({start} → {end})\n")
            f.write(f"TOOL:\n{tool_cap}\n\n")
            f.write(f"ACTION:\n{action_cap}\n\n")
            f.write(f"ANATOMY:\n{anatomy_cap}\n")
            f.write("\n\n" + "="*60 + "\n\n")
    print(f"Merged summary: {merged_summary}")

    print("\nDone!")


if __name__ == "__main__":
    main()