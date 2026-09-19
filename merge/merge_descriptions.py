import os
import re
import sys
import json
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merge.qwen_text_utils import load_qwen, ask_same_combined, merge_two


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptions_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="./data/Qwen3-VL-8B-Thinking")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=100.0)
    return parser.parse_args()


def read_part(filepath):
    if not os.path.exists(filepath):
        return None, None, None
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    gid_match = re.search(r'Group:\s*(\d+)', content)
    time_match = re.search(r'Time Range:\s*(.+)', content)
    parts = content.split('=' * 60, 1)
    caption = parts[1].strip() if len(parts) > 1 else ""
    gid = int(gid_match.group(1)) if gid_match else 0
    time_range = time_match.group(1).strip() if time_match else ""
    return gid, time_range, caption


def extract_anatomy_parts(anatomy_text):
    before_match = re.search(r'BEFORE MANIPULATION:?\s*(.*?)(?=AFTER MANIPULATION|$)', anatomy_text, re.DOTALL | re.IGNORECASE)
    after_match = re.search(r'AFTER MANIPULATION:?\s*(.*)', anatomy_text, re.DOTALL | re.IGNORECASE)
    before = before_match.group(1).strip() if before_match else ""
    after = after_match.group(1).strip() if after_match else anatomy_text.strip()
    return before, after


def load_clips(desc_dir):
    clips = []
    files = [f for f in os.listdir(desc_dir) if re.match(r'group_\d+_action\.txt$', f)]
    files.sort(key=lambda x: int(re.search(r'group_(\d+)', x).group(1)))

    for f in files:
        gid = int(re.search(r'group_(\d+)', f).group(1))
        _, time_range, action_caption = read_part(os.path.join(desc_dir, f))
        _, _, tool_caption = read_part(os.path.join(desc_dir, f"group_{gid:03d}_tool.txt"))
        _, _, anatomy_caption = read_part(os.path.join(desc_dir, f"group_{gid:03d}_anatomy.txt"))

        anatomy_before, anatomy_after = extract_anatomy_parts(anatomy_caption)

        clips.append({
            "gid": gid,
            "time_range": time_range,
            "action": action_caption,
            "tool": tool_caption,
            "anatomy": anatomy_caption,
            "anatomy_before": anatomy_before,
            "anatomy_after": anatomy_after
        })
    return clips


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    load_qwen(args.model_path, args.device)

    clips = load_clips(args.descriptions_dir)
    print(f"Loaded {len(clips)} clips")
    print(f"Combined similarity threshold: {args.threshold}")

    if len(clips) == 0:
        print("No clips found, exiting")
        return

    merged_results = []
    merge_log = []

    current = clips[0]
    idx = 1

    while idx < len(clips):
        next_clip = clips[idx]
        print(f"\nComparing: Clip {current['gid']} ↔ Clip {next_clip['gid']}")

        score = ask_same_combined(
            current['anatomy_after'],
            next_clip['anatomy_before'],
            current['action'],
            next_clip['action']
        )
        print(f"  Combined similarity: {score:.1f} (threshold {args.threshold})")

        is_same = score >= args.threshold
        print(f"  Merge: {is_same}")

        merge_log.append({
            "clip_a": current['gid'],
            "clip_b": next_clip['gid'],
            "combined_similarity": score,
            "merged": is_same
        })

        if is_same:
            print(f"  Merging...")
            new_action = merge_two(current['action'], next_clip['action'],
                                   current['time_range'], next_clip['time_range'])
            new_tool = merge_two(current['tool'], next_clip['tool'],
                                 current['time_range'], next_clip['time_range'])
            new_anatomy = merge_two(current['anatomy'], next_clip['anatomy'],
                                    current['time_range'], next_clip['time_range'])

            start_time = current['time_range'].split('→')[0].strip()
            end_time = next_clip['time_range'].split('→')[-1].strip()
            new_time = f"{start_time} → {end_time}"

            if 'merged_from' in current:
                merged_from = current['merged_from'] + [next_clip['gid']]
            else:
                merged_from = [current['gid'], next_clip['gid']]

            _, new_anatomy_after = extract_anatomy_parts(new_anatomy)

            current = {
                "gid": current['gid'],
                "time_range": new_time,
                "action": new_action,
                "tool": new_tool,
                "anatomy": new_anatomy,
                "anatomy_before": current['anatomy_before'],
                "anatomy_after": new_anatomy_after,
                "merged_from": merged_from
            }
            print(f"  → Merged gid={current['gid']}, time={new_time}")
        else:
            merged_results.append(current)
            print(f"  → Keep gid={current['gid']}, move pointer to Clip {next_clip['gid']}")
            current = next_clip

        idx += 1

    merged_results.append(current)

    for mc in merged_results:
        out_file = os.path.join(args.output_dir, f"merged_{mc['gid']:03d}.txt")
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(f"Group: {mc['gid']}\n")
            f.write(f"Time Range: {mc['time_range']}\n")
            f.write(f"Merged From: {mc.get('merged_from', [mc['gid']])}\n")
            f.write(f"{'='*60}\n")
            f.write(f"TOOL:\n{mc['tool']}\n\n")
            f.write(f"ACTION:\n{mc['action']}\n\n")
            f.write(f"ANATOMY:\n{mc['anatomy']}\n")

    summary_file = os.path.join(args.output_dir, "summary_merged.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Total groups: {len(merged_results)}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"{'='*60}\n\n")
        for mc in merged_results:
            f.write(f"[Group {mc['gid']}] ({mc['time_range']})\n")
            f.write(f"TOOL:\n{mc['tool']}\n\n")
            f.write(f"ACTION:\n{mc['action']}\n\n")
            f.write(f"ANATOMY:\n{mc['anatomy']}\n")
            f.write("\n\n" + "="*60 + "\n\n")

    log_file = os.path.join(args.output_dir, "merge_log.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "threshold": args.threshold,
            "comparisons": merge_log
        }, f, indent=2, ensure_ascii=False)

    print(f"\nMerged into {len(merged_results)} segments")
    print(f"Summary file: {summary_file}")
    print(f"Merge log: {log_file}")


if __name__ == "__main__":
    main()