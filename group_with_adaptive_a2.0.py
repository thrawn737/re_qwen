import os
import sys
import argparse
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import math
import cv2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LanguageBind'))
from languagebind import LanguageBindImage, LanguageBindImageTokenizer, LanguageBindImageProcessor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_root", default="./sampled_groups")
    parser.add_argument("--a", type=float, default=None)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--X", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--model_path", default="./models/LanguageBind_Image")
    parser.add_argument("--output_txt", default="./output/groups.txt")
    parser.add_argument("--no_overlap", action="store_true")
    parser.add_argument("--left_crop", type=float, default=0.18)
    parser.add_argument("--right_crop", type=float, default=0.25)
    parser.add_argument("--fps", type=float, default=3)
    parser.add_argument("--clip_file", type=str, default="./output/clip.txt")
    parser.add_argument("--percentile_a", type=float, default=0.85)
    parser.add_argument("--percentile_c", type=float, default=0.95)
    return parser.parse_args()

def extract_frame_number(filename):
    match = re.search(r'(\d+)\.jpg$', filename)
    if match:
        return int(match.group(1)) * 10
    return 0

def frame_to_time(frame_num, fps=30):
    seconds = frame_num / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def get_feature(img_path, model, processor, device):
    data = processor([img_path], [''], return_tensors='pt')
    data = {k: v.to(device) if hasattr(v, 'to') else v for k, v in data.items()}
    with torch.no_grad():
        out = model(**data)
        return out.image_embeds.cpu().numpy().flatten()

def crop_frame(frame, left_ratio=0.18, right_ratio=0.25):
    h, w = frame.shape[:2]
    crop_w = int(w * (1 - left_ratio - right_ratio))
    start_x = int(w * left_ratio)
    return frame[:, start_x:start_x + crop_w]

def sample_group_from_video(video_path, start_frame, end_frame, n_frames, output_dir, prefix, target_size, left_crop=0.18, right_crop=0.25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total_frames - 1)

    if start_frame == end_frame:
        n_frames = 1

    positions = np.linspace(start_frame, end_frame, n_frames, dtype=int).tolist()
    os.makedirs(output_dir, exist_ok=True)

    for idx, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        cropped = crop_frame(frame, left_crop, right_crop)
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        filename = f"{prefix}_{idx+1:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), resized)

    cap.release()
    return len(positions)

def merge_small_groups(groups_info):
    if len(groups_info) <= 1:
        return groups_info

    changed = True
    while changed:
        changed = False
        durations = [g['end_frame'] - g['start_frame'] for g in groups_info]
        if not durations:
            break
        avg_duration = np.mean(durations)
        threshold = avg_duration / 2

        for idx in range(len(groups_info)):
            dur = groups_info[idx]['end_frame'] - groups_info[idx]['start_frame']
            if dur < threshold:
                left_idx = idx - 1
                right_idx = idx + 1

                candidates = []
                if left_idx >= 0:
                    left_dur = groups_info[left_idx]['end_frame'] - groups_info[left_idx]['start_frame']
                    candidates.append((left_idx, left_dur))
                if right_idx < len(groups_info):
                    right_dur = groups_info[right_idx]['end_frame'] - groups_info[right_idx]['start_frame']
                    candidates.append((right_idx, right_dur))

                if not candidates:
                    continue

                merge_idx = min(candidates, key=lambda x: x[1])[0]

                if merge_idx < idx:
                    groups_info[merge_idx]['end_frame'] = groups_info[idx]['end_frame']
                    del groups_info[idx]
                else:
                    groups_info[merge_idx]['start_frame'] = groups_info[idx]['start_frame']
                    del groups_info[idx]

                changed = True
                break

    for new_gid, g in enumerate(groups_info, 1):
        g['gid'] = new_gid

    return groups_info

def compute_a_c_from_clips(clip_file, image_dir, model, processor, device, percentile_a=0.85, percentile_c=0.95):
    if not os.path.exists(clip_file):
        print(f"Warning: {clip_file} not found, using default a=0.2, c=0.5")
        return 0.2, 0.5

    clips = []
    with open(clip_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.search(r'Clip \d+: (\d+) → (\d+)', line)
            if match:
                start = int(match.group(1))
                end = int(match.group(2))
                clips.append((start, end))

    if not clips:
        print(f"Warning: no valid clips in {clip_file}, using default a=0.2, c=0.5")
        return 0.2, 0.5

    print(f"Read {len(clips)} clips from {clip_file}")

    all_angles = []
    all_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    all_files.sort(key=extract_frame_number)

    for idx, (start, end) in enumerate(clips, 1):
        clip_files = []
        for f in all_files:
            num = extract_frame_number(f)
            if start <= num <= end:
                clip_files.append(f)

        if len(clip_files) < 2:
            continue

        features = []
        for f in clip_files:
            path = os.path.join(image_dir, f)
            features.append(get_feature(path, model, processor, device))
        features = np.array(features)

        for i in range(len(features) - 1):
            cos_sim = cosine_similarity([features[i]], [features[i+1]])[0][0]
            cos_sim = max(-1.0, min(1.0, cos_sim))
            theta = math.acos(cos_sim)
            all_angles.append(theta)

    if not all_angles:
        print("Warning: cannot compute angles, using default a=0.2, c=0.5")
        return 0.2, 0.5

    all_angles = np.array(all_angles)
    a_value = np.percentile(all_angles, percentile_a * 100)
    c_value = np.percentile(all_angles, percentile_c * 100)

    print(f"  Angle stats: min={np.min(all_angles):.4f}, "
          f"median={np.percentile(all_angles, 50):.4f}, "
          f"{percentile_a*100:.0f}%={a_value:.4f}, "
          f"{percentile_c*100:.0f}%={c_value:.4f}, "
          f"max={np.max(all_angles):.4f}")

    return a_value, c_value

def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = LanguageBindImage.from_pretrained(args.model_path, cache_dir='./cache_dir')
    tokenizer = LanguageBindImageTokenizer.from_pretrained(args.model_path, cache_dir='./cache_dir')
    processor = LanguageBindImageProcessor(model.config, tokenizer)
    model.eval()
    model.to(device)

    if args.a is None or args.c is None:
        print("a or c not specified, computing from clip.txt...")
        a_value, c_value = compute_a_c_from_clips(
            args.clip_file, args.image_dir, model, processor, device,
            args.percentile_a, args.percentile_c
        )
        if args.a is not None:
            a_value = args.a
        if args.c is not None:
            c_value = args.c
        print(f"Computed a = {a_value:.6f}, c = {c_value:.6f}")
    else:
        a_value = args.a
        c_value = args.c
        print(f"Using specified a = {a_value}, c = {c_value}")

    print(f"Grouping: a={a_value}, c={c_value}, X={args.X}, N={args.N}")
    print(f"Crop: left {args.left_crop*100:.0f}%, right {args.right_crop*100:.0f}%")

    all_files = sorted([f for f in os.listdir(args.image_dir) if f.startswith('frame_') and f.endswith('.jpg')],
                       key=extract_frame_number)
    if not all_files:
        print("Error: no frame images found")
        return
    total = len(all_files)
    print(f"Total frames: {total}")

    print("Extracting features...")
    features = []
    for i, f in enumerate(all_files):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{total}")
        path = os.path.join(args.image_dir, f)
        features.append(get_feature(path, model, processor, device))

    print("Grouping...")
    groups_info = []
    i = 0
    while i < total:
        start = i
        cum = 0.0
        group_indices = [i]
        j = i + 1
        while j < total:
            cos = cosine_similarity([features[start]], [features[j]])[0][0]
            theta = math.acos(max(-1, min(1, cos)))
            d = args.b if theta <= a_value else theta - a_value
            if d > c_value:
                d = c_value
            cum += d
            if cum >= args.X:
                group_indices.append(j)
                break
            else:
                group_indices.append(j)
                j += 1

        start_frame = extract_frame_number(all_files[group_indices[0]])
        end_frame = extract_frame_number(all_files[group_indices[-1]])

        if start_frame == end_frame:
            if args.no_overlap:
                i = group_indices[-1] + 1
            else:
                i = group_indices[-1]
                if i == start:
                    i += 1
            continue

        groups_info.append({
            'gid': 0,
            'start_frame': start_frame,
            'end_frame': end_frame
        })

        if args.no_overlap:
            i = group_indices[-1] + 1
        else:
            i = group_indices[-1]
            if i == start:
                i += 1

    print(f"Initial grouping done, {len(groups_info)} groups")

    groups_info = merge_small_groups(groups_info)
    print(f"After merging small groups, {len(groups_info)} groups")

    print("Sampling...")
    for info in groups_info:
        gid = info['gid']
        output_dir = os.path.join(args.output_root, f"group_{gid:03d}")
        n_sampled = sample_group_from_video(
            args.video_path, info['start_frame'], info['end_frame'], args.N,
            output_dir, f"group_{gid:03d}", tuple(args.size),
            args.left_crop, args.right_crop
        )
        info['output_dir'] = output_dir
        info['n_sampled'] = n_sampled
        print(f"Group {gid}: frames {info['start_frame']} → {info['end_frame']}, sampled {n_sampled}")

    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, 'w') as f:
        f.write(f"total_groups: {len(groups_info)}\n")
        f.write(f"fps: {args.fps}\n")
        f.write(f"a: {a_value:.6f}\n")
        f.write(f"c: {c_value:.6f}\n")
        f.write(f"X: {args.X}\n")
        for info in groups_info:
            start_time = frame_to_time(info['start_frame'])
            end_time = frame_to_time(info['end_frame'])
            f.write(f"\nGroup {info['gid']}:\n")
            f.write(f"  start_frame: {info['start_frame']}\n")
            f.write(f"  end_frame: {info['end_frame']}\n")
            f.write(f"  start_time: {start_time}\n")
            f.write(f"  end_time: {end_time}\n")
            f.write(f"  output_dir: {info['output_dir']}\n")
            f.write(f"  n_sampled: {info['n_sampled']}\n")

    print(f"Grouping info saved to: {args.output_txt}")

if __name__ == "__main__":
    main()