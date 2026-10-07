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
from crop_utils import auto_detect_crop_ratios


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_root", default="./sampled_groups_b")
    parser.add_argument("--a", type=float, default=None)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--b", type=float, default=0.0)
    parser.add_argument("--X", type=float, default=1.0)
    parser.add_argument("--N", type=int, default=30)
    parser.add_argument("--size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--model_path", default="./models/LanguageBind_Image")
    parser.add_argument("--output_txt", default="./output/groups_b.txt")
    parser.add_argument("--no_overlap", action="store_true")
    parser.add_argument("--left_crop", type=float, default=0.18)
    parser.add_argument("--right_crop", type=float, default=0.25)
    parser.add_argument("--auto_crop", action="store_true", help="Auto-detect left/right black borders")
    parser.add_argument("--fps", type=float, default=3)
    parser.add_argument("--clip_file", type=str, default="./output/clip.txt")
    parser.add_argument("--percentile_a", type=float, default=0.85)
    parser.add_argument("--percentile_c", type=float, default=0.95)

    # ===== Clipping limit switch =====
    parser.add_argument("--limit_clip", action="store_true",
                        help="Enable clipping limit: at most 5 truncations, no consecutive truncations")
    parser.add_argument("--max_clip_count", type=int, default=5,
                        help="Maximum number of truncations per group when --limit_clip is enabled")
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
    data = processor([img_path], [''])
    pixel_values = data['pixel_values']
    if not isinstance(pixel_values, torch.Tensor):
        pixel_values = torch.tensor(pixel_values)
    pixel_values = pixel_values.to(device)

    input_ids = data['input_ids']
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    input_ids = input_ids.to(device)

    attention_mask = data.get('attention_mask', None)
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask)
        attention_mask = attention_mask.to(device)

    inputs = {'pixel_values': pixel_values, 'input_ids': input_ids}
    if attention_mask is not None:
        inputs['attention_mask'] = attention_mask

    with torch.no_grad():
        out = model(**inputs)
        return out.image_embeds.cpu().numpy().flatten()


def angular_distance(feat1, feat2):
    cos_sim = cosine_similarity([feat1], [feat2])[0][0]
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return math.acos(cos_sim)


def crop_frame(frame, left_ratio=0.18, right_ratio=0.25):
    h, w = frame.shape[:2]
    crop_w = int(w * (1 - left_ratio - right_ratio))
    start_x = int(w * left_ratio)
    return frame[:, start_x:start_x + crop_w]


def select_v0_v1_v2(features):
    n = len(features)
    if n <= 3:
        return list(range(n))

    avg_dist = []
    for i in range(n):
        total = 0
        for j in range(n):
            if i != j:
                total += angular_distance(features[i], features[j])
        avg_dist.append(total)
    v0 = np.argmin(avg_dist)

    max_dist = -1
    v1 = -1
    for i in range(n):
        if i != v0:
            dist = angular_distance(features[v0], features[i])
            if dist > max_dist:
                max_dist = dist
                v1 = i

    v0_vec = features[v0]
    v1_vec = features[v1]
    n_vec = np.cross(v0_vec[:3], v1_vec[:3])
    if np.linalg.norm(n_vec) < 1e-6:
        n_vec = np.array([0.0, 0.0, 1.0])
    else:
        n_vec = n_vec / np.linalg.norm(n_vec)

    max_cos = -1
    v2 = -1
    for i in range(n):
        if i == v0 or i == v1:
            continue
        feat = features[i][:3]
        cos_sim = abs(np.dot(feat, n_vec) / (np.linalg.norm(feat) + 1e-8))
        if cos_sim > max_cos:
            max_cos = cos_sim
            v2 = i

    return [v0, v1, v2] if v2 != -1 else [v0, v1]


def sample_group_from_video_adaptive(video_path, start_frame, end_frame, output_dir, prefix, target_size,
                                     left_crop=0.18, right_crop=0.25,
                                     model=None, processor=None, device=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(end_frame, total_frames - 1)

    if start_frame == end_frame:
        os.makedirs(output_dir, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, frame = cap.read()
        if ret:
            cropped = crop_frame(frame, left_crop, right_crop)
            resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
            filename = f"{prefix}_0001.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), resized)
        cap.release()
        return 1

    temp_dir = os.path.join(output_dir, "temp_features")
    os.makedirs(temp_dir, exist_ok=True)

    segment_length = end_frame - start_frame + 1
    chunk_size = max(1, segment_length // 10)

    os.makedirs(output_dir, exist_ok=True)
    saved_count = 0
    temp_file_counter = 0

    for chunk_idx in range(10):
        chunk_start = start_frame + chunk_idx * chunk_size
        chunk_end = min(start_frame + (chunk_idx + 1) * chunk_size - 1, end_frame)
        if chunk_start > chunk_end:
            break

        chunk_frames = []
        chunk_positions = []
        for pos in range(chunk_start, chunk_end + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                chunk_frames.append(frame)
                chunk_positions.append(pos)

        if len(chunk_frames) < 3:
            for i, pos in enumerate(chunk_positions):
                frame = chunk_frames[i]
                cropped = crop_frame(frame, left_crop, right_crop)
                resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
                filename = f"{prefix}_{saved_count+1:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), resized)
                saved_count += 1
            continue

        features = []
        for frame in chunk_frames:
            temp_file_counter += 1
            temp_path = os.path.join(temp_dir, f"temp_{temp_file_counter}.jpg")
            cv2.imwrite(temp_path, frame)
            feat = get_feature(temp_path, model, processor, device)
            features.append(feat)
            os.remove(temp_path)

        if not features:
            continue

        features = np.array(features)
        selected_indices = select_v0_v1_v2(features)

        for idx in selected_indices:
            pos = chunk_positions[idx]
            frame = chunk_frames[idx]
            cropped = crop_frame(frame, left_crop, right_crop)
            resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
            filename = f"{prefix}_{saved_count+1:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), resized)
            saved_count += 1

    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    cap.release()
    return saved_count


def compute_a_c_from_clips(clip_file, image_dir, model, processor, device,
                           percentile_a=0.85, percentile_c=0.95):
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
            cos_sim = cosine_similarity([features[i]], [features[i + 1]])[0][0]
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

    if args.auto_crop:
        print("Auto-detecting left/right black borders...")
        auto_left, auto_right = auto_detect_crop_ratios(args.video_path)
        args.left_crop = auto_left
        args.right_crop = auto_right
        print(f"Using auto-detected crop: left {args.left_crop:.4f}, right {args.right_crop:.4f}")
    else:
        print(f"Using manual crop: left {args.left_crop}, right {args.right_crop}")

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

    print(f"Cumulative angle grouping: a={a_value}, c={c_value}, X={args.X}, N={args.N}")
    print(f"Clipping limit: {'enabled' if args.limit_clip else 'disabled'}")

    all_files = sorted([f for f in os.listdir(args.image_dir)
                        if f.startswith('frame_') and f.endswith('.jpg')],
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

        clip_count = 0
        last_truncated = False

        while j < total:
            cos = cosine_similarity([features[start]], [features[j]])[0][0]
            theta = math.acos(max(-1, min(1, cos)))
            d = args.b if theta <= a_value else theta - a_value

            if args.limit_clip:
                if d > c_value:
                    if (not last_truncated) and (clip_count < args.max_clip_count):
                        d = c_value
                        clip_count += 1
                        last_truncated = True
                    else:
                        last_truncated = False
                else:
                    last_truncated = False
            else:
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
            'gid': len(groups_info) + 1,
            'start_frame': start_frame,
            'end_frame': end_frame
        })

        if args.no_overlap:
            i = group_indices[-1] + 1
        else:
            i = group_indices[-1]
            if i == start:
                i += 1

    print(f"Grouping done, {len(groups_info)} groups")

    print("Sampling...")
    for info in groups_info:
        gid = info['gid']
        output_dir = os.path.join(args.output_root, f"group_{gid:03d}")
        n_sampled = sample_group_from_video_adaptive(
            args.video_path, info['start_frame'], info['end_frame'],
            output_dir, f"group_{gid:03d}", tuple(args.size),
            args.left_crop, args.right_crop,
            model, processor, device
        )
        info['output_dir'] = output_dir
        info['n_sampled'] = n_sampled
        print(f"Group {gid}: frames {info['start_frame']} -> {info['end_frame']}, sampled {n_sampled} (adaptive)")

    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    with open(args.output_txt, 'w') as f:
        f.write(f"total_groups: {len(groups_info)}\n")
        f.write(f"fps: {args.fps}\n")
        f.write(f"a: {a_value:.6f}\n")
        f.write(f"c: {c_value:.6f}\n")
        f.write(f"X: {args.X}\n")
        f.write(f"limit_clip: {args.limit_clip}\n")
        if args.limit_clip:
            f.write(f"max_clip_count: {args.max_clip_count}\n")
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

    print(f"Group info saved to: {args.output_txt}")


if __name__ == "__main__":
    main()