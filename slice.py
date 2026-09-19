import cv2
import os
import argparse
import math


def extract_frames(video_path, output_dir, fps, target_size=(448, 448)):
    """
    Extract frames from video at a given FPS and save as images,
    resizing them to target_size.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Original FPS: {original_fps:.2f}, total frames: {total_frames}")

    interval = max(1, int(round(original_fps / fps)))
    print(f"Extraction interval: {interval} frames, target size: {target_size}")

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    saved_count = 0
    mapping = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            # Resize
            resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            filename = f"frame_{saved_count+1:06d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, resized)
            mapping.append((frame_idx, filename))
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"Extraction done, {saved_count} images saved")

    # Save mapping
    mapping_file = os.path.join(output_dir, "mapping.txt")
    with open(mapping_file, 'w') as f:
        for video_frame, img_name in mapping:
            f.write(f"{video_frame}\t{img_name}\n")
    print(f"Mapping saved to: {mapping_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--fps", type=float, required=True, help="Extraction FPS")
    parser.add_argument("--size", type=int, nargs=2, default=(448, 448),
                        help="Target size: width height, e.g. --size 448 448")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.fps, tuple(args.size))