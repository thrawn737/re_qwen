import cv2
import numpy as np


def auto_detect_crop_ratios(video_path, threshold=5, sample_frame_idx=100):
    """
    Take one frame from the video, detect left/right black borders by column.
    Center = midpoint of left and right borders.
    Square side = original frame height.
    Return (left_ratio, right_ratio).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frame_idx = min(sample_frame_idx, total_frames - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Warning: cannot read frame, using default crop 0.18/0.25")
        return 0.18, 0.25

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape  # h=1080, w=1920

    binary = (gray > 127).astype(np.uint8)
    col_white_count = binary.sum(axis=0)

    left_bound = 0
    for x in range(w):
        if col_white_count[x] >= threshold:
            left_bound = x
            break

    right_bound = w - 1
    for x in range(w - 1, -1, -1):
        if col_white_count[x] >= threshold:
            right_bound = x
            break

    center = (left_bound + right_bound) // 2
    half_side = h // 2

    new_left = center - half_side
    new_right = center + half_side

    if new_left < 0:
        new_left = 0
        new_right = h
    if new_right > w:
        new_right = w
        new_left = w - h

    left_ratio = new_left / w
    right_ratio = (w - new_right) / w

    print(f"Frame: {w}x{h}")
    print(f"Detected: left_bound={left_bound}, right_bound={right_bound}, center={center}")
    print(f"Square side: {h}, crop region: x=[{new_left}, {new_right}]")
    print(f"Crop ratios: left {left_ratio:.4f}, right {right_ratio:.4f}")

    return left_ratio, right_ratio