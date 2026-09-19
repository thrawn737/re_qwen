"""
Data Loader Module
Scans directories, filters by surgery ID, sorts, and samples images
"""

import os
from typing import List, Optional
from .file_parser import parse_filename, sort_by_frame


def list_images_by_surgery(
    image_dir: str,
    surgery_id: str,
    num_samples: Optional[int] = None
) -> List[str]:
    """
    Load all images for a given surgery ID, sort them, and optionally sample.

    Args:
        image_dir: Image directory path
        surgery_id: Surgery ID, e.g. "L01" or "S01"
        num_samples: Number of samples. None means return all images.

    Returns:
        List of image filenames (sorted, optionally sampled)

    Examples:
        >>> list_images_by_surgery("./data/images", "L01", num_samples=20)
        ["L01_30.jpg", "L01_60.jpg", ...]  # 20 uniformly sampled images

        >>> list_images_by_surgery("./data/images", "S03")
        ["S03_30.jpg", "S03_60.jpg", ...]  # all images
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    # 1. Scan directory, filter files matching the surgery ID
    matched_files = []
    for f in os.listdir(image_dir):
        if not f.lower().endswith('.jpg'):
            continue

        info = parse_filename(f)
        if info is None:
            continue

        if info["surgery_id"] == surgery_id:
            matched_files.append(f)

    if not matched_files:
        raise ValueError(f"No images found for surgery ID '{surgery_id}'. Check directory and ID.")

    # 2. Sort by frame number
    sorted_files = sort_by_frame(matched_files)

    # 3. If num_samples is specified, uniformly sample
    if num_samples is not None and num_samples > 0:
        total = len(sorted_files)
        if total <= num_samples:
            return sorted_files  # Not enough, return all

        # Uniform sampling: pick num_samples indices
        indices = [int(i * (total - 1) / (num_samples - 1)) for i in range(num_samples)]
        return [sorted_files[i] for i in indices]

    return sorted_files


def get_all_surgery_ids(image_dir: str) -> List[str]:
    """
    Get all surgery IDs in the dataset.

    Args:
        image_dir: Image directory path

    Returns:
        List of surgery IDs, e.g. ["L01", "L02", "S01", "S02", ...]
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    surgery_ids = set()
    for f in os.listdir(image_dir):
        if not f.lower().endswith('.jpg'):
            continue

        info = parse_filename(f)
        if info is not None:
            surgery_ids.add(info["surgery_id"])

    return sorted(list(surgery_ids))


def get_image_paths(image_dir: str, filenames: List[str]) -> List[str]:
    """
    Convert a list of filenames to full paths.

    Args:
        image_dir: Image directory path
        filenames: List of filenames

    Returns:
        List of full paths
    """
    return [os.path.join(image_dir, f) for f in filenames]


def get_surgery_info(image_dir: str, surgery_id: str) -> dict:
    """
    Get statistics for a given surgery.

    Args:
        image_dir: Image directory path
        surgery_id: Surgery ID

    Returns:
        dict: {
            "surgery_id": "L01",
            "total_frames": 1500,
            "first_frame": 30,
            "last_frame": 1500,
            "prefix": "L",
            "video_id": "01"
        }
    """
    filenames = list_images_by_surgery(image_dir, surgery_id)
    if not filenames:
        return {}

    first_info = parse_filename(filenames[0])
    last_info = parse_filename(filenames[-1])

    return {
        "surgery_id": surgery_id,
        "total_frames": len(filenames),
        "first_frame": first_info["frame"] if first_info else None,
        "last_frame": last_info["frame"] if last_info else None,
        "prefix": first_info["prefix"] if first_info else None,
        "video_id": first_info["video_id"] if first_info else None
    }


def get_sliding_windows(
    image_dir: str,
    surgery_id: str,
    window_size: int = 20,
    step: int = 20
) -> List[List[str]]:
    """
    Get sliding window batches of images.

    Args:
        image_dir: Image directory path
        surgery_id: Surgery ID, e.g. "L01"
        window_size: Number of images per window
        step: Step size for sliding

    Returns:
        List of windows, each window is a list of image filenames

    Examples:
        >>> windows = get_sliding_windows("./images", "L01", window_size=20, step=20)
        >>> len(windows)  # 1154/20 ≈ 57 complete windows
        >>> windows[0]  # ['L01_30.jpg', 'L01_60.jpg', ...] 20 images
    """
    # 1. Get all images for the surgery (no sampling)
    all_files = list_images_by_surgery(image_dir, surgery_id, num_samples=None)
    total = len(all_files)

    if total == 0:
        return []

    windows = []
    for start in range(0, total, step):
        end = start + window_size
        if end <= total:  # Only complete windows
            windows.append(all_files[start:end])
        else:
            # If the last window is incomplete, use the last window_size images
            if start < total:
                windows.append(all_files[-window_size:])
            break

    return windows


def get_sliding_windows_with_info(
    image_dir: str,
    surgery_id: str,
    window_size: int = 20,
    step: int = 20
) -> List[dict]:
    """
    Get sliding window batches of images with window info.

    Returns:
        List[dict]: [
            {
                "window_id": 1,
                "filenames": ["L01_30.jpg", ...],
                "start_frame": 30,
                "end_frame": 600,
                "frame_count": 20,
                "time_span_seconds": 19.0
            },
            ...
        ]
    """
    all_files = list_images_by_surgery(image_dir, surgery_id, num_samples=None)
    total = len(all_files)

    if total == 0:
        return []

    windows = []
    window_id = 1

    for start in range(0, total, step):
        end = start + window_size

        if end <= total:
            window_files = all_files[start:end]
        else:
            if start < total:
                window_files = all_files[-window_size:]
            else:
                break

        # Parse first and last frame numbers
        first_info = parse_filename(window_files[0])
        last_info = parse_filename(window_files[-1])

        start_frame = first_info["frame"] if first_info else 0
        end_frame = last_info["frame"] if last_info else 0

        # Assume 30 fps, compute time span in seconds
        time_span = (end_frame - start_frame) / 30 if end_frame > start_frame else 0

        windows.append({
            "window_id": window_id,
            "filenames": window_files,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": len(window_files),
            "time_span_seconds": time_span,
            "start_index": start,
            "end_index": end
        })

        window_id += 1

        # Stop if we reached the end
        if end >= total:
            break

    return windows