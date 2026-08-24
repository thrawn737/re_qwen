"""
数据加载模块
负责扫描目录、按手术ID过滤、排序和采样
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
    加载指定手术ID的所有图片，排序后可选采样
    
    Args:
        image_dir: 图片目录路径
        surgery_id: 手术ID，如 "L01" 或 "S01"
        num_samples: 采样数量，None表示返回所有图片
        
    Returns:
        图片文件名列表（已排序，已采样）
        
    Examples:
        >>> list_images_by_surgery("./data/images", "L01", num_samples=20)
        ["L01_30.jpg", "L01_60.jpg", ...]  # 20张均匀采样
        
        >>> list_images_by_surgery("./data/images", "S03")
        ["S03_30.jpg", "S03_60.jpg", ...]  # 所有图片
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"目录不存在: {image_dir}")
    
    # 1. 扫描目录，筛选匹配手术ID的文件
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
        raise ValueError(f"未找到手术ID '{surgery_id}' 的任何图片，请检查目录和ID是否正确")
    
    # 2. 按帧号排序
    sorted_files = sort_by_frame(matched_files)
    
    # 3. 如果指定了采样数量，均匀采样
    if num_samples is not None and num_samples > 0:
        total = len(sorted_files)
        if total <= num_samples:
            return sorted_files  # 总数不够，返回全部
        
        # 均匀采样：取 num_samples 个索引
        indices = [int(i * (total - 1) / (num_samples - 1)) for i in range(num_samples)]
        return [sorted_files[i] for i in indices]
    
    return sorted_files


def get_all_surgery_ids(image_dir: str) -> List[str]:
    """
    获取数据集中所有手术ID
    
    Args:
        image_dir: 图片目录路径
        
    Returns:
        手术ID列表，如 ["L01", "L02", "S01", "S02", ...]
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"目录不存在: {image_dir}")
    
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
    将文件名列表转换为完整路径列表
    
    Args:
        image_dir: 图片目录路径
        filenames: 文件名列表
        
    Returns:
        完整路径列表
    """
    return [os.path.join(image_dir, f) for f in filenames]


def get_surgery_info(image_dir: str, surgery_id: str) -> dict:

    """
    获取某个手术的统计信息
    
    Args:
        image_dir: 图片目录路径
        surgery_id: 手术ID
        
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
    获取滑动窗口的图片批次
    
    Args:
        image_dir: 图片目录路径
        surgery_id: 手术ID，如 "L01"
        window_size: 每个窗口包含多少张图
        step: 每次滑动多少张图
        
    Returns:
        窗口列表，每个窗口是图片文件名列表
        
    Examples:
        >>> windows = get_sliding_windows("./images", "L01", window_size=20, step=20)
        >>> len(windows)  # 1154/20 ≈ 57 个完整窗口
        >>> windows[0]  # ['L01_30.jpg', 'L01_60.jpg', ...] 共20张
    """
    # 1. 获取该手术的所有图片（不采样）
    all_files = list_images_by_surgery(image_dir, surgery_id, num_samples=None)
    total = len(all_files)
    
    if total == 0:
        return []
    
    windows = []
    for start in range(0, total, step):
        end = start + window_size
        if end <= total:  # 只取完整的窗口
            windows.append(all_files[start:end])
        else:
            # 最后一个窗口如果不足，用最后 window_size 张
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
    获取滑动窗口的图片批次，包含窗口信息
    
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
        
        # 解析首尾帧号
        first_info = parse_filename(window_files[0])
        last_info = parse_filename(window_files[-1])
        
        start_frame = first_info["frame"] if first_info else 0
        end_frame = last_info["frame"] if last_info else 0
        
        # 假设30fps，计算时间跨度（秒）
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
        
        # 如果已经到了末尾，停止
        if end >= total:
            break
    
    return windows