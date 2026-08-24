"""
文件解析模块
负责解析 UW Sinus Surgery 数据集的文件名
命名规则：
  - cadaver: S[video_ID]_[frame_index].jpg  如 S01_30.jpg
  - live:    L[video_ID]_[frame_index].jpg  如 L03_120.jpg
"""

import re
from typing import Optional, Tuple, Dict


def parse_filename(filename: str) -> Optional[Dict[str, any]]:
    """
    解析单个文件名，提取手术ID和帧号
    
    Args:
        filename: 文件名，如 "L01_30.jpg" 或 "S03_120.jpg"
        
    Returns:
        dict: {
            "surgery_id": "L01",      # 完整手术ID
            "prefix": "L",            # L 或 S
            "video_id": "01",         # 视频编号
            "frame": 30,              # 帧号（整数）
            "full_name": "L01_30.jpg"
        }
        如果解析失败，返回 None
        
    Examples:
        >>> parse_filename("L01_30.jpg")
        {"surgery_id": "L01", "prefix": "L", "video_id": "01", "frame": 30}
        
        >>> parse_filename("S10_255.jpg")
        {"surgery_id": "S10", "prefix": "S", "video_id": "10", "frame": 255}
    """
    # 匹配模式：前缀(L/S) + 数字(视频ID) + 下划线 + 数字(帧号) + .jpg
    pattern = r'^([LS])(\d+)_(\d+)\.jpg$'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    prefix = match.group(1)      # "L" 或 "S"
    video_id = match.group(2)    # "01", "10" 等
    frame = int(match.group(3))  # 帧号，转为整数
    
    return {
        "surgery_id": f"{prefix}{video_id}",  # "L01"
        "prefix": prefix,                      # "L"
        "video_id": video_id,                  # "01"
        "frame": frame,                        # 30
        "full_name": filename
    }


def get_surgery_id(filename: str) -> Optional[str]:
    """
    快捷函数：只提取手术ID
    
    Args:
        filename: 文件名
        
    Returns:
        手术ID字符串，如 "L01"，解析失败返回 None
    """
    result = parse_filename(filename)
    return result["surgery_id"] if result else None


def get_frame_number(filename: str) -> Optional[int]:
    """
    快捷函数：只提取帧号
    
    Args:
        filename: 文件名
        
    Returns:
        帧号整数，解析失败返回 None
    """
    result = parse_filename(filename)
    return result["frame"] if result else None


def is_valid_filename(filename: str) -> bool:
    """
    检查文件名是否符合数据集格式
    
    Args:
        filename: 文件名
        
    Returns:
        True 如果符合格式，否则 False
    """
    return parse_filename(filename) is not None


def sort_by_frame(filenames: list) -> list:
    """
    按帧号排序文件列表
    
    Args:
        filenames: 文件名列表
        
    Returns:
        排序后的文件名列表
        
    Examples:
        >>> sort_by_frame(["L01_90.jpg", "L01_30.jpg", "L01_60.jpg"])
        ["L01_30.jpg", "L01_60.jpg", "L01_90.jpg"]
    """
    # 提取帧号，过滤无效文件名
    valid_files = []
    for f in filenames:
        info = parse_filename(f)
        if info is not None:
            valid_files.append((info["frame"], f))
    
    # 按帧号排序
    valid_files.sort(key=lambda x: x[0])
    
    return [f for _, f in valid_files]