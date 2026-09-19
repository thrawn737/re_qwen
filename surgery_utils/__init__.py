"""
surgery_utils 工具包
提供文件解析、数据加载、Prompt构建等功能
"""

from .file_parser import (
    parse_filename,
    get_surgery_id,
    get_frame_number,
    is_valid_filename,
    sort_by_frame
)

from .data_loader import (
    list_images_by_surgery,
    get_all_surgery_ids,
    get_image_paths,
    get_surgery_info,
    get_sliding_windows,           # 新增
    get_sliding_windows_with_info  # 新增
)

from .prompt_builder2 import (
    build_multi_image_prompt,
    build_simple_prompt,
    build_frame_reference_prompt
)

__all__ = [
    # file_parser
    'parse_filename',
    'get_surgery_id',
    'get_frame_number',
    'is_valid_filename',
    'sort_by_frame',
    # data_loader
    'list_images_by_surgery',
    'get_all_surgery_ids',
    'get_image_paths',
    'get_surgery_info',
    'get_sliding_windows' ,           # 新增
    'get_sliding_windows_with_info' , # 新增
    # prompt_builder
    'build_multi_image_prompt',
    'build_simple_prompt',
    'build_frame_reference_prompt',
]