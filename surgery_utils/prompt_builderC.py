"""
Prompt 构建模块
负责构建多图输入的 Prompt，包含视觉 ID 和时序说明
"""

from typing import List, Dict, Optional
from .file_parser import parse_filename


def build_multi_image_prompt(
    filenames: List[str],
    surgery_id: str,
    num_frames: int = 20,
    include_frame_numbers: bool = True
) -> str:
    """
    构建多图输入的 Prompt
    """
    # 1. 构建图片占位符
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    
    # 2. 计算时间跨度
    frame_numbers = []
    for f in filenames:
        info = parse_filename(f)
        if info:
            frame_numbers.append(info["frame"])
    
    if len(frame_numbers) >= 2:
        total_span = frame_numbers[-1] - frame_numbers[0]
        time_info = f"（约 {total_span/30:.1f} 秒的视频片段）"
    else:
        time_info = ""

    prompt = f"""这是一段鼻窦内窥镜手术视频的截图序列，按时间顺序排列。共 {len(filenames)} 张图{time_info}。

【图片序列】
{image_placeholders}

【任务要求】
请观察以上截图序列，用一段话总结这段视频片段中发生的手术操作，而不是单纯描述画面变化。

【核心规则 - 必须遵守】
1. **禁止按图片顺序罗列**：绝对不要出现"图1...图2..."或"最初...随后..."这样的分时列举。
2. **区分"画面在变"和"手术在动"**:
   - 错误："器械在鼻腔内移动"（这只是画面变化，没有说明在做什么）
   - 正确："器械在清理组织表面的血"（这是具体的操作动作）
3. **尝试识别具体器械**：不要只说"器械"，根据形状推测它是**吸引器、剥离子、咬骨钳、刮匙或内镜**中的哪一种。
4. **如果画面长时间无变化**：不要硬说"器械在移动"，要说"操作暂停"或"器械稳定不动"。

【输出格式】
只输出一段连贯的文字描述，不要分点，不要编号。

请开始描述："""
    
    return prompt

# def build_multi_image_prompt(
#     filenames: List[str],
#     surgery_id: str,
#     num_frames: int = 20,
#     include_frame_numbers: bool = True
# ) -> str:
#     """
#     构建多图输入的 Prompt
#     """
#     # 1. 构建图片占位符
#     image_placeholders = "\n".join(["<image>"] * len(filenames))
    
#     # 2. 计算时间跨度
#     frame_numbers = []
#     for f in filenames:
#         info = parse_filename(f)
#         if info:
#             frame_numbers.append(info["frame"])
    
#     if len(frame_numbers) >= 2:
#         total_span = frame_numbers[-1] - frame_numbers[0]
#         time_info = f"（约 {total_span/30:.1f} 秒的视频片段）"
#     else:
#         time_info = ""

#     prompt = f"""这是一段鼻窦内窥镜手术视频的截图序列，按时间顺序排列。共 {len(filenames)} 张图{time_info}。

# 【图片序列】
# {image_placeholders}

# 【任务要求】
# 请仔细观察以上截图序列，用一段话总结这段视频片段中发生的手术操作。

# 【核心规则】
# 1. **禁止按图片顺序罗列**：不要出现"图1...图2..."或"最初...随后..."这样的列举。
# 2. **区分画面变化和手术动作**：不要只说"器械在移动"，要说清楚在做什么（如"清理血迹"、"剥离组织"）。
# 3. **尝试识别具体器械**：根据形状推测是**吸引器、剥离子、咬骨钳、刮匙或内镜**中的哪一种。
# 4. **画面长时间无变化时**：直接说"操作暂停"或"器械稳定不动"，不要硬编。

# 【输出格式】
# 请生成一段连贯的文字描述，不要分点,不要编号,句子一定要说完整。

# 请开始描述："""
    
#     return prompt

def build_simple_prompt(
    filenames: List[str],
    surgery_id: str,
    custom_instruction: Optional[str] = None
) -> str:
    """简化版 Prompt"""
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    
    instruction = custom_instruction or f"""
这是一段鼻窦内窥镜手术（ID: {surgery_id}）的视频截图。
请用一段流畅的文字描述整段视频中的手术过程，不要逐张列举图片。
"""
    
    prompt = f"""{image_placeholders}

{instruction}
"""
    return prompt


def build_frame_reference_prompt(
    filenames: List[str],
    surgery_id: str,
    reference_indices: List[int]
) -> str:
    """带重点帧引用的 Prompt（保留但简化）"""
    base_prompt = build_multi_image_prompt(filenames, surgery_id)
    return base_prompt