# """
# Prompt Builder Module
# Builds prompts for multi-image input with visual IDs and temporal context
# """

# from typing import List, Optional
# from .file_parser import parse_filename

# WINDOW_PROMPT_TEMPLATE = """You are observing a short clip from a nasal endoscopic surgery.
# This clip is approximately {time_info} long and contains {num_frames} frames.

# Describe ONLY what you see in terms of the following three sections, don't guess or infer:

# 1. TOOL APPEARANCE: Describe the physical appearance of any instrument visible \
# (shape, tip geometry, size relative to the cavity, whether it has jaws/blades/suction, \
# whether it is rigid or flexible, single or double-jointed, smooth or serrated edges). \
# Do NOT name the instrument nor guess its identity and purpose. Just describe what you see. 

# 2. MANIPULATION & ACTION: Describe precisely what the instrument is doing to the tissue:
# - The direction of movement (pushing medially, pulling laterally, advancing superiorly, etc.)
# - The nature of contact with tissue (pressing against, gripping and tearing, slicing through, \
# scraping along, perforating, retracting, probing)
# - What happens to the tissue as a result (tissue separates, bleeds, folds away, is removed, \
# bone is exposed, cavity opens up)
# - Any change in the field of view (cavity deepens, new structure comes into view, \
# bleeding obscures view momentarily)

# 3. ANATOMY: Provide an in-depth description of the anatomy visible in this clip, \
# covering both the state before and after tool manipulation within this window only.
# - BEFORE manipulation: Describe the anatomical structures visible at the start of the clip \
# before the tool makes contact — their shape, color, texture, surface quality, spatial \
# relationships to each other, and any notable tissue condition (edema, erythema, bleeding, \
# scarring, polyp burden, exposed bone, mucosal integrity).
# - AFTER manipulation: Describe how the anatomy has changed by the end of the clip — \
# what structures are now visible that were not before, what has been displaced or removed, \
# whether new cavities or planes have opened, and the condition of the tissue at the \
# manipulation site (raw surface, bleeding edge, exposed bone, residual tissue).
# - If no significant anatomical change occurs within this window, state that explicitly.
# - Do NOT name surgical steps or procedures. Describe only what is visually present.

# Be purely observational and specific. Write 6-10 sentences total across all three sections."""


# def build_multi_image_prompt(
#     filenames: List[str],
#     surgery_id: str,
#     num_frames: int = 20,
#     include_frame_numbers: bool = True
# ) -> str:
#     """
#     Build prompt for multi-image input using WINDOW_PROMPT structure
#     """
#     # 1. Build image placeholders
#     image_placeholders = "\n".join(["<image>"] * len(filenames))
    
#     # 2. Calculate time span
#     frame_numbers = []
#     for f in filenames:
#         info = parse_filename(f)
#         if info:
#             frame_numbers.append(info["frame"])
    
#     if len(frame_numbers) >= 2:
#         total_span = frame_numbers[-1] - frame_numbers[0]
#         seconds = total_span / 30
#         if seconds >= 60:
#             time_info = f"{seconds/60:.1f} minutes"
#         else:
#             time_info = f"{seconds:.1f} seconds"
#     else:
#         time_info = "a short moment"

#     # 3. Build prompt with WINDOW_PROMPT_TEMPLATE
#     prompt_text = WINDOW_PROMPT_TEMPLATE.format(
#         time_info=time_info,
#         num_frames=len(filenames)
#     )
    
#     # 4. Combine with image placeholders
#     prompt = f"""{image_placeholders}

# {prompt_text}"""
    
#     return prompt


# def build_simple_prompt(
#     filenames: List[str],
#     surgery_id: str,
#     custom_instruction: Optional[str] = None
# ) -> str:
#     """Simplified prompt"""
#     image_placeholders = "\n".join(["<image>"] * len(filenames))
    
#     instruction = custom_instruction or f"""
# This is a sequence of endoscopic sinus surgery screenshots (ID: {surgery_id}).
# Describe what happens in this clip in one fluent paragraph. DO NOT list images one by one. If multiple images look similar, describe them once. Keep it concise and avoid inferring overall surgical stages.

# Describe the specific actions visible: what instrument is being used and what is being done to the tissue. Use plain language (e.g., "knife" not "scalpel", "suction" not "aspirator").
# """
    
#     prompt = f"""{image_placeholders}

# {instruction}"""
#     return prompt


# def build_frame_reference_prompt(
#     filenames: List[str],
#     surgery_id: str,
#     reference_indices: List[int]
# ) -> str:
#     """Build prompt with key frame references"""
#     base_prompt = build_multi_image_prompt(filenames, surgery_id)
#     return base_prompt

"""
Prompt Builder Module
Builds prompts for multi-image input with visual IDs and temporal context
"""

from typing import List, Optional
from .file_parser import parse_filename


# ============================================================
# 三个独立的 prompt 模板
# ============================================================

TOOL_PROMPT = """You are observing a short clip from a nasal endoscopic surgery.

Describe the physical appearance of any instrument visible in this clip:
- Shape, tip geometry, size relative to the cavity
- Whether it has jaws/blades/suction
- Whether it is rigid or flexible, single or double-jointed
- Smooth or serrated edges

Do NOT name the instrument. Do NOT guess its identity or purpose. Just describe what you see.

Write 2-3 sentences."""


ACTION_PROMPT = """You are observing a short clip from a nasal endoscopic surgery.

Describe precisely what the instrument is doing to the tissue in this clip:
- The direction of movement (pushing medially, pulling laterally, advancing superiorly, etc.)
- The nature of contact with tissue (pressing against, gripping and tearing, slicing through, scraping along, perforating, retracting, probing)
- What happens to the tissue as a result (tissue separates, bleeds, folds away, is removed, bone is exposed, cavity opens up)
- Any change in the field of view (cavity deepens, new structure comes into view, bleeding obscures view momentarily)

Write 2-3 sentences."""


ANATOMY_PROMPT = """You are observing a short clip from a nasal endoscopic surgery.

Previous clip anatomy context (for continuity):
{prev_anatomy}

Now describe the anatomy visible in THIS clip:
- BEFORE manipulation: shape, color, texture, surface quality, spatial relationships, tissue condition (edema, erythema, bleeding, scarring, polyp burden, exposed bone, mucosal integrity)
- AFTER manipulation: what structures are now visible that were not before, what has been displaced or removed, whether new cavities or planes have opened, condition of the tissue at the manipulation site

If no significant anatomical change occurs, state that explicitly.
Do NOT name surgical steps or procedures. Describe only what is visually present.

Write 3-4 sentences."""


# ============================================================
# 构建函数
# ============================================================

def build_tool_prompt(filenames: List[str]) -> str:
    """构建 TOOL 部分的 prompt"""
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    prompt = f"""{image_placeholders}

{TOOL_PROMPT}"""
    return prompt


def build_action_prompt(filenames: List[str]) -> str:
    """构建 ACTION 部分的 prompt"""
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    prompt = f"""{image_placeholders}

{ACTION_PROMPT}"""
    return prompt


def build_anatomy_prompt(filenames: List[str], prev_anatomy: str = "None (this is the first clip)") -> str:
    """构建 ANATOMY 部分的 prompt，包含上一个 clip 的 anatomy 信息"""
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    prompt_text = ANATOMY_PROMPT.format(prev_anatomy=prev_anatomy)
    prompt = f"""{image_placeholders}

{prompt_text}"""
    return prompt


# ============================================================
# 保留原有函数（兼容旧代码）
# ============================================================

WINDOW_PROMPT_TEMPLATE = """You are observing a short clip from a nasal endoscopic surgery.
This clip is approximately {time_info} long and contains {num_frames} frames.

Describe ONLY what you see in terms of the following three sections, don't guess or infer:

1. TOOL APPEARANCE: ...
2. MANIPULATION & ACTION: ...
3. ANATOMY: ...

Be purely observational and specific. Write 6-10 sentences total across all three sections."""


def build_multi_image_prompt(filenames, surgery_id, num_frames=20, include_frame_numbers=True):
    """旧版（保留兼容）"""
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    prompt = f"""{image_placeholders}

{TOOL_PROMPT}"""
    return prompt


def build_simple_prompt(filenames, surgery_id, custom_instruction=None):
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    instruction = custom_instruction or "Describe what happens in this clip."
    return f"""{image_placeholders}

{instruction}"""


def build_frame_reference_prompt(filenames, surgery_id, reference_indices):
    return build_multi_image_prompt(filenames, surgery_id)