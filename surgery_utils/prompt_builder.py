"""
Prompt Builder Module
Builds prompts for multi-image input with visual IDs and temporal context
"""

from typing import List, Optional
from .file_parser import parse_filename

WINDOW_PROMPT_TEMPLATE = """You are observing a short clip from a nasal endoscopic surgery.
This clip is approximately {time_info} long and contains {num_frames} frames.

Describe ONLY what you see in terms of the following three sections:

1. TOOL APPEARANCE: Describe the physical appearance of any instrument visible \
(shape, tip geometry, size relative to the cavity, whether it has jaws/blades/suction, \
whether it is rigid or flexible, single or double-jointed, smooth or serrated edges). \
Do NOT name the instrument — describe what it looks like.

2. MANIPULATION & ACTION: Describe precisely what the instrument is doing to the tissue:
- The direction of movement (pushing medially, pulling laterally, advancing superiorly, etc.)
- The nature of contact with tissue (pressing against, gripping and tearing, slicing through, \
scraping along, perforating, retracting, probing)
- What happens to the tissue as a result (tissue separates, bleeds, folds away, is removed, \
bone is exposed, cavity opens up)
- Any change in the field of view (cavity deepens, new structure comes into view, \
bleeding obscures view momentarily)

3. ANATOMY: Provide an in-depth description of the anatomy visible in this clip, \
covering both the state before and after tool manipulation within this window only.
- BEFORE manipulation: Describe the anatomical structures visible at the start of the clip \
before the tool makes contact — their shape, color, texture, surface quality, spatial \
relationships to each other, and any notable tissue condition (edema, erythema, bleeding, \
scarring, polyp burden, exposed bone, mucosal integrity).
- AFTER manipulation: Describe how the anatomy has changed by the end of the clip — \
what structures are now visible that were not before, what has been displaced or removed, \
whether new cavities or planes have opened, and the condition of the tissue at the \
manipulation site (raw surface, bleeding edge, exposed bone, residual tissue).
- If no significant anatomical change occurs within this window, state that explicitly.
- Do NOT name surgical steps or procedures. Describe only what is visually present.

Be purely observational and specific. Write 6-10 sentences total across all three sections."""


def build_multi_image_prompt(
    filenames: List[str],
    surgery_id: str,
    num_frames: int = 20,
    include_frame_numbers: bool = True
) -> str:
    """
    Build prompt for multi-image input using WINDOW_PROMPT structure
    """
    # 1. Build image placeholders
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    
    # 2. Calculate time span
    frame_numbers = []
    for f in filenames:
        info = parse_filename(f)
        if info:
            frame_numbers.append(info["frame"])
    
    if len(frame_numbers) >= 2:
        total_span = frame_numbers[-1] - frame_numbers[0]
        seconds = total_span / 30
        if seconds >= 60:
            time_info = f"{seconds/60:.1f} minutes"
        else:
            time_info = f"{seconds:.1f} seconds"
    else:
        time_info = "a short moment"

    # 3. Build prompt with WINDOW_PROMPT_TEMPLATE
    prompt_text = WINDOW_PROMPT_TEMPLATE.format(
        time_info=time_info,
        num_frames=len(filenames)
    )
    
    # 4. Combine with image placeholders
    prompt = f"""{image_placeholders}

{prompt_text}"""
    
    return prompt


def build_simple_prompt(
    filenames: List[str],
    surgery_id: str,
    custom_instruction: Optional[str] = None
) -> str:
    """Simplified prompt"""
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    
    instruction = custom_instruction or f"""
This is a sequence of endoscopic sinus surgery screenshots (ID: {surgery_id}).
Describe what happens in this clip in one fluent paragraph. DO NOT list images one by one. If multiple images look similar, describe them once. Keep it concise and avoid inferring overall surgical stages.

Describe the specific actions visible: what instrument is being used and what is being done to the tissue. Use plain language (e.g., "knife" not "scalpel", "suction" not "aspirator").
"""
    
    prompt = f"""{image_placeholders}

{instruction}"""
    return prompt


def build_frame_reference_prompt(
    filenames: List[str],
    surgery_id: str,
    reference_indices: List[int]
) -> str:
    """Build prompt with key frame references"""
    base_prompt = build_multi_image_prompt(filenames, surgery_id)
    return base_prompt