# """
# Prompt Builder Module
# Builds prompts for multi-image input with visual IDs and temporal context
# """

# from typing import List, Dict, Optional
# from .file_parser import parse_filename


# def build_multi_image_prompt(
#     filenames: List[str],
#     surgery_id: str,
#     num_frames: int = 20,
#     include_frame_numbers: bool = True
# ) -> str:
#     """
#     Build prompt for multi-image input
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
#         time_info = f"(~{total_span/30:.1f} seconds of video)"
#     else:
#         time_info = ""

#     prompt = f"""This is a short clip from an endoscopic sinus surgery, consisting of a sequence of screenshots arranged in chronological order. Total {len(filenames)} images{time_info}.

# This clip represents only a small segment of the entire surgery. DO NOT infer or speculate about the overall surgical stage, phase, or progress.

# [Image Sequence]
# {image_placeholders}

# [Task]
# Observe the image sequence above and summarize the surgical operation in a single paragraph. Focus on WHAT the surgeon is DOING within this clip, not just visual changes.

# [CRITICAL RULES - MUST FOLLOW]

# 1. **NEVER list images in order**: Absolutely avoid "Image 1 shows...", "Image 2 shows...", or "First... then... later..." structures.

# 2. **Describe surgical actions, NOT camera movements**: 
#    - WRONG: "The instrument moves in the nasal cavity." (this describes camera view changes, not surgery)
#    - CORRECT: "The surgeon uses a suction to clear blood from the tissue surface."

# 3. **CONDENSE similar frames**: Many consecutive images look nearly identical. DO NOT repeat the same description for each. Aggregate them. Say it ONCE.

# 4. **When nothing happens**: If the instrument remains stationary for an extended period, simply state "The instrument is held steady" or "No active manipulation occurs." Do NOT fabricate movement or description.

# 6. **ONE sentence per observation**: Do not list multiple observations in parallel structure (e.g., "the instrument moves, the instrument clears blood, the instrument observes tissue"). Instead, write in narrative flow: "The surgeon uses a suction to clear blood, then pauses to inspect the field."

# 7. **CRITICAL - Avoid enumeration**: Never write "A, B, C" style lists. Combine into natural prose.

# 8. **Do NOT infer surgical stage**: This is just a short clip. Do not say "preparing for the next step," "later in the surgery," or any other statements that speculate about the overall procedure phase. Describe ONLY what is visible in these images.

# [Output Format]
# Output ONLY one continuous paragraph. No bullet points. No numbering.

# Begin description:"""
    
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
# Describe the entire surgical process in one fluent paragraph. DO NOT list images one by one. If multiple images look similar, describe them once and move on.
# """
    
#     prompt = f"""{image_placeholders}

# {instruction}
# """
#     return prompt


# def build_frame_reference_prompt(

#     filenames: List[str],
#     surgery_id: str,
#     reference_indices: List[int]
# ) -> str:
#     """Prompt with key frame references (simplified)"""
#     base_prompt = build_multi_image_prompt(filenames, surgery_id)
#     return base_prompt

"""
Prompt Builder Module
Builds prompts for multi-image input with visual IDs and temporal context
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
    Build prompt for multi-image input
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
            time_info = f"(~{seconds/60:.1f} minutes of video)"
        else:
            time_info = f"(~{seconds:.1f} seconds of video)"
    else:
        time_info = ""

    prompt = f"""This is a short clip from an endoscopic sinus surgery, consisting of a sequence of screenshots arranged in chronological order. Total {len(filenames)} images {time_info}.

This clip represents only a small segment of the entire surgery. DO NOT infer or speculate about the overall surgical stage, phase, or progress.

[Image Sequence]
{image_placeholders}

[Task]
Observe the image sequence above and summarize the surgical operation in a single paragraph. Focus on WHAT the surgeon is DOING within this clip, not just visual changes.

[CRITICAL RULES - MUST FOLLOW]

1. **NEVER list images in order**: Absolutely avoid "Image 1 shows...", "Image 2 shows...", or "First... then... later..." structures.

2. **Describe surgical actions, NOT camera movements**: 
   - WRONG: "The instrument moves in the nasal cavity." (this describes camera view changes, not surgery)
   - CORRECT: "The surgeon uses a suction to clear blood from the tissue surface."

3. **CONDENSE similar frames**: Many consecutive images look nearly identical. DO NOT repeat the same description for each. Aggregate them. Say it ONCE.

4. **When nothing happens**: If the instrument remains stationary for an extended period, simply state "The instrument is held steady" or "No active manipulation occurs." Do NOT fabricate movement or description.

5. **ONE sentence per observation**: Do not list multiple observations in parallel structure (e.g., "the instrument moves, the instrument clears blood, the instrument observes tissue"). Instead, write in narrative flow: "The surgeon uses a suction to clear blood, then pauses to inspect the field."

6. **CRITICAL - Avoid enumeration**: Never write "A, B, C" style lists. Combine into natural prose.

7. **Do NOT infer surgical stage**: This is just a short clip. Do not say "preparing for the next step," "later in the surgery," or any other statements that speculate about the overall procedure phase. Describe ONLY what is visible in these images.

8. **Keep it CONCISE**: Output should be 50-150 words. If multiple frames are similar, describe them collectively, not repeatedly. Avoid unnecessary detail.

9. **USE PLAIN, NON‑MEDICAL LANGUAGE**: Describe instruments with simple everyday names. For example:
   - Say "knife" or "blade", NOT "scalpel" or "microsurgical blade".
   - Say "forceps" or "grasper", NOT "clamp" or "hemostat".
   - Say "suction tube", NOT "suction aspirator".
   - Say "clear blood" or "clean the area", NOT "evacuate hematoma" or "debride tissue".
   - Avoid jargon like "dissection", "retraction", "cauterization". Instead say "cutting", "pulling", "burning" (if relevant).
   The goal is to make the description understandable to a non‑medical person.

[Output Format]
Output ONLY one continuous paragraph. No bullet points. No numbering.

Begin description:"""
    
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

{instruction}
"""
    return prompt


def build_frame_reference_prompt(
    filenames: List[str],
    surgery_id: str,
    reference_indices: List[int]
) -> str:
    """Build prompt with key frame references"""
    base_prompt = build_multi_image_prompt(filenames, surgery_id)
    return base_prompt