"""
Prompt Builder Module
Builds prompts for multi-image input with visual IDs and temporal context
"""

from typing import List, Optional


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

Previous clip's anatomy AFTER manipulation (for continuity):
{prev_anatomy}

Now describe the anatomy visible in THIS clip in exactly two subsections:

BEFORE MANIPULATION:
Describe the anatomical structures visible at the start of the clip before the tool makes contact — their shape, color, texture, surface quality, spatial relationships, and any notable tissue condition (edema, erythema, bleeding, scarring, polyp burden, exposed bone, mucosal integrity).

AFTER MANIPULATION:
Describe how the anatomy has changed by the end of the clip — what structures are now visible that were not before, what has been displaced or removed, whether new cavities or planes have opened, and the condition of the tissue at the manipulation site.

If no significant anatomical change occurs, state that explicitly.
Do NOT name surgical steps or procedures. Describe only what is visually present.

Write 2-3 sentences for BEFORE and 2-3 sentences for AFTER."""


def build_tool_prompt(filenames: List[str]) -> str:
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    return f"""{image_placeholders}

{TOOL_PROMPT}"""


def build_action_prompt(filenames: List[str]) -> str:
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    return f"""{image_placeholders}

{ACTION_PROMPT}"""


def build_anatomy_prompt(filenames: List[str], prev_anatomy: str = "None (this is the first clip)") -> str:
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    prompt_text = ANATOMY_PROMPT.format(prev_anatomy=prev_anatomy)
    return f"""{image_placeholders}

{prompt_text}"""


def build_multi_image_prompt(filenames, surgery_id, num_frames=20, include_frame_numbers=True):
    return build_tool_prompt(filenames)


def build_simple_prompt(filenames, surgery_id, custom_instruction=None):
    image_placeholders = "\n".join(["<image>"] * len(filenames))
    instruction = custom_instruction or "Describe what happens in this clip."
    return f"""{image_placeholders}

{instruction}"""


def build_frame_reference_prompt(filenames, surgery_id, reference_indices):
    return build_multi_image_prompt(filenames, surgery_id)