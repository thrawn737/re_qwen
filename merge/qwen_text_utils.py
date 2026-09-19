import os
import sys
import torch
import re

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

_model = None
_processor = None
_device = None


def load_qwen(model_path, device="cuda:0"):
    global _model, _processor, _device
    if _model is None:
        print(f"加载 Qwen3-VL 模型: {model_path}")
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map=device
        )
        _processor = AutoProcessor.from_pretrained(model_path)
        _device = device
        print("Qwen3-VL 加载完成")
    return _model, _processor, _device


def strip_thinking(text):
    match = re.search(r'</think\s*>', text, re.IGNORECASE)
    if match:
        text = text[match.end():].strip()

    for tag in ["Thinking Process:", "<|begin_of_thought|>", "<|end_of_thought|>"]:
        if tag in text:
            text = text.split(tag, 1)[-1].strip()

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith(("Okay,", "Hmm,", "First,", "Let me", "I need",
                            "I should", "Looking at", "The user", "Wait,")):
            continue
        if line:
            cleaned_lines.append(line)

    if cleaned_lines:
        text = '\n'.join(cleaned_lines)

    return text.strip()


def qwen_generate(prompt, max_new_tokens=2048):
    model, processor, device = _model, _processor, _device
    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    generated = processor.decode(
        output_ids[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )
    del inputs, output_ids
    torch.cuda.empty_cache()

    return strip_thinking(generated).strip()


def ask_same_combined(anatomy_after_a, anatomy_before_b, action_a, action_b):
    """
    把 anatomy 和 action 一起放进同一个 prompt，综合判断是否该合并
    返回分数 0-100
    """
    prompt = f"""You are a medical documentation editor analyzing an endoscopic sinus surgery.

Clip A:
- Anatomy AFTER manipulation: {anatomy_after_a}
- Action: {action_a}

Clip B:
- Anatomy BEFORE manipulation: {anatomy_before_b}
- Action: {action_b}

Considering BOTH the anatomical structure being operated on AND the surgical action being performed, should these two clips be merged into one continuous segment?

Rate the likelihood from 0 to 100, where 0 means they should NOT be merged (different structure or different action), and 100 means they should be merged (same structure and same action).

Output ONLY one integer from 0 to 100. No explanation. No reasoning. Just the number.
"""
    response = qwen_generate(prompt, max_new_tokens=2048)
    print(f"[DEBUG] combined raw: {response}")
    nums = re.findall(r'\d+', response)
    if nums:
        score = float(nums[-1])
        score = max(0.0, min(100.0, score))
        return score
    return 0.0


def merge_two(text1, text2, time1, time2):
    prompt = f"""You are a medical documentation editor. Below are two adjacent surgical clip descriptions that describe highly similar surgical actions. Merge them into ONE coherent paragraph, preserving all meaningful details and chronological order. Do NOT add new information. Do NOT output a step-by-step list. Output only one paragraph.

Clip 1 ({time1}):
{text1}

Clip 2 ({time2}):
{text2}

Merged description:"""
    return qwen_generate(prompt, max_new_tokens=2048)