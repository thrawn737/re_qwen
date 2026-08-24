import os
import argparse
from openai import OpenAI

# ===== API Config =====
BASE_URL = "https://gpt-agent.cc/v1"
API_KEY = "sk-ZOCrqOc84afoXOG4AHYeaSTvpNELZzHm9RRY4A6ivlh5lQSm"
MODEL_NAME = "claude-sonnet-4-6"

# ===== Read file =====
def read_summary(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

# ===== Call API =====
def consolidate(content):
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )
    
    prompt = f"""
You are a medical documentation editor. Below is a set of surgical descriptions, each tagged with a time range, arranged in chronological order.

Your task is to combine these descriptions into a **chronologically accurate, detailed, and easy-to-read** document that describes the entire surgery.

**STRICT RULES - MUST FOLLOW:**

1. **EVERY action must include a time marker**: For each described surgical action, you MUST include its exact time position using the format `[MM:SS]` or `[HH:MM:SS]`. If the original description has a time range (e.g., "01:23 → 01:45"), present it as `[01:23 → 01:45]`.

2. **DO NOT merge descriptions EXCEPT for temporally adjacent, highly similar events**: 
   - If two adjacent time points describe nearly identical actions with no meaningful change (e.g., "suctioning blood" at 01:23 and "suctioning blood" at 01:24 with no new event), you MAY merge them into a single description with a combined time range: `[01:23 → 01:24]`.
   - If there is ANY meaningful difference in action, instrument, tissue state, or visual change, you MUST keep them separate.

3. **STRICT chronological order**: Maintain the exact temporal sequence. If the surgery goes through states A → B → A, your document MUST describe them as A → B → A. DO NOT reorganize or merge A's together across non-adjacent time points.

4. **DO NOT remove or skip any meaningful event**: Every significant action or change must be represented in the output. Only adjacent near-identical repetitions may be compressed.

5. **Use plain, everyday language**: 
   - Say "knife" or "blade", NOT "scalpel" or "microsurgical blade".
   - Say "forceps" or "grasper", NOT "clamp" or "hemostat".
   - Say "suction tube", NOT "suction aspirator".
   - Say "clear blood" or "clean the area", NOT "evacuate hematoma" or "debride tissue".
   - Say "cutting", "pulling", "burning" instead of "dissection", "retraction", "cauterization".

6. **Output format**: One continuous narrative in English, with time markers clearly embedded. Do NOT use bullet points or numbered lists.

Write the final document directly. Do not add any extra explanations or commentary.

Here are the original descriptions:
---
{content}
---
"""
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a medical documentation editor who specializes in translating surgical terminology into plain, everyday language."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input file path (e.g., ./output/surgery_segments_v1/L01_summary.txt)")
    parser.add_argument("--output", required=True, help="Output file path (e.g., ./output/L01_summary_v1_consolidated.txt)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    print(f"Reading: {args.input}")
    content = read_summary(args.input)
    
    print("Calling API to consolidate...")
    try:
        consolidated = consolidate(content)
    except Exception as e:
        print(f"API call failed: {e}")
        return
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(consolidated)
    
    print(f"Done! Output saved to: {args.output}")

if __name__ == "__main__":
    main()