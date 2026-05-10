"""
03_skill_multimodal.py — Text → Image → Text, three different providers.

  1. text  → text   Claude  (claude-sonnet-4-6)
  2. text  → image  Grok    (grok-imagine-image-pro)
  3. image → text   Qwen    (qwen-vl-max)

Required env vars:
    ANTHROPIC_API_KEY
    XAI_API_KEY
    DASHSCOPE_API_KEY
"""

import os, sys, base64, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models import Model
from skills import Skill

# ── 1. Text → Text (Claude) ───────────────────────────────────────────────────
text_skill = Skill(
    model = Model("claude-sonnet-4-6", api_key=os.getenv("ANTHROPIC_API_KEY")),
    input = {"messages": [{"role": "user", "parts": ["Describe a sunset in one sentence."]}]},
)

description = text_skill.run()
print(f"[text → text · Claude]\n{description}\n")

# ── 2. Text → Image (Grok) ────────────────────────────────────────────────────
image_skill = Skill(
    model  = Model("grok-imagine-image-pro", api_key=os.getenv("XAI_API_KEY")),
    input  = {"messages": [{"role": "user", "parts": [description]}]},
    output = {"modalities": ["image"], "format": {"type": "image", "size": "1024x1024"}},
)

image    = image_skill.run()
img_path = pathlib.Path(__file__).parent / "output_sunset.png"
img_path.write_bytes(base64.b64decode(image["base64"]))
print(f"[text → image · Grok]\nsaved → {img_path}\n")

# ── 3. Image → Text (Qwen Vision) ────────────────────────────────────────────
vision_skill = Skill(
    model = Model("qwen-vl-max", api_key=os.getenv("DASHSCOPE_API_KEY")),
    input = {
        "messages": [{
            "role": "user",
            "parts": [
                {"type": "image", "source": {"kind": "base64",
                                              "data": image["base64"],
                                              "mime": image["mime_type"]}},
                {"type": "text",  "text": "What do you see in this image?"},
            ],
        }]
    },
)

analysis = vision_skill.run()
print(f"[image → text · Qwen]\n{analysis}")
